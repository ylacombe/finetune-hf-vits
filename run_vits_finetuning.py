"""
Fine-tuning Vits for TTS.
"""

import logging
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import os

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, is_wandb_available, set_seed
from datasets import DatasetDict, load_dataset
from monotonic_align import maximum_path
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    VitsModel
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import send_example_telemetry
from utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, VitsDiscriminator, VitsModelForPreTraining, VitsFeatureExtractor, slice_segments, VitsConfig, uromanize


if is_wandb_available():
    import wandb


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
logger = logging.getLogger(__name__)


#### ARGUMENTS


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    override_speaker_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True` and if `speaker_id_column_name` is specified, it will replace current speaker embeddings with a new set of speaker embeddings."
                "If the model from the checkpoint didn't have speaker embeddings, it will initialize speaker embeddings."
            )
        },
    )

    override_vocabulary_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True`, it will resize the token embeddings based on the vocabulary size of the tokenizer. In other words, use this when you use a different tokenizer than the one that was used during pretraining."
            )
        },
    )


@dataclass
class VITSTrainingArguments(TrainingArguments):
    do_step_schedule_per_epoch: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to perform scheduler steps per epoch or per steps. If `True`, the scheduler will be `ExponentialLR` parametrized with `lr_decay`."
            )
        },
    )

    lr_decay: float = field(
        default=0.999875,
        metadata={"help": "Learning rate decay, used with `ExponentialLR` when `do_step_schedule_per_epoch`."},
    )

    weight_duration: float = field(default=1.0, metadata={"help": "Duration loss weight."})

    weight_kl: float = field(default=1.5, metadata={"help": "KL loss weight."})

    weight_mel: float = field(default=35.0, metadata={"help": "Mel-spectrogram loss weight"})

    weight_disc: float = field(default=3.0, metadata={"help": "Discriminator loss weight"})

    weight_gen: float = field(default=1.0, metadata={"help": "Generator loss weight"})

    weight_fmaps: float = field(default=1.0, metadata={"help": "Feature map loss weight"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    project_name: str = field(
        default="vits_finetuning",
        metadata={"help": "The project name associated to this run. Useful to track your experiment."},
    )
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    speaker_id_column_name: str = field(
        default=None,
        metadata={
            "help": """If set, corresponds to the name of the speaker id column containing the speaker ids.
                If `override_speaker_embeddings=False`:
                    it assumes that speakers are indexed from 0 to `num_speakers-1`.
                    `num_speakers` and `speaker_embedding_size` have to be set in the model config.

                If `override_speaker_embeddings=True`:
                        It will use this column to compute how many speakers there are.

                Defaults to None, i.e it is not used by default."""
        },
    )
    filter_on_speaker_id: int = field(
        default=None,
        metadata={
            "help": (
                "If `speaker_id_column_name` and `filter_on_speaker_id` are set, will filter the dataset to keep a single speaker_id (`filter_on_speaker_id`)  "
            )
        },
    )

    max_tokens_length: float = field(
        default=450,
        metadata={
            "help": ("Truncate audio files with a transcription that are longer than `max_tokens_length` tokens")
        },
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether the input text should be lower cased."},
    )
    do_normalize: bool = field(
        default=False,
        metadata={"help": "Whether the input waveform should be normalized."},
    )
    full_generation_sample_text: str = field(
        default="This is a test, let's see what comes out of this.",
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    uroman_path: str = field(
        default=None,
        metadata={
            "help": (
                "Absolute path to the uroman package. To use if your model requires `uroman`."
                "An easy way to check it is to go on your model card and manually check `is_uroman` in the `tokenizer_config.json,"
                "e.g the French checkpoint doesn't need it: https://huggingface.co/facebook/mms-tts-fra/blob/main/tokenizer_config.json#L4"
            )
        },
    )

# DATA COLLATOR


@dataclass
class DataCollatorTTSWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`VitsTokenizer`])
            The tokenizer used for processing the data.
        feature_extractor ([`VitsFeatureExtractor`])
            The tokenizer used for processing the data.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    tokenizer: Any
    feature_extractor: Any
    forward_attention_mask: bool

    def pad_waveform(self, raw_speech):
        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        # convert into correct format for padding

        padded_inputs = self.feature_extractor.pad(
            batched_speech,
            padding=True,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_features"]

        return padded_inputs

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = "input_ids"
        input_ids = [{model_input_name: feature[model_input_name]} for feature in features]

        # pad input tokens
        batch = self.tokenizer.pad(input_ids, return_tensors="pt", return_attention_mask=self.forward_attention_mask)

        # pad waveform
        waveforms = [np.array(feature["waveform"]) for feature in features]
        batch["waveform"] = self.pad_waveform(waveforms)

        # pad spectrogram
        label_features = [np.array(feature["labels"]) for feature in features]
        labels_batch = self.feature_extractor.pad(
            {"input_features": [i.T for i in label_features]}, return_tensors="pt", return_attention_mask=True
        )

        labels = labels_batch["input_features"].transpose(1, 2)
        batch["labels"] = labels
        batch["labels_attention_mask"] = labels_batch["attention_mask"]

        # pad mel spectrogram
        mel_scaled_input_features = {
            "input_features": [np.array(feature["mel_scaled_input_features"]).squeeze().T for feature in features]
        }
        mel_scaled_input_features = self.feature_extractor.pad(
            mel_scaled_input_features, return_tensors="pt", return_attention_mask=True
        )["input_features"].transpose(1, 2)

        batch["mel_scaled_input_features"] = mel_scaled_input_features
        batch["speaker_id"] = (
            torch.tensor([feature["speaker_id"] for feature in features]) if "speaker_id" in features[0] else None
        )

        return batch


# LOSSES

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    real_losses = 0
    generated_losses = 0
    for disc_real, disc_generated in zip(disc_real_outputs, disc_generated_outputs):
        real_loss = torch.mean((1 - disc_real) ** 2)
        generated_loss = torch.mean(disc_generated**2)
        loss += real_loss + generated_loss
        real_losses += real_loss
        generated_losses += generated_loss

    return loss, real_losses, generated_losses


def feature_loss(feature_maps_real, feature_maps_generated):
    loss = 0
    for feature_map_real, feature_map_generated in zip(feature_maps_real, feature_maps_generated):
        for real, generated in zip(feature_map_real, feature_map_generated):
            real = real.detach()
            loss += torch.mean(torch.abs(real - generated))

    return loss * 2


def generator_loss(disc_outputs):
    total_loss = 0
    gen_losses = []
    for disc_output in disc_outputs:
        disc_output = disc_output
        loss = torch.mean((1 - disc_output) ** 2)
        gen_losses.append(loss)
        total_loss += loss

    return total_loss, gen_losses


def kl_loss(prior_latents, posterior_log_variance, prior_means, prior_log_variance, labels_mask):
    """
    z_p, logs_q: [b, h, t_t]
    prior_means, prior_log_variance: [b, h, t_t]
    """

    kl = prior_log_variance - posterior_log_variance - 0.5
    kl += 0.5 * ((prior_latents - prior_means) ** 2) * torch.exp(-2.0 * prior_log_variance)
    kl = torch.sum(kl * labels_mask)
    loss = kl / torch.sum(labels_mask)
    return loss


# LOGGING AND EVALUATION METHODS


def log_on_trackers(
    trackers,
    generated_audio,
    generated_attn,
    generated_spec,
    target_spec,
    full_generation_waveform,
    epoch,
    sampling_rate,
):
    max_num_samples = min(len(generated_audio), 50)
    generated_audio = generated_audio[:max_num_samples]
    generated_attn = generated_attn[:max_num_samples]
    generated_spec = generated_spec[:max_num_samples]
    target_spec = target_spec[:max_num_samples]

    for tracker in trackers:
        if tracker.name == "tensorboard":
            for cpt, audio in enumerate(generated_audio):
                tracker.writer.add_audio(f"train_step_audio_{cpt}", audio[None, :], epoch, sample_rate=sampling_rate)

            for cpt, audio in enumerate(full_generation_waveform):
                tracker.writer.add_audio(
                    f"full_generation_sample{cpt}", audio[None, :], epoch, sample_rate=sampling_rate
                )

            tracker.writer.add_images("alignements", np.stack(generated_attn), dataformats="NHWC")
            tracker.writer.add_images("spectrogram", np.stack(generated_spec), dataformats="NHWC")
            tracker.writer.add_images("target spectrogram", np.stack(target_spec), dataformats="NHWC")
        elif tracker.name == "wandb":
            # wandb can only loads 100 audios per step
            tracker.log(
                {
                    "alignments": [wandb.Image(attn, caption=f"Audio epoch {epoch}") for attn in generated_attn],
                    "spectrogram": [wandb.Image(spec, caption=f"Audio epoch {epoch}") for spec in generated_spec],
                    "target spectrogram": [wandb.Image(spec, caption=f"Audio epoch {epoch}") for spec in target_spec],
                    "train generated audio": [
                        wandb.Audio(
                            audio[0],
                            caption=f"Audio during train step epoch {epoch}",
                            sample_rate=sampling_rate,
                        )
                        for audio in generated_audio
                    ],
                    "full generations samples": [
                        wandb.Audio(w, caption=f"Full generation sample {epoch}", sample_rate=sampling_rate)
                        for w in full_generation_waveform
                    ],
                }
            )
        else:
            logger.warn(f"audio logging not implemented for {tracker.name}")


def compute_val_metrics_and_losses(
    val_losses,
    accelerator,
    model_outputs,
    mel_scaled_generation,
    mel_scaled_target,
    batch_size,
    compute_clap_similarity=False,
):
    loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
    loss_kl = kl_loss(
        model_outputs.prior_latents,
        model_outputs.posterior_log_variances,
        model_outputs.prior_means,
        model_outputs.prior_log_variances,
        model_outputs.labels_padding_mask,
    )

    losses_mel_kl = loss_mel + loss_kl

    losses = torch.stack([loss_mel, loss_kl, losses_mel_kl])
    losses = accelerator.gather(losses.repeat(batch_size, 1)).mean(0)

    for key, loss in zip(["val_loss_mel", "val_loss_kl", "val_loss_mel_kl"], losses):
        val_losses[key] = val_losses.get(key, 0) + loss.item()

    return val_losses


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VITSTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_vits_finetuning", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if (
        data_args.speaker_id_column_name is not None
        and data_args.speaker_id_column_name not in next(iter(raw_datasets.values())).column_names
    ):
        raise ValueError(
            f"--speaker_id_column_name {data_args.speaker_id_column_name} not found in dataset '{data_args.speaker_id_column_name}'. "
            "Make sure to set `--speaker_id_column_name` to the correct text column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 5. Load config, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = VitsConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    feature_extractor = VitsFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        verbose=False,
    )

    # 6. Resample speech dataset if necessary
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        with training_args.main_process_first(desc="resample"):
            raw_datasets = raw_datasets.cast_column(
                data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
            )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    max_tokens_length = data_args.max_tokens_length
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = tokenizer.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    speaker_id_column_name = data_args.speaker_id_column_name
    filter_on_speaker_id = data_args.filter_on_speaker_id
    do_normalize = data_args.do_normalize
    is_uroman = tokenizer.is_uroman
    uroman_path = None
    if is_uroman:
        uroman_path = data_args.uroman_path if data_args.uroman_path is not None else os.environ.get("UROMAN")
        if uroman_path is None:
            raise ValueError(
            f"The checkpoint that you're using needs the uroman package, but this one is not specified."
            "Make sure to clone the uroman package (`git clone https://github.com/isi-nlp/uroman.git`),"
            "and to set `uroman_path=PATH_TO_UROMAN`."
        )

    num_speakers = config.num_speakers

    # return attention_mask for Vits models
    forward_attention_mask = True

    with training_args.main_process_first(desc="select range of samples"):
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    speaker_id_dict = {}
    new_num_speakers = 0
    if speaker_id_column_name is not None:
        if training_args.do_train:
            # if filter_on_speaker_id, filter so that we keep only the speaker id
            if filter_on_speaker_id is not None:
                with training_args.main_process_first(desc="filter speaker id"):
                    raw_datasets["train"] = raw_datasets["train"].filter(
                        lambda speaker_id: (speaker_id == filter_on_speaker_id),
                        num_proc=num_workers,
                        input_columns=[speaker_id_column_name],
                    )

            with training_args.main_process_first(desc="get speaker id dict"):
                speaker_id_dict = {
                    speaker_id: i for (i, speaker_id) in enumerate(set(raw_datasets["train"][speaker_id_column_name]))
                }
                new_num_speakers = len(speaker_id_dict)

    def prepare_dataset(batch):
        # process target audio
        sample = batch[audio_column_name]
        audio_inputs = feature_extractor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_attention_mask=False,
            do_normalize=do_normalize,
        )

        batch["labels"] = audio_inputs.get("input_features")[0]

        # process text inputs
        input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]
        
        if is_uroman:
            input_str = uromanize(input_str, uroman_path=uroman_path)
        string_inputs = tokenizer(input_str, return_attention_mask=False)

        batch[model_input_name] = string_inputs.get("input_ids")[: max_tokens_length + 1]
        batch["waveform_input_length"] = len(sample["array"])
        batch["tokens_input_length"] = len(batch[model_input_name])
        batch["waveform"] = batch[audio_column_name]["array"]

        batch["mel_scaled_input_features"] = audio_inputs.get("mel_scaled_input_features")[0]

        if speaker_id_column_name is not None:
            if new_num_speakers > 1:
                # align speaker_id to [0, num_speaker_id-1].
                batch["speaker_id"] = speaker_id_dict.get(batch[speaker_id_column_name], 0)
        return batch

    remove_columns = next(iter(raw_datasets.values())).column_names
    if speaker_id_column_name is not None:
        remove_columns = [col for col in remove_columns if col != speaker_id_column_name]

    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length, text):
        length_ = len(length["array"])
        return (length_ > min_input_length and length_ < max_input_length) and text is not None

    with training_args.main_process_first(desc="filter audio lengths"):
        vectorized_datasets = raw_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=[audio_column_name, text_column_name],
        )

    with training_args.main_process_first(desc="dataset map pre-processing"):
        # convert from np.float64 to np.float32
        vectorized_datasets.set_format(type="numpy", columns=[audio_column_name])
        vectorized_datasets = vectorized_datasets.map(
            prepare_dataset,
            remove_columns=remove_columns,
            num_proc=num_workers,
            desc="preprocess train dataset",
        )

    with training_args.main_process_first(desc="filter tokens lengths"):
        vectorized_datasets = vectorized_datasets.filter(
            lambda x: x < data_args.max_tokens_length,
            num_proc=num_workers,
            input_columns=["tokens_input_length"],
        )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # 8. Load pretrained model,
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    model = VitsModelForPreTraining.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    
    with training_args.main_process_first(desc="apply_weight_norm"):
        # apply weight norms
        model.decoder.apply_weight_norm()
        for flow in model.flow.flows:
            torch.nn.utils.weight_norm(flow.conv_pre)
            torch.nn.utils.weight_norm(flow.conv_post)

        # override speaker embeddings if necessary
        if model_args.override_speaker_embeddings and data_args.speaker_id_column_name is not None:
            if new_num_speakers != num_speakers and new_num_speakers > 1:
                speaker_embedding_size = config.speaker_embedding_size if config.speaker_embedding_size > 1 else 256
                logger.info(
                    f"Resize speaker emeddings from {num_speakers} to {new_num_speakers} with embedding size {speaker_embedding_size}."
                )
                model.resize_speaker_embeddings(new_num_speakers, speaker_embedding_size)
            elif new_num_speakers == 1:
                logger.info("Only one speaker detected on the training set. Embeddings are not reinitialized.")
            else:
                logger.info(
                    "Same number of speakers on the new dataset than on the model. Embeddings are not reinitialized."
                )

        # override token embeddings if necessary
        if model_args.override_vocabulary_embeddings:
            new_num_tokens = len(tokenizer)
            model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of=2)

    # 9. Save configs
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorTTSWithPadding(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        forward_attention_mask=forward_attention_mask,
    )

    with training_args.main_process_first():
        input_str = data_args.full_generation_sample_text
        if is_uroman:
            input_str = uromanize(input_str, uroman_path=uroman_path)
        full_generation_sample = tokenizer(input_str, return_tensors="pt")

    # 11. Set up accelerate
    project_name = data_args.project_name
    train_dataset = vectorized_datasets["train"]
    eval_dataset = vectorized_datasets.get("eval", None)

    # inspired from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
    # and https://github.com/huggingface/community-events/blob/main/huggan/pytorch/cyclegan/train.py

    logging_dir = os.path.join(training_args.output_dir, training_args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=training_args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    per_device_train_batch_size = (
        training_args.per_device_train_batch_size if training_args.per_device_train_batch_size else 1
    )
    total_batch_size = (
        per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps
    )

    num_speakers = model.config.num_speakers

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 12. Define train_dataloader and eval_dataloader if relevant
    train_dataloader = None
    if training_args.do_train:
        sampler = (
            LengthGroupedSampler(
                batch_size=per_device_train_batch_size,
                dataset=train_dataset,
                lengths=train_dataset["tokens_input_length"],
            )
            if training_args.group_by_length
            else None
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=not training_args.group_by_length,
            collate_fn=data_collator,
            batch_size=training_args.per_device_train_batch_size,
            num_workers=training_args.dataloader_num_workers,
            sampler=sampler,
        )

    eval_dataloader = None
    if training_args.do_eval:
        eval_sampler = (
            LengthGroupedSampler(
                batch_size=training_args.per_device_eval_batch_size,
                dataset=eval_dataset,
                lengths=eval_dataset["tokens_input_length"],
            )
            if training_args.group_by_length
            else None
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            sampler=eval_sampler,
        )

    model_segment_size = model.segment_size
    config_segment_size = model.config.segment_size
    sampling_rate = model.config.sampling_rate

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps == -1:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

    # hack to be able to train on multiple device
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.discriminator.save_pretrained(tmpdirname)
        discriminator = VitsDiscriminator.from_pretrained(tmpdirname)
        for disc in discriminator.discriminators:
            disc.apply_weight_norm()
    del model.discriminator

    # init gen_optimizer, gen_lr_scheduler, disc_optimizer, dics_lr_scheduler
    gen_optimizer = torch.optim.AdamW(
        model.parameters(),
        training_args.learning_rate,
        betas=[training_args.adam_beta1, training_args.adam_beta2],
        eps=training_args.adam_epsilon,
    )

    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        training_args.learning_rate,
        betas=[training_args.adam_beta1, training_args.adam_beta2],
        eps=training_args.adam_epsilon,
    )

    num_warmups_steps = (
        training_args.get_warmup_steps(training_args.num_train_epochs * accelerator.num_processes)
        if training_args.do_step_schedule_per_epoch
        else training_args.get_warmup_steps(training_args.max_steps * accelerator.num_processes)
    )
    num_training_steps = (
        training_args.num_train_epochs * accelerator.num_processes
        if training_args.do_step_schedule_per_epoch
        else training_args.max_steps * accelerator.num_processes
    )

    if training_args.do_step_schedule_per_epoch:
        gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            gen_optimizer, gamma=training_args.lr_decay, last_epoch=-1
        )
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            disc_optimizer, gamma=training_args.lr_decay, last_epoch=-1
        )
    else:
        gen_lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=gen_optimizer,
            num_warmup_steps=num_warmups_steps if num_warmups_steps > 0 else None,
            num_training_steps=num_training_steps,
        )
        disc_lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=disc_optimizer,
            num_warmup_steps=num_warmups_steps if num_warmups_steps > 0 else None,
            num_training_steps=num_training_steps,
        )

    # Prepare everything with our `accelerator`.
    (
        model,
        discriminator,
        gen_optimizer,
        gen_lr_scheduler,
        disc_optimizer,
        disc_lr_scheduler,
        train_dataloader,
        eval_dataloader,
    ) = accelerator.prepare(
        model,
        discriminator,
        gen_optimizer,
        gen_lr_scheduler,
        disc_optimizer,
        disc_lr_scheduler,
        train_dataloader,
        eval_dataloader,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = training_args.to_sanitized_dict()
        accelerator.init_trackers(project_name, tracker_config)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint != "latest":
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(training_args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{training_args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            training_args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(training_args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, training_args.max_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, training_args.num_train_epochs):
        # keep track of train losses
        train_losses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if training_args.do_step_schedule_per_epoch:
            disc_lr_scheduler.step()
            gen_lr_scheduler.step()

        for step, batch in enumerate(train_dataloader):
            # print(f"batch {step}, process{accelerator.process_index}, waveform {(batch['waveform'].shape)}, tokens {(batch['input_ids'].shape)}... ")
            with accelerator.accumulate(model, discriminator):
                # forward through model
                model_outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    labels_attention_mask=batch["labels_attention_mask"],
                    speaker_id=batch["speaker_id"],
                    return_dict=True,
                    monotonic_alignment_function=maximum_path,
                )

                mel_scaled_labels = batch["mel_scaled_input_features"]
                mel_scaled_target = slice_segments(mel_scaled_labels, model_outputs.ids_slice, model_segment_size)
                mel_scaled_generation = feature_extractor._torch_extract_fbank_features(
                    model_outputs.waveform.squeeze(1)
                )[1]

                target_waveform = batch["waveform"].transpose(1, 2)
                target_waveform = slice_segments(
                    target_waveform, model_outputs.ids_slice * feature_extractor.hop_length, config_segment_size
                )

                # -----------------------
                #  Train Discriminator
                # -----------------------

                discriminator_target, _ = discriminator(target_waveform)
                discriminator_candidate, _ = discriminator(model_outputs.waveform.detach())

                loss_disc, loss_real_disc, loss_fake_disc = discriminator_loss(
                    discriminator_target, discriminator_candidate
                )

                # backpropagate
                accelerator.backward(loss_disc * training_args.weight_disc)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), training_args.max_grad_norm)
                disc_optimizer.step()
                if not training_args.do_step_schedule_per_epoch:
                    disc_lr_scheduler.step()
                disc_optimizer.zero_grad()

                # -----------------------
                #  Train Generator
                # -----------------------

                _, fmaps_target = discriminator(target_waveform)
                discriminator_candidate, fmaps_candidate = discriminator(model_outputs.waveform)

                loss_duration = torch.sum(model_outputs.log_duration)
                loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                loss_kl = kl_loss(
                    model_outputs.prior_latents,
                    model_outputs.posterior_log_variances,
                    model_outputs.prior_means,
                    model_outputs.prior_log_variances,
                    model_outputs.labels_padding_mask,
                )
                loss_fmaps = feature_loss(fmaps_target, fmaps_candidate)
                loss_gen, losses_gen = generator_loss(discriminator_candidate)

                total_generator_loss = (
                    loss_duration * training_args.weight_duration
                    + loss_mel * training_args.weight_mel
                    + loss_kl * training_args.weight_kl
                    + loss_fmaps * training_args.weight_fmaps
                    + loss_gen * training_args.weight_gen
                )

                # backpropagate
                accelerator.backward(total_generator_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                gen_optimizer.step()
                if not training_args.do_step_schedule_per_epoch:
                    gen_lr_scheduler.step()
                gen_optimizer.zero_grad()

                # update and gather losses
                losses = torch.stack(
                    [
                        # for fair comparison, don't use weighted loss
                        loss_duration + loss_mel + loss_kl + loss_fmaps + loss_gen,
                        loss_duration,
                        loss_mel,
                        loss_kl,
                        loss_fmaps,
                        loss_gen,
                        loss_disc,
                        loss_real_disc,
                        loss_fake_disc,
                    ]
                )
                losses = accelerator.gather(losses.repeat(per_device_train_batch_size, 1)).mean(0)

                train_losses = [
                    l + losses[i].item() / training_args.gradient_accumulation_steps
                    for (i, l) in enumerate(train_losses)
                ]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                (
                    train_summed_losses,
                    train_loss_duration,
                    train_loss_mel,
                    train_loss_kl,
                    train_loss_fmaps,
                    train_loss_gen,
                    train_loss_disc,
                    train_loss_real_disc,
                    train_loss_fake_disc,
                ) = train_losses
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_summed_losses": train_summed_losses,
                        "train_loss_disc": train_loss_disc,
                        "train_loss_real_disc": train_loss_real_disc,
                        "train_loss_fake_disc": train_loss_fake_disc,
                        "train_loss_duration": train_loss_duration,
                        "train_loss_mel": train_loss_mel,
                        "train_loss_kl": train_loss_kl,
                        "train_loss_fmaps": train_loss_fmaps,
                        "train_loss_gen": train_loss_gen,
                        "lr": disc_lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
                train_losses = [0.0 for _ in train_losses]

                if global_step % training_args.save_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `save_total_limit`
                        if training_args.save_total_limit is not None:
                            checkpoints = os.listdir(training_args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `save_total_limit - 1` checkpoints
                            if len(checkpoints) >= training_args.save_total_limit:
                                num_to_remove = len(checkpoints) - training_args.save_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(training_args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": total_generator_loss.detach().item(),
                "lr": disc_lr_scheduler.get_last_lr()[0],
                "step_loss_duration": loss_duration.detach().item(),
                "step_loss_mel": loss_mel.detach().item(),
                "step_loss_kl": loss_kl.detach().item(),
                "step_loss_fmaps": loss_fmaps.detach().item(),
                "step_loss_gen": loss_gen.detach().item(),
                "step_loss_disc": loss_disc.detach().item(),
                "step_loss_real_disc": loss_real_disc.detach().item(),
                "step_loss_fake_disc": loss_fake_disc.detach().item(),
            }
            progress_bar.set_postfix(**logs)

            if global_step >= training_args.max_steps:
                break

            eval_steps = training_args.eval_steps if training_args.eval_steps else 1
            do_eval = training_args.do_eval and (global_step % eval_steps == 0) and accelerator.sync_gradients

            if do_eval:
                logger.info("Running validation... ")
                generated_audio = []
                generated_attn = []
                generated_spec = []
                target_spec = []
                val_losses = {}
                for step, batch in enumerate(eval_dataloader):
                    print(
                        f"VALIDATION - batch {step}, process{accelerator.process_index}, waveform {(batch['waveform'].shape)}, tokens {(batch['input_ids'].shape)}... "
                    )
                    with torch.no_grad():
                        model_outputs_train = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            labels_attention_mask=batch["labels_attention_mask"],
                            speaker_id=batch["speaker_id"],
                            return_dict=True,
                            monotonic_alignment_function=maximum_path,
                        )

                        mel_scaled_labels = batch["mel_scaled_input_features"]
                        mel_scaled_target = slice_segments(
                            mel_scaled_labels, model_outputs_train.ids_slice, model_segment_size
                        )
                        mel_scaled_generation = feature_extractor._torch_extract_fbank_features(
                            model_outputs_train.waveform.squeeze(1)
                        )[1]

                        val_losses = compute_val_metrics_and_losses(
                            val_losses,
                            accelerator,
                            model_outputs_train,
                            mel_scaled_generation,
                            mel_scaled_target,
                            per_device_train_batch_size,
                            compute_clap_similarity=False,
                        )

                    print(f"VALIDATION - batch {step}, process{accelerator.process_index}, PADDING AND GATHER... ")
                    specs = feature_extractor._torch_extract_fbank_features(model_outputs_train.waveform.squeeze(1))[0]
                    padded_attn, specs, target_specs = accelerator.pad_across_processes(
                        [model_outputs_train.attn.squeeze(1), specs, batch["labels"]], dim=1
                    )
                    padded_attn, specs, target_specs = accelerator.pad_across_processes(
                        [padded_attn, specs, target_specs], dim=2
                    )

                    generated_train_waveform, padded_attn, specs, target_specs = accelerator.gather_for_metrics(
                        [model_outputs_train.waveform, padded_attn, specs, target_specs]
                    )

                    if accelerator.is_main_process:
                        with torch.no_grad():
                            speaker_id = None if num_speakers < 2 else list(range(min(5, num_speakers)))
                            full_generation = model(**full_generation_sample.to(model.device), speaker_id=speaker_id)

                        generated_audio.append(generated_train_waveform.cpu())
                        generated_attn.append(padded_attn.cpu())
                        generated_spec.append(specs.cpu())
                        target_spec.append(target_specs.cpu())

                logger.info("Validation inference done, now evaluating... ")
                if accelerator.is_main_process:
                    generated_audio = [audio.numpy() for audio_batch in generated_audio for audio in audio_batch]
                    generated_attn = [
                        plot_alignment_to_numpy(attn.numpy()) for attn_batch in generated_attn for attn in attn_batch
                    ]
                    generated_spec = [
                        plot_spectrogram_to_numpy(attn.numpy()) for attn_batch in generated_spec for attn in attn_batch
                    ]
                    target_spec = [
                        plot_spectrogram_to_numpy(attn.numpy()) for attn_batch in target_spec for attn in attn_batch
                    ]
                    full_generation_waveform = full_generation.waveform.cpu().numpy()

                    accelerator.log(val_losses, step=global_step)

                    log_on_trackers(
                        accelerator.trackers,
                        generated_audio,
                        generated_attn,
                        generated_spec,
                        target_spec,
                        full_generation_waveform,
                        epoch,
                        sampling_rate,
                    )

                    logger.info("Validation finished... ")

                accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        epoch = training_args.num_train_epochs if training_args.num_train_epochs else 1
        eval_steps = training_args.eval_steps if training_args.eval_steps else 1

        # Run a final round of inference.
        do_eval = training_args.do_eval

        if do_eval:
            logger.info("Running final validation... ")
            generated_audio = []
            generated_attn = []
            generated_spec = []
            target_spec = []
            val_losses = {}
            for step, batch in enumerate(eval_dataloader):
                print(
                    f"VALIDATION - batch {step}, process{accelerator.process_index}, waveform {(batch['waveform'].shape)}, tokens {(batch['input_ids'].shape)}... "
                )
                with torch.no_grad():
                    model_outputs_train = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        labels_attention_mask=batch["labels_attention_mask"],
                        speaker_id=batch["speaker_id"],
                        return_dict=True,
                        monotonic_alignment_function=maximum_path,
                    )

                    mel_scaled_labels = batch["mel_scaled_input_features"]
                    mel_scaled_target = slice_segments(
                        mel_scaled_labels, model_outputs_train.ids_slice, model_segment_size
                    )
                    mel_scaled_generation = feature_extractor._torch_extract_fbank_features(
                        model_outputs_train.waveform.squeeze(1)
                    )[1]

                    val_losses = compute_val_metrics_and_losses(
                        val_losses,
                        accelerator,
                        model_outputs_train,
                        mel_scaled_generation,
                        mel_scaled_target,
                        per_device_train_batch_size,
                        compute_clap_similarity=False,
                    )
                specs = feature_extractor._torch_extract_fbank_features(model_outputs_train.waveform.squeeze(1))[0]
                padded_attn, specs, target_specs = accelerator.pad_across_processes(
                    [model_outputs_train.attn.squeeze(1), specs, batch["labels"]], dim=1
                )
                padded_attn, specs, target_specs = accelerator.pad_across_processes(
                    [padded_attn, specs, target_specs], dim=2
                )

                generated_train_waveform, padded_attn, specs, target_specs = accelerator.gather_for_metrics(
                    [model_outputs_train.waveform, padded_attn, specs, target_specs]
                )

                if accelerator.is_main_process:
                    with torch.no_grad():
                        speaker_id = None if num_speakers < 2 else list(range(min(5, num_speakers)))
                        full_generation = model(**full_generation_sample.to(model.device), speaker_id=speaker_id)

                    generated_audio.append(generated_train_waveform.cpu())
                    generated_attn.append(padded_attn.cpu())
                    generated_spec.append(specs.cpu())
                    target_spec.append(target_specs.cpu())

            logger.info("Validation inference done, now evaluating... ")
            if accelerator.is_main_process:
                generated_audio = [audio.numpy() for audio_batch in generated_audio for audio in audio_batch]
                generated_attn = [
                    plot_alignment_to_numpy(attn.numpy()) for attn_batch in generated_attn for attn in attn_batch
                ]
                generated_spec = [
                    plot_spectrogram_to_numpy(attn.numpy()) for attn_batch in generated_spec for attn in attn_batch
                ]
                target_spec = [
                    plot_spectrogram_to_numpy(attn.numpy()) for attn_batch in target_spec for attn in attn_batch
                ]
                full_generation_waveform = full_generation.waveform.cpu().numpy()

                log_on_trackers(
                    accelerator.trackers,
                    generated_audio,
                    generated_attn,
                    generated_spec,
                    target_spec,
                    full_generation_waveform,
                    epoch,
                    sampling_rate,
                )

                accelerator.log(val_losses, step=global_step)
                logger.info("Validation finished... ")

            accelerator.wait_for_everyone()

        # unwrap, save and push final model
        model = accelerator.unwrap_model(model)
        discriminator = accelerator.unwrap_model(discriminator)

        model.discriminator = discriminator

        # add weight norms
        for disc in model.discriminator.discriminators:
            disc.remove_weight_norm()
        model.decoder.remove_weight_norm()
        for flow in model.flow.flows:
            torch.nn.utils.remove_weight_norm(flow.conv_pre)
            torch.nn.utils.remove_weight_norm(flow.conv_post)

        model.save_pretrained(training_args.output_dir)

        if training_args.push_to_hub:
            VitsModel.from_pretrained(training_args.output_dir).push_to_hub(training_args.hub_model_id, private = training_args.hub_private_repo, token = training_args.hub_token)


    accelerator.end_training()

    # 13. Push FE and tokenizer
    if training_args.push_to_hub:
        feature_extractor.push_to_hub(training_args.hub_model_id, private = training_args.hub_private_repo, token = training_args.hub_token)
        tokenizer.push_to_hub(training_args.hub_model_id, private = training_args.hub_private_repo, token = training_args.hub_token)


    logger.info("***** Training / Inference Done *****")


if __name__ == "__main__":
    main()
