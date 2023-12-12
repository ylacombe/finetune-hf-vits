# Finetune VITS and MMS using HuggingFace's tools

## Introduction

[VITS](https://huggingface.co/docs/transformers/model_doc/vits) is a light weight, low-latency model for English text-to-speech (TTS). Massively Multilingual Speech ([MMS](https://huggingface.co/docs/transformers/model_doc/mms#speech-synthesis-tts)) is an extension of VITS for multilingual TTS, that supports over [1100 languages](https://huggingface.co/facebook/mms-tts#supported-languages). 

Both use the same underlying VITS architecture, consisting of a discriminator and a generator for GAN-based training. They differ in their tokenizers: the VITS tokenizer transforms English input text into phonemes, while the MMS tokenizer transforms input text into character-based tokens.

You should fine-tune VITS-based checkpoints if you want to use a permissive English TTS model and fine-tune MMS-based checkpoints for every other cases.

Coupled with the right data and the following training recipe, you can get an excellent finetuned version of every VITS/MMS checkpoints in **20 minutes** with as little as **80 to 150 samples**.

Finetuning VITS or MMS requires multiples stages to be completed in successive order:

1. [Install requirements](#1-requirements)
2. [Choose or create the initial model](#2-model-selection)
3. [Finetune the model](#3-finetuning)
4. [Optional - how to use the finetuned model](#4-inference)


TODO: 
- Add automatic space deployment?
- Report to wandb
- Mention GPU usage and so on
- Mention Spaces

## License
The VITS checkpoints are released under the permissive [MIT License](https://opensource.org/license/mit/). The MMS checkpoints, on the other hand, are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), a non-commercial license. 

**Note:** Any finetuned models derived from these checkpoints will inherit the same licenses as their respective base models. **Please ensure that you comply with the terms of the applicable license when using or distributing these models.**
    

## 1. Requirements


0. Install common requirements.

```sh
pip install -r requirements.txt
```

1. Link your Hugging Face account so that you can pull/push model repositories on the Hub. This will allow you to save the finetuned weights on the Hub so that you can share them with the community and reuse them easily. Run the command:

```bash
git config --global credential.helper store
huggingface-cli login
```
And then enter an authentication token from https://huggingface.co/settings/tokens. Create a new token if you do not have one already. You should make sure that this token has "write" privileges.


2. Build the monotonic alignment search function using cython. This is absolutely necessary since the Python-native-version is awfully slow.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
```


3. (**Optional**) If you're using an **original VITS checkpoint**, as opposed to MMS checkpoints, install **phonemizer**.

Follow steps indicated [here](https://bootphon.github.io/phonemizer/install.html).

<details>
  <summary>Open for an example on Debian/Unbuntu </summary>

E.g, if you're on Debian/Unbuntu:
```sh
# Install dependencies
sudo apt-get install festival espeak-ng mbrola
# Install phonemizer
pip install phonemizer
```
</details>

4. (**Optional**) With MMS checkpoints, **some languages** require to install **uroman**.

<details>
  <summary>Open for details </summary>
    
Some languages require to use `uroman` before feeding the text to `VitsTokenizer`, since currently the tokenizer does not support performing the pre-processing itself.

To do this, you need to clone the uroman repository to your local machine and set the bash variable UROMAN to the local path:

```sh
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```

The rest is taking care of by the training script. Don't forget to adapt the inference snippet as indicated [here](#use-the-finetuned-models).

</details>

## 2. Model selection

There are two options:

**Option 1: a training checkpoint is already available**

In that case, you're lucky and you pass directly to the next step 🤗.

**Option 2: no training checkpoint is available for your language**

Let's say that you want have a text-to-speech dataset in Ghari, a Malayo-Polynesian language. First identify if there is a MMS checkpoint trained on this language by searching for the language in the [MMS Language Coverage Overview](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html). If it is TTS-supported, identify the ISO 693-3 language code, here `gri`.

Contrary to inference, finetuning requires the use of a discriminator that needs to be converted. 
So you want to first creates a new checkpoint with this converted discriminator.

0. (Do once) - create a new checkpoint that includes the discriminator. See [here](#convert-a-discriminator-checkpoint) for more details on how to convert the discriminator.
In the following steps, replace `gri` with the language code you identified [here](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html) and DISCRIMINATOR_TEMPORARY_LOCATION with where you want to download the weights.

- Download the original discriminator weights locally.  
```sh
cd DISCRIMINATOR_TEMPORARY_LOCATION
wget https://huggingface.co/facebook/mms-tts/resolve/main/full_models/gri/D_100000.pth?download=true -O "gri_D_100000.pth"
```
- Now convert the weights, and optionally push them to the hub. Simply remove `--push_to_hub TRAIN_CHECKPOINT_NAME` if you don't want to push to the hub:
```sh
cd PATH_TO_THIS_REPO
python convert_original_discriminator_checkpoint.py --checkpoint_path PATH_TO_gri_D_10000.pth --generator_checkpoint_path "facebook/mms-tts-gri" --pytorch_dump_folder_path LOCAL_PATH_WHERE_TO_STORE_CHECKPOINT
--push_to_hub TRAIN_CHECKPOINT_NAME
```

---------------------

## TL;DR pointers

First and foremost, [install everything](#installation-steps).

----------------------

<details>
  <summary>Open if you want to finetune an English model </summary>

  1. Update this [English configuration template](training_config_examples/finetune_english.json) by:
  * updating the `project_name` and the output artefacts (`hub_model_id`, `output_dir`) to keep track of the model.
  
  * updating `model_name_or_path` in the config with one of the following checkpoints: 
    - `ylacombe/vits-ljs-train` (make sure [the phonemizer package is installed](https://bootphon.github.io/phonemizer/install.html)) - ideal for monolingual finetuning
    - `ylacombe/vits_vctk_train` (make sure [the phonemizer package is installed](https://bootphon.github.io/phonemizer/install.html)) - ideal for multispeaker English finetuning.
    - `ylacombe/mms-tts-eng-train` - if you want to avoid the use of the `phonemizer` package.
  * selecting the dataset you want to finetune on and update the config, e.g the dataset by default in [`finetune_english.json`](training_config_examples/finetune_english.json) is a [British Isles accents dataset](https://huggingface.co/datasets/ylacombe/english_dialects):
    - Make particular attention to the `dataset_name`, `dataset_config_name`, column names.
    - If there are multiple speakers and you want to only keep one, be careful to `speaker_id_column_name`, `override_speaker_embeddings` and `filter_on_speaker_id`. The latter allows to keep only one speaker but you can also train on multiple speakers.

  * (Optional - ) changing hyperparameters at your convenience.  
  
  2. Launch training:

```sh
accelerate launch run_vits_finetuning.py ./training_config_examples/finetune_english.json
```

  3. Use your [finetuned model](#use-the-finetuned-models)

</details>

---------------------

<details>
  <summary>Open if you want to finetune on another language using MMS checkpoints</summary>

  There are two options:

  **Option 1: a training checkpoint is already available**

<details>
  <summary>Open for details </summary>

  1. Update this [configuration template](training_config_examples/finetune_mms.json) by:
  * updating the `project_name` and the output artefacts (`hub_model_id`, `output_dir`) to keep track of the model.
  
  * updating `model_name_or_path` in the config with the already existing checkpoint (e.g `"ylacombe/mms-tts-guj-train"`). 
  * selecting the dataset you want to finetune on and update the config, e.g the dataset by default in [`finetune_mms.json`](training_config_examples/finetune_mms.json) is a [Gujarati dataset](https://huggingface.co/datasets/ylacombe/google-gujarati):
    - Make particular attention to the `dataset_name`, `dataset_config_name`, column names.
    - If there are multiple speakers and you want to only keep one, be careful to `speaker_id_column_name`, `override_speaker_embeddings` and `filter_on_speaker_id`. The latter allows to keep only one speaker but you can also train on multiple speakers.

  * (Optional - ) changing hyperparameters at your convenience.  
  
  2. Launch training:

```sh
accelerate launch run_vits_finetuning.py ./training_config_examples/finetune_mms.json
```

  3. Use your [finetuned model](#use-the-finetuned-models)

</details>

  **Option 2: no training checkpoint is available for your language**
<details> 
  <summary>Open for details steps</summary>
    


1. Update this [configuration template](training_config_examples/finetune_mms.json) by:
* updating the `project_name` and the output artefacts (`hub_model_id`, `output_dir`) to keep track of the model.

* updating `model_name_or_path` in the config with the checkpoint you just created (e.g `LOCAL_PATH_WHERE_TO_STORE_CHECKPOINT` or the hub repo id `TRAIN_CHECKPOINT_NAME`). 
* selecting the dataset you want to finetune on and update the config, e.g the dataset by default in [`finetune_mms.json`](training_config_examples/finetune_mms.json) is a [Gujarati dataset](https://huggingface.co/datasets/ylacombe/google-gujarati). With our example, it would be a Ghari dataset.
- Make particular attention to the `dataset_name`, `dataset_config_name` and column names.
- If there are multiple speakers and you want to only keep one, be careful to `speaker_id_column_name`, `override_speaker_embeddings` and `filter_on_speaker_id`. The latter allows to keep only one speaker but you can also train on multiple speakers.

* (Optional - ) changing hyperparameters at your convenience.  

2. Launch training:

```sh
accelerate launch run_vits_finetuning.py ./training_config_examples/finetune_mms.json
```

3. Use your finetuned model </details>

</details>

-----------------------------




## Finetune VITS and MMS

There are two ways to run the finetuning scrip, both using command lines. Note that you only need one GPU to finetune VITS/MMS as the models are really lightweight (83M parameters).

**Preferred way: use a json config file**

 > [!NOTE]
> Using a config file is the prefered way to use the finetuning script as it includes the most important parameters to consider. For a full list of parameters, run `python run_vits_finetuning.py --help`. Note that some parameters are not ignored by the training script.


The [training_config_examples](./training_config_examples) folder hosts examples of config files. Once satisfied with your config file, you can then finetune the model:

```sh
accelerate launch run_vits_finetuning.py PATH_TO_CONFIG
```

**Other option: pass parameters directly to the command line.**

For example:

```sh
accelerate launch run_vits_finetuning.py --model_name_or_path MODEL_NAME_OR_PATH --output_dir OUTPUT_DIR ...
```

**Important parameters to consider:**
* Everything related to artefacts: the `project_name` and the output directories (`hub_model_id`, `output_dir`) to keep track of the model.
* The model to finetune: `model_name_or_path`
* The dataset used `dataset_name` and its details: `dataset_config_name`, column names, etc. If there are multiple speakers and you want to only keep one, be careful to `speaker_id_column_name`, `override_speaker_embeddings` and `filter_on_speaker_id`. The latter allows to keep only one speaker but you can also train on multiple speakers.
* The most important hyperparameters
   - `learning_rate`
   - `batch_size`
   - the different losses weights: weight_duration, weight_kl, weight_mel, weight_disc, weight_gen, weight_fmaps



## Use the finetuned models

You can use a finetuned model via the Text-to-Speech (TTS) [pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline) in just a few lines of code!
Just replace `ylacombe/vits_ljs_welsh_female_monospeaker_2` with your own model id (`hub_model_id`) or path to the model (`output_dir`).

```python
from transformers import pipeline
import scipy

model_id = "ylacombe/vits_ljs_welsh_female_monospeaker_2"
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

speech = synthesiser("Hello, my dog is cooler than you!")

scipy.io.wavfile.write("finetuned_output.wav", rate=speech["sampling_rate"], data=speech["audio"])
```

Note that if your model needs to use `uroman` to train, you also should apply the uroman package to your text inputs prior to passing them to the pipeline:

```python
import os
import subprocess
from transformers import pipeline
import scipy

model_id = "facebook/mms-tts-kor"
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

def uromanize(input_string, uroman_path):
    """Convert non-Roman strings to Roman using the `uroman` perl package."""
    script_path = os.path.join(uroman_path, "bin", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Execute the perl command
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    # Return the output as a string and skip the new-line character at the end
    return stdout.decode()[:-1]

text = "이봐 무슨 일이야"
uromanized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

speech = synthesiser(uromanized_text)

scipy.io.wavfile.write("finetuned_output.wav", rate=speech["sampling_rate"], data=speech["audio"])
```

## Acknowledgements

It was proposed in 2021, in [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) by Jaehyeon Kim, Jungil Kong, Juhee Son. 

[VITS](https://huggingface.co/docs/transformers/model_doc/vits) is a light weight, low-latency TTS model.

Massively Multilingual Speech ([MMS](https://arxiv.org/abs/2305.13516)) models are light-weight, low-latency TTS models based on the [VITS architecture](https://huggingface.co/docs/transformers/model_doc/vits). They support [1107 languages](https://huggingface.co/facebook/mms-tts#supported-languages). You can find more details about the supported languages and their ISO 639-3 codes in the [MMS Language Coverage Overview](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html),
and see all MMS-TTS checkpoints on the Hugging Face Hub: [facebook/mms-tts](https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts).
