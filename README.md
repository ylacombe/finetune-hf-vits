# Finetune VITS and MMS using HuggingFace's tools

[VITS](https://huggingface.co/docs/transformers/model_doc/vits) is a light weight, low-latency TTS model.
It was proposed in 2021, in [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) by Jaehyeon Kim, Jungil Kong, Juhee Son. 

Massively Multilingual Speech (MMS) models are light-weight, low-latency TTS models based on the [VITS architecture](https://huggingface.co/docs/transformers/model_doc/vits).

Meta's [MMS](https://arxiv.org/abs/2305.13516) project, aims to provide speech technology across a diverse range of languages. You can find more details about the supported languages and their ISO 639-3 codes in the [MMS Language Coverage Overview](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html),
and see all MMS-TTS checkpoints on the Hugging Face Hub: [facebook/mms-tts](https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts).
    
Coupled with the right data and the right training recipe, you can get an excellent finetuned version of every MMS checkpoints in **20 minutes** with as little as **80 to 150 samples**.

TODO: 
- Uroman - add support and guidance: https://huggingface.co/docs/transformers/v4.36.0/en/model_doc/vits#usage-examples
- 


> [!NOTE]
> VITS is under the MIT license, a permissive license, but MMS checkpoints are under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) a non-commercial license.

---------------------

## TL;DR pointers

First and foremost, [install everything](#installation-steps).

----------------------

<details>
  <summary>Open if you want to finetune an English model </summary>

  1. Update this [English configuration template](training_config_examples/finetune_english.json) by:
  * update the `project_name` and the output artefacts (`hub_model_id`, `output_dir`) to keep track of the model.
  
  * update `model_name_or_path` in the config with one of the following checkpoints: 
    - `ylacombe/vits-ljs-train` (make sure [the phonemizer package is installed](https://bootphon.github.io/phonemizer/install.html)) - ideal for monolingual finetuning
    - `ylacombe/vits_vctk_train` (make sure [the phonemizer package is installed](https://bootphon.github.io/phonemizer/install.html)) - ideal for multispeaker English finetuning.
    - `ylacombe/mms-tts-eng-train` - if you want to avoid the use of the `phonemizer` package.
  * selecting the dataset you want to finetune on and update the config, e.g the dataset by default in [`finetune_english.json`](training_config_examples/finetune_english.json) is a [British Isles accents dataset](https://huggingface.co/datasets/ylacombe/english_dialects):
    - Make particular attention to the `dataset_name`, `dataset_config_name`, column names.
    - If there are multiple speakers and you want to only keep one, be careful to `speaker_id_column_name`, `override_speaker_embeddings` and `filter_on_speaker_id`. The latter allows to keep only one speaker but you can also train on multiple speakers.

  * (Optional - ) changing hyperparameters at your convenience.  
  
  2. Launch training:

```sh
accelerate launch run_vits_finetuning.py ./training_configs/finetune_english.json
```

  3. Use your finetuned model:

  - You can use your model with `output_dir` or `hub_model_id` if you decided to `push_to_the_hub`.
</details>

---------------------

<details>
  <summary>Open if you want to finetune on another language using MMS checkpoints</summary>

  There are two options:

  **Option 1: a training checkpoint is already available**

  0. (Do once) - create a new checkpoint that includes the discriminator.

  Contrary to inference, finetuning requires the use of a discriminator that needs to be converted. 
  So you want to first creates a new checkpoint with this converted discriminator.
  <details> <summary>Open for details steps</summary>
  1. 

  </details>




  1. 

  1. Update this [English configuration template](training_config_examples/finetune_english.json) by:
  
  - selecting one of the following checkpoints and update the config: TODO
  - selecting the dataset you want to finetune on and update the config. E.g 
  - (Optional - ) changin hyperparameters at your convenience.  
  
  2. Launch training:

```sh
accelerate launch run_vits_finetuning.py ./training_configs/finetune_english.json
```
</details>

## Installation steps

0. Install common requirements.

```sh
pip install -r requirements.txt
```

1. Build the monotonic alignment search function using cython. This is absolutely necessary since the Python-native-version is awfully slow.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
```

**Optional steps depending on the checkpoint/language you're using.**

2. (Optional) If you're using an original VITS checkpoint, as opposed to MMS checkpoints, install **phonemizer**.

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

3. (Optional) With MMS checkpoints, some languages require to install **uroman**.

<details>
  <summary>Open for details </summary>

If required, you should apply the uroman package to your text inputs prior to passing them to the VitsTokenizer, since currently the tokenizer does not support performing the pre-processing itself.


To do this, first clone the uroman repository to your local machine and set the bash variable UROMAN to the local path:

```sh
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```
</details>


2. Option: create a new checkpoint
<details>
  <summary>Open for details</summary>

  ### Heading
  1. Foo
  2. Bar
     * Baz
     * Qux

  ### Some Javascript
  ```js
  function logSomething(something) {
    console.log('Something', something);
  }
  ```
</details>

