# Finetune VITS and MMS using HuggingFace's tools

[VITS](https://huggingface.co/docs/transformers/model_doc/vits) is a light weight, low-latency TTS model. 

Massively Multilingual Speech (MMS) models are light-weight, low-latency TTS models based on the [VITS architecture](https://huggingface.co/docs/transformers/model_doc/vits).
Meta's [MMS](https://arxiv.org/abs/2305.13516) project, aiming to provide speech technology across a diverse range of languages. You can find more details about the supported languages and their ISO 639-3 codes in the [MMS Language Coverage Overview](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html),
and see all MMS-TTS checkpoints on the Hugging Face Hub: [facebook/mms-tts](https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts).
    
Coupled with the right data and the right training recipe, you can get an excellent finetuned version of every MMS checkpoints in **20 minutes** with as little as **80 to 150 samples**.    


> [!NOTE]
> TODO: Note on license!


## Pre-requisites
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
```
