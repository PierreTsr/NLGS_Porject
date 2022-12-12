# Production of sound devices in poetry generation

The aim of this project is to work on the automation of sound-device creation through both fine-tuning and constrained-decoding.

## Setup

The code has been developped with `python=3.10`, `cuda=11.7`, `torch=1.13` on Ubuntu 22.04-LTS and Debian 11 environments. It comes with no guarantee to run on a different setup. To setup the envrionment, simply create a new environment on a CUDA-11.7 enabled machine and run from the project root:

```{bash} 
python -m pip install requirements.txt
export PYTHONPATH=.:$PYTHONPATH
```
> The second command needs to be excuted after every reboot.

## Data

This project relies on the CMU dictionary and the Gutenberg Poetry Corpus. To download both datasets run:
> First make sure the apt package `unzip` is installed
```{bash}
src/cmu_tools/get_cmu.sh
src/gutenberg_tools/get_gutenberg.sh
```

The formatted dataset can then be computed with the script `src/prepocess_dataset.py` depending on the desired configuration. Here are the configuration used for my experiments:
> It is possible to create a new train/test/val split with the flag `--new_split` or to omit it to use the one from this repository

```{bash}
// Training dataset
python src/preprocess_dataset.py gpt2 -p -m 8 -n 8 -s 8 -l 128 --workers 4

// Filtered Validation dataset
python src/preprocess_dataset.py gpt2 -p -m 8 --workers 4
python src/select_clean_prompts.py -d data/datasets/gutenberg_pronunciation_8 -m iambic_pentameter -t 0 --dest data/datasets/pentameter_prompts
python src/select_clean_prompts.py -d data/datasets/gutenberg_pronunciation_8 -m mixed_meter -t 0 --dest data/datasets/mixed_meter_prompts
```

## Experiments configuration

### Training

Two training scripts are available:
1. fine-tuning a GPT-Neo model from the off-the-shelf pretrained model (baseline);
2. training a pronunciation aware GPT-Neo using a fine-tuned or an off-the-shelf model;


The training configurations can be found under `etc/config/fine_tuning_*`. The parameters documentation is available by adding the flag `--help` after the deired script. You can either chose the path to a local model saved with HuggingFace's Transformers or chose a model available online. In particular, the pronunciation-aware model can be built from both a vanilla or a fine-tuned model.

The supported off-the-shelf models are:
- `EleutherAI/gpt-neo-125M`
- `EleutherAI/gpt-neo-1.3B`
- `EleutherAI/gpt-neo-2.7B`

To start training the models run

```{bash}
// Fine-tune GPT-Neo (baseline)
python src/fine_tune_vanilla.py etc/config/fine_tune_vanilla.json

// Train Pronunciation Aware GPT-Neo
python src/fine_tune_custom.py etc/config/fine_tune_custom.json
```

Depending on the chosen base-model the expected required VRAM is roughly:
- **Gpt-Neo-125M** 1Gb
- **Gpt-Neo-1.3B** 13Gb
- **Gpt-Neo-2.7B** 22Gb

The results of each experiment are stored in the directory specified in the provided configuration. To load a trained model use the Transformers' API and use the path to the desired checkpoint.

### Evaluating

The evaluation is done by generating samplings and measuring metrics and statistics on them. Currenlty, the only available sampling script generates quatrains from a single verse prompt. The configuration files for those experiments are found under `etc/config/evaluation_config_vanilla.json`. Two scripts are available:
1. Using simple beam-search;
2. Using A-star like decoding, guiding the model to introduce more sound devices;

To run those evalauations, run:

```{bash}
// Standard decoding
python src/evaluate_model.py etc/config/evaluation_config_vanilla.json

// Soft-constrained decoding
python src/evaluate_model_constrained_decoding.py etc/config/evaluation_config_constrained.json
```
The statistics are printed in the terminal, and the generations are saved in the provided directory if specified in the configuration. The statistics are also added below the genrations.

It is also possible to evaluate the same metrics directly on a kept-back part of the training set using the following command. The resutls are printed directly in the terminal.

```{bash}
python src/evaluate_ref.py
```

Finally, it is possible to run 2 Weight and Biase sweeps (values are hard-coded in the scripts). 
- The first one runs a set of trained models over two sets of prompts;
- The second one is a hyperparameter sweep for the constrained-decoding task;

To run them, simply run each of the two evaluation scripts above without any arguments or flag. The script are using the default configuration `etc/config/evaluation_config_vanilla.json`.
