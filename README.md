# Typological Similarity in Transformer-Based Cross-Lingual Dependency Parsing

This repository contains the code and LaTeX source for my MSc Artificial Intelligence thesis (VU Amsterdam, 2025). The thesis investigates whether typological similarity between languages predicts the success of cross-lingual transfer in dependency parsing. An M-BERT-based biaffine parser is fine-tuned in three stages: first on German, then on one of four auxiliary languages (Dutch, Norwegian, Czech, Finnish), and finally on 50 sentences of a low-resource language that M-BERT was not trained on (Swiss German, Faroese, Upper Sorbian, or Veps). Layer-wise probes and t-SNE visualizations are used to inspect how fine-tuning reorganizes the model's embedding space.

The parser is a reimplementation of the UDify architecture (Kondratyuk & Straka, 2019), files adapted from the original implementation say so in a header.

## Repository structure

- `latex/`: source of the thesis itself
- `python/thesis/src/`: everything needed to train and evaluate models
  - `builders/`: construction of the vocabulary, data loaders, model, and trainer
  - `callbacks/`: trainer callbacks, most importantly staged unfreezing of the BERT layers
  - `config/`: configuration logic, all experiment settings live in `config.yaml`
  - `dataset_readers/`: CoNLL-U reading, adapted from UDify
  - `models/` and `modules/`: the parser components (biaffine decoder, scalar mix)
  - `train.py`, `evaluate.py`, `probe.py`: entry points
- `python/thesis/scripts/`: treebank download and splitting
- `python/thesis/plots/`: scripts that produce the figures in the thesis

## Environment

The code targets AllenNLP, which has been archived since. It will not run on current versions of torch, so use the pinned versions:

```
pip install -r requirements.txt
```

Last verified August 2026 on Python 3.10: the pinned set installs, and all three entry points run. Python 3.11 or newer will not work.

## Data

`scripts/download_data.sh` fetches the four low-resource treebanks (Swiss German, Faroese, Upper Sorbian, Veps) and calls `split_datasets.py` to create their splits. The other treebanks are streamed from Hugging Face at run time. Sampling is seeded, so the splits are reproducible. Note that the script sources a virtual environment at a hard-coded relative path, either create your venv there or run the downloaders and `split_datasets.py` by hand.

Before training, build the shared vocabulary once. From `python/thesis`:

```
PYTHONPATH=src python scripts/build_vocab.py
```

This reads all treebanks (triggering the Hugging Face downloads) and writes `master_vocab/`, which `train.py` expects to find in the working directory.

## Usage

Run everything from `python/thesis`, in this order: install the pinned requirements, run the data script, build the vocabulary, then train.

Every experiment is a key in `src/config/config.yaml`. Training the German base model, for example:

```
python src/train.py --config first_stage_deu
```

Any value from the config can be overridden on the command line:

```
python src/train.py --config first_stage_deu --seed=42
```

`evaluate.py` works the same way. `probe.py` trains the layer-wise language identification probes on a trained model. Outputs (weights, vocabulary, metrics, and the resolved config) are written to the `output_dir` set in the config.

## A note on reproducibility

All experiments in the thesis were run on the DAS-5 cluster, one seed per configuration. Seeds for python, numpy, and torch are set from the config, with cuDNN in deterministic mode. Exact hyperparameters are listed in the thesis appendix, the hyperparameter search itself was done with the W&B sweep defined in `sweep.yaml`.
