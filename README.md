# Aura

Attention-based model for handling table semantics

## Setup

Run the following sequence of commands to create and setup a project environment:

```sh
conda create -n aura python=3.12
conda activate aura
conda install python-lsp-server click
pip install lxml transformers torch torchvision torchaudio
```

## Usage

### Read docx files and parse annotations

Run the following command which will read data from `assets/data/raw` and save results to `assets/data/prepared`, one output `json` file per one input `docx` file:

```sh
python -m aura prepare-corpus
```
