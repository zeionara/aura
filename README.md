# Aura

Attention-based model for handling table semantics

## Setup

Run the following sequence of commands to create and setup a project environment:

```sh
conda create -n aura python=3.12
conda activate aura
conda install python-lsp-server click
pip install lxml
```

## Usage

### Read docx files and parse annotations

Run the following command which will read data from `assets/data/raw` and log the results:

```sh
python -m aura prepare-corpus
```
