# Aura

Attention-based model for handling table semantics

## Setup

Run the following sequence of commands to create and setup a project environment:

```sh
conda create -n aura python=3.12
conda activate aura
conda install python-lsp-server click
pip install lxml transformers torch torchvision torchaudio pandas scikit-learn
```

## Usage

### Prepare input documents

Put annotated `.docx` files in the `assets/data/raw` folder.

### Read docx files and parse annotations

Run the following command which will read data from `assets/data/raw` and save results to `assets/data/prepared`, one output `json` file per one input `docx` file:

```sh
python -m aura prepare
```

### Generate flat embeddings

To generate embeddings for paragraphs and separate cells run the following command which takes as input embedder type and base model:

```sh
python -m aura embed -m Qwen/Qwen3-Embedding-0.6B
```

### Generate structured embeddings

To generate embeddings that would preserve information about table structure, run the following command which takes as input embedder type and base model:

```sh
python -m aura embed -m Qwen/Qwen3-Embedding-0.6B -a structured
```
