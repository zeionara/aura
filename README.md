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

## Annotation report generation workflow

1. Download annotated `docx` files and put them to folder `$HOME/Document/PhD/$version`
2. Upload to remote server: `phdta $version` (`phdta` is a `bash` function implemented [here](https://github.com/zeionara/shell/blob/71eb0fba4cd0d4af3d4c505d85f71868d88d20c8/phd.sh#L1))
3. On remote server parse annotations: `phdga $version` and generate report
4. Download report from remote server: `phdda $version`, the report will be located at `$HOME/Documents/PhD/$version.docx`
5. Generate flat embeddings:

```sh
python -m aura embed assets/data/$version/prepared assets/data/$version/prepared -m DeepPavlov/rubert-base-cased
python -m aura embed assets/data/$version/prepared assets/data/$version/prepared -m intfloat/multilingual-e5-large-instruct
python -m aura embed assets/data/$version/prepared assets/data/$version/prepared -m Qwen/Qwen3-Embedding-0.6B
```

6. Train hierarchical attention head:

```sh
python -m aura train assets/data/$version/prepared assets/weights/$version.pth -m DeepPavlov/rubert-base-cased -d 768
```

Example of training log:

```sh
Reading СП 450.1325800.2019.json...
Reading СП 125.13330.2012.json...
Reading СП 474.1325800.2019.json...
Reading СП 497.1325800.2020.json...
Reading СП 302.1325800.2017.json...
Reading СП 364.1311500.2018.json...
Reading СП 506.1311500.2021.json...

Training...

Average epoch loss: 0.2367
Average epoch loss: 0.1222
Average epoch loss: 0.1048
Average epoch loss: 0.1005
Average epoch loss: 0.0980
Average epoch loss: 0.0958
Average epoch loss: 0.0923
Average epoch loss: 0.0876
Average epoch loss: 0.0826
Average epoch loss: 0.0836
Average epoch loss: 0.0818
Average epoch loss: 0.0804
Average epoch loss: 0.0779
Average epoch loss: 0.0738
Average epoch loss: 0.0673
Average epoch loss: 0.0643
Average epoch loss: 0.0599
Average epoch loss: 0.0574
Average epoch loss: 0.0552
Average epoch loss: 0.0528

Saved as assets/weights/2025.11.07.02.pth
```

7. Generate structured embeddings

```sh
python -m aura embed assets/data/$version/prepared assets/data/$version/prepared -p assets/weights/$version.pth -a structured -m DeepPavlov/rubert-base-cased -d 768
```

8. Evaluate

```sh
python -m aura embed assets/data/$version/prepared assets/reports/$version.tsv
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
