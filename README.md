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

## Automatic annotation

To train table embedding model, you need to annotate some data. The document annotation consists of the following two stages.
As an input, the annotation procedure takes a set of `.docx` documents, and as output, it produces another set of `.docx` documents with comments, which contain paragraph annotation.
Each paragraph annotations contains information about the relevance of a particular paragraph to a document table.

### Generating annotations

To annotate tables using LLM, first, generate the annotations:

```sh
python -m aura annotate assets/data/<version>/source assets/data/<version>/annotations --batch-size 10
```

To handle limited number of data use parameters `--n-batches` and `--n-files` (this command will handle `2` first batches, each including `10` paragraphs for the first `2` files in the provided directory):

```sh
python -m aura annotate assets/data/<version>/source assets/data/<version>/annotations --batch-size 10 --n-batches 2 --n-files 2
```

The command will generate a set of `.json` files in the target directory, each output `.json` file corresponds to an input `.docx` file and structured as follows (the score `0.7` is provided as an example, there score value might be any real number in the interval `[0; 1]`):

```json
{
    "[inferred table name]": {
        "type": "table",
        "paragraphs": [
            {
                "id": "[uuid4]",
                "text": "[paragraph text]",
                "scores": {
                    "[model name]": {
                        "comment": "[assigned score justification]",
                        "score": 0.7
                    }
                }
            },
            ...
        ]
    },

    ...
}
```

The script supports extending previously generated annotations located in the target folder. If required score is already provided, then it will not be calculated again.

### Dataset versions

The full dataset contains **420** files. This dataset was split into parts, and the following versions were generated:

| Dataset ID | Source | Annotations | Manual | Description |
| --- | --- | --- | --- | --- |
| `2025.11.07.02` | Yes | No | Yes | Contains **7** documents annotated manually |
| `sets-of-rules` | Yes | No | No | Contains **108** documents which made up the first large batch passed to `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| `2025.11.07.02` | No | Yes | No | Includes annotations for **119** files with the lowest size from the original dataset. This dataset is corrupted, because it contains annotations from multiple `source` datasets (`sets-of-rules`, `2026.01.22.01`, `2026.02.01.01`, and the full dataset) |
| `2026.01.22.01` | Yes | Yes | No | Contains **9** documents excluded from `sets-of-rules` due to the table size, which resulted in the prompt containing too much text (this problem was solved by deleting empty cells from large tables) |
| `2026.01.24.01` | Yes | Yes | No | The improved version of `2025.11.07.02`, which consists of **117** documents (`sets-of-rules` + `2026.01.22.01`) and includes only files, from which there is a source in `sets-of-rules` |
| `2026.01.24.02` | Yes | Yes | No | Contains **80** documents which made up the second large batched passed to `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| `2026.02.01.01` | Yes | Yes | No | Contains **8** documents, which were originally excluded from the `sets-of-rules` due to unconventional table naming patterns |
| `2026.02.01.02` | Yes | Yes | No | Contains **197** documents, which result from merging `2026.01.24.01` and `2026.01.24.02` |
| `2026.02.01.03` | Yes | Yes | No | Contains the remaining **210** documents from the full dataset |

### Applying annotations

Then apply these annotations:

```sh
python -m aura apply assets/data/<version>/source assets/data/<version>/annotations assets/data/<version>/raw --threshold 0.5
```

## Embedding model training

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
