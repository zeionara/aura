from os import walk, path as os_path, mkdir, getenv
from json import dump, load
from logging import getLogger
from torch import optim

from click import group, argument, option

from .util import read_elements, make_system_prompt, dict_from_json_file
from .document import Paragraph, Table, INDENT
from .document.ZipFile import ZipFile
from .embedder import EmbedderType, BaseModel, FlatEmbedder, StructuredEmbedder
from .evaluation import evaluate as run_evaluation, average
from .embedder.AttentionTableEmbedder import DEFAULT_INPUT_DIM
from .Annotator import Annotator, TABLE_TITLE_PATTERN

from .prepare import prepare as prepare_impl


RAW_DATA_PATH = 'assets/data/raw'
SOURCE_DATA_PATH = 'assets/data/source'
ANNOTATIONS_PATH = 'assets/data/annotations'
PREPARED_DATA_PATH = 'assets/data/prepared'
REPORT_PATH = 'assets/data/report.tsv'
ANNOTATIONS_DOCUMENT_PATH = 'assets/data/annotations.docx'
MODEL_PARAMS_PATH = None  # 'assets/weights.pth'

DEFAULT_TRAIN_FRACTION = 0.6
DEFAULT_SEED = 17


logger = getLogger(__name__)


@group()
def main():
    pass


def aggregate(**annotations: dict):
    scores = []
    comment = None

    for annotation in annotations.values():
        scores.append(annotation['score'])

        if comment is None or len(annotation['comment']) > len(comment):
            comment = annotation['comment']

    return sum(scores) / len(scores), comment


@main.command()
@argument('input-path', type = str, default = SOURCE_DATA_PATH)
@argument('annotations-path', type = str, default = ANNOTATIONS_PATH)
@argument('output-path', type = str, default = RAW_DATA_PATH)
@option('--threshold', type = float, default = 0.5)
@option('--table-label-search-window', type = int, default = 5)
def apply(input_path: str, annotations_path: str, output_path: str, threshold: float, table_label_search_window: int):
    if not os_path.isdir(output_path):
        mkdir(output_path)

    for root, _, files in walk(input_path):
        for file in files:
            if not file.endswith('.docx'):
                continue

            annotations_file = file[:-5] + '.json'
            output_file = ZipFile(os_path.join(root, file))

            annotations = dict_from_json_file(
                os_path.join(annotations_path, annotations_file)
            )

            elements = read_elements(os_path.join(root, file))

            comment_id = 0

            tables = []
            paragraphs = []

            table_label_cahdidates = []

            for element in elements:
                if element.tag.endswith('}p'):
                    paragraph = Paragraph.from_xml(element)

                    if paragraph:
                        paragraphs.append(paragraph)
                        table_label_cahdidates.append(paragraph.text)
                else:
                    i = 0
                    label = None

                    while i < table_label_search_window:
                        title = table_label_cahdidates.pop().lower()
                        i += 1

                        match = TABLE_TITLE_PATTERN.match(title)

                        if match is not None:
                            label = match.group(1)
                            break

                    if label is None:
                        logger.error('Couldn\'t find table label')

                    table_label_search_window = []

                    table = Table.from_xml(element, label = label)

                    tables.append(table)

            for table in tables:
                output_file.insert_comment(table.xml, table.label, comment_id = comment_id, tag = 'tbl')
                comment_id += 1

                if (n_paragraph_annotations := len(paragraph_annotations := annotations[table.label]['paragraphs'])) != (n_paragraphs := len(paragraphs)):
                    logger.warning(f'Paragraph annotations count does not match the number of paragraphs ({n_paragraph_annotations} != {n_paragraphs})')

                for paragraph_annotation, paragraph in zip(paragraph_annotations, paragraphs):
                    score, comment = aggregate(**paragraph_annotation['scores'])

                    if score > threshold:
                        output_file.insert_comment(
                            paragraph.xml,
                            f'{table.label} {score:.3f}' if comment is None else f'{table.label} {score:.3f} {comment}',
                            comment_id = comment_id
                        )
                        comment_id += 1

            output_file.save(os_path.join(output_path, file))


@main.command()
@argument('input-path', type = str, default = RAW_DATA_PATH)
@argument('output-path', type = str, default = ANNOTATIONS_PATH)
@option('--batch-size', '-b', type = int, default = None)
@option('--n-batches', '-n', type = int, default = None)
@option('--ckpt-period', '-c', type = int, default = 2)
def annotate(input_path: str, output_path: str, batch_size: int, n_batches: int, ckpt_period: int):
    annotator = Annotator(
        llm_configs = [
            {
                'host': getenv('AURA_VLLM_HOST'),
                'port': int(getenv('AURA_VLLM_PORT')),
                'model': getenv('AURA_VLLM_MODEL'),
                'system_prompt': make_system_prompt(),
                'label': getenv('AURA_VLLM_LABEL')
            },
            # VllmClient(
            #     getenv('AURA_VLLM_HOST'),
            #     int(getenv('AURA_VLLM_PORT')),
            #     getenv('AURA_VLLM_MODEL'),
            #     make_system_prompt(),
            #     label = getenv('AURA_VLLM_LABEL') + "-duplicate"
            # ),
            # GigaChatClient(
            #     getenv('AURA_GIGACHAT_AUTHORIZATION_KEY'),
            #     GigaChatModel(getenv('AURA_GIGACHAT_MODEL')),
            #     make_system_prompt(),
            #     label = getenv('AURA_GIGACHAT_LABEL')
            # )
        ]
    )

    annotator.annotate(input_path, output_path, batch_size, n_batches, ckpt_period = ckpt_period)


@main.command()
@argument('input-path', type = str, default = PREPARED_DATA_PATH)
@argument('output-path', type = str, default = MODEL_PARAMS_PATH)
@option('--model', '-m', type = BaseModel, default = BaseModel.E5_LARGE)
@option('--cpu', '-c', is_flag = True)
@option('--input-dim', '-d', type = int, default = DEFAULT_INPUT_DIM)
def train(input_path: str, output_path: str, model: BaseModel, cpu: bool, input_dim: int):
    embedder = StructuredEmbedder(model, cuda = not cpu, input_dim = input_dim)

    weights_directory = os_path.dirname(output_path)

    if not os_path.isdir(weights_directory):
        mkdir(weights_directory)

    elements = []

    for root, _, files in walk(input_path):
        for filename in files:
            if not filename.endswith('json'):
                continue

            print(f'Reading {filename}...')

            with open(os_path.join(root, filename), 'r') as file:
                data = load(file)

            elements.extend(data['elements'])

    print()
    print('Training...')
    print()

    optimizer = optim.Adam(embedder.model.parameters(), lr=1e-4)
    embedder.train(elements, optimizer)

    embedder.save(output_path)

    print()
    print(f'Saved as {output_path}')


@main.command()
@argument('input-path', type = str, default = PREPARED_DATA_PATH)
@argument('output-path', type = str, default = REPORT_PATH)
def evaluate(input_path: str, output_path: str):
    dfs = []

    reports_directory = os_path.dirname(output_path)

    if not os_path.isdir(reports_directory):
        mkdir(reports_directory)

    for root, _, files in walk(input_path):
        for filename in files:
            if not filename.endswith('json'):
                continue

            print()
            print(filename)
            print()

            with open(os_path.join(root, filename), 'r', encoding = 'utf-8') as file:
                data = load(file)

            df = run_evaluation(data['elements'])
            dfs.append(df)

    average_df = average(dfs)
    average_df.to_csv(output_path, sep = '\t')


@main.command()
@argument('input-path', type = str, default = PREPARED_DATA_PATH)
@argument('output-path', type = str, default = PREPARED_DATA_PATH)
@option('--model-path', '-p', type = str, default = MODEL_PARAMS_PATH)
@option('--architecture', '-a', type = EmbedderType, default = EmbedderType.FLAT)
@option('--model', '-m', type = BaseModel, default = BaseModel.E5_LARGE)
@option('--cpu', '-c', is_flag = True)
@option('--input-dim', '-d', type = int, default = DEFAULT_INPUT_DIM)
def embed(input_path: str, output_path: str, model_path: str, architecture: EmbedderType, model: BaseModel, cpu: bool, input_dim: int):
    if architecture == EmbedderType.FLAT:
        embedder = FlatEmbedder(model, cuda = not cpu)
    elif architecture == EmbedderType.STRUCTURED:
        embedder = StructuredEmbedder(model, cuda = not cpu, input_dim = input_dim, path = model_path)
    else:
        raise ValueError('Unsupported embedder architecture')

    for root, _, files in walk(input_path):
        for filename in files:
            if not filename.endswith('json'):
                continue

            print()
            print(filename)
            print()

            with open(os_path.join(root, filename), 'r') as file:
                data = load(file)

            embedder.embed(data['elements'])

            if not os_path.isdir(output_path):
                mkdir(output_path)

            with open(os_path.join(output_path, filename), 'w') as file:
                dump(data, file, indent = INDENT, ensure_ascii = False)


@main.command()
@argument('input-path', type = str, default = RAW_DATA_PATH)
@argument('output-path', type = str, default = PREPARED_DATA_PATH)
@argument('annotations-path', type = str, required = False)
@option('--train-fraction', '-t', type = float, default = DEFAULT_TRAIN_FRACTION)
@option('--seed', '-s', type = int, default = DEFAULT_SEED)
def prepare(input_path: str, output_path: str, annotations_path: str, train_fraction: float, seed: int):
    prepare_impl(input_path, output_path, annotations_path, train_fraction, seed)


if __name__ == '__main__':
    main()
