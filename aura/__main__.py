from os import walk, path as os_path, mkdir, getenv
from pathlib import Path
from queue import Queue
from json import dump, load
from logging import getLogger, StreamHandler, INFO, Formatter
from random import seed as random_seed
from torch import optim

from click import group, argument, option

from .util import get_comments, get_elements, get_xml, get_text, read_elements, normalize_spaces, make_annotation_prompt, make_system_prompt, dict_from_json_file  # , get_paragraph_style
from .document import Paragraph, Table, Document, INDENT, Cell, ZipFile
from .document.ZipFile import ZipFile
from .embedder import EmbedderType, BaseModel, FlatEmbedder, StructuredEmbedder
from .evaluation import evaluate as run_evaluation, average
from .Stats import Stats
from .Subset import Subset
from .embedder.AttentionTableEmbedder import DEFAULT_INPUT_DIM
from .VllmClient import VllmClient
from .Annotator import Annotator, TABLE_TITLE_PATTERN


RAW_DATA_PATH = 'assets/data/raw'
SOURCE_DATA_PATH = 'assets/data/source'
ANNOTATIONS_PATH = 'assets/data/annotations'
PREPARED_DATA_PATH = 'assets/data/prepared'
REPORT_PATH = 'assets/data/report.tsv'
ANNOTATIONS_DOCUMENT_PATH = 'assets/data/annotations.docx'
MODEL_PARAMS_PATH = None  # 'assets/weights.pth'

DEFAULT_TRAIN_FRACTION = 0.6
DEFAULT_SEED = 17

root = getLogger()

handler = StreamHandler()
handler.setLevel(INFO)

formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

root.addHandler(handler)


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
@option('--host', default = 'localhost')
@option('--port', default = 8080)
@option('--model', default = 'default')
@option('--batch-size', '-b', type = int, default = None)
@option('--n-batches', '-n', type = int, default = None)
@option('--dry-run', '-d', is_flag = True, default = False)
def annotate(input_path: str, output_path: str, host: str, port: int, model: str, batch_size: int, n_batches: int, dry_run: bool):
    annotator = Annotator(
        llms = [
            VllmClient(host, port, model, make_system_prompt(), label = 'mistral-24b'),
            # VllmClient(host, port, model, make_system_prompt(), label = 'local bar')
        ]
    )

    annotator.annotate(input_path, output_path, batch_size, n_batches, dry_run = dry_run)


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
@argument('annotations-document-path', type = str, default = ANNOTATIONS_DOCUMENT_PATH)
@option('--train-fraction', '-t', type = float, default = DEFAULT_TRAIN_FRACTION)
@option('--seed', '-s', type = int, default = DEFAULT_SEED)
def prepare(input_path: str, output_path: str, annotations_document_path: str, train_fraction: float, seed: int):
    random_seed(seed)

    doc = Document()

    doc.append_h1('Результаты разметки контекста таблиц для интерпретации неполных табличных данных')

    doc.append_h2('Введение')
    doc.append_paragraph((
        'Этот документ был сгенерирован автоматически по результатам аннотирования контекста таблиц в рамках кандидатской диссертации, посвященной исследованию '
        'методов интерпретации неполных табличных данных.'
    ))
    doc.append_paragraph((
        'В связи с особенностями структуры docx документа графическое содержимое (рисунки, формулы, чертежи и т.п.) отсутствует. Обработка таких данных выходит за рамки выполненного '
        'исследования, и подобные элементы нормативных документов не учитываются в предложенной реализации метода вопросно-ответного поиска по табличным данным нормативной документации.'
    ))

    doc.append_h2('Результаты разметки')
    doc.append_paragraph((
        'Далее приведен список документов, для каждого документа указан список таблиц, а для каждой таблицы приведено описание ее структуры и текстового содержимого, а также список параграфов, '
        'которые были размечены как релевантные по результатам аннотирования. Параграфы расположены по убыванию коэффициента релевантности.'
    ))

    doc.append_paragraph((
        'Предложенные результаты могут быть использованы в качестве примеров как для ручного аннотирования новых документов, '
        'так и для автоматического аннотирования с использованием больших языковых моделей.'
    ))

    stats = Stats()

    # doc.append(paragraph_xml)
    # doc.append(table_xml)

    # doc.to_docx('document.docx')

    # return

    for root, _, files in walk(input_path):
        for file in files:

            doc.append_h3(f'Документ {file}')

            path = os_path.join(root, file)

            output_file = os_path.join(output_path, f'{Path(file).stem}.json')

            if not os_path.isdir(output_path):
                mkdir(output_path)

            comments, content = get_comments(path)
            elements = get_elements(content, comments)

            records = []

            label_to_paragraph_ids = {}
            paragraph_id_to_xml = {}
            paragraph_id_to_element = {}

            paragraphs = Queue()

            # 1. Handle paragraph comments, group by target (table) id

            for element, comments in elements:
                if element.tag.endswith('}p'):

                    # style = get_paragraph_style(element)

                    # if style is None:
                    #     print(get_xml(element))

                    paragraph = Paragraph.from_xml(element, comments)
                    paragraphs.put(paragraph)

                    if paragraph:
                        if comments is not None:
                            paragraph.subset = Subset.random(train_fraction)
                            for comment in comments:
                                if (label := comment.body.target) is not None:
                                    if (paragraph_ids := label_to_paragraph_ids.get(comment.body.target)) is None:
                                        label_to_paragraph_ids[label] = [(paragraph.id, comment.body.score)]
                                    else:
                                        paragraph_ids.append((paragraph.id, comment.body.score))

                                    paragraph_id_to_xml[paragraph.id] = get_xml(element)
                                    paragraph_id_to_element[paragraph.id] = element

            # 2. Handle table comments, extract linked paragraphs

            for element, comments in elements:
                if element.tag.endswith('}p'):
                    paragraph = paragraphs.get()

                    if paragraph:
                        records.append(paragraph.json)
                else:
                    label = None

                    if comments is not None:
                        for comment in comments:
                            if comment.body.id is not None:
                                label = comment.body.id

                    context = None

                    if label in label_to_paragraph_ids:
                        doc.append_h4(f'Таблица {label}')

                        doc.append_h5('Содержимое')

                        doc.append(get_xml(element))
                        table_paragraphs = []

                        doc.append_h5('Контекст')

                        for paragraph, score in sorted(label_to_paragraph_ids.get(label), key = lambda entry: entry[1], reverse = True):
                            if context is None:
                                context = {paragraph: score}
                            else:
                                context[paragraph] = score

                            xml = paragraph_id_to_xml.get(paragraph)

                            if xml is None:
                                logger.warning('Missing xml for paragraph %s', paragraph)
                            else:
                                doc.append(doc.remove_style(xml))
                                table_paragraphs.append(paragraph_id_to_element[paragraph])

                        stats.append_table(file, label, table_paragraphs)
                    elif label is not None:
                        logger.warning('Table %s is missing annotations', label)

                    table = Table.from_xml(
                        element,
                        label,
                        comments,
                        context
                        # {
                        #     paragraph: score
                        #     for paragraph, score in sorted(
                        #         label_to_paragraph_ids.get(label),
                        #         key = lambda entry: entry[1],
                        #         reverse = True
                        #     )
                        # } if label in label_to_paragraph_ids else None
                    )

                    records.append(table.json)

            with open(output_file, 'w', encoding = 'utf-8') as file:
                dump(
                    {
                        'elements': records
                    }, file, indent = INDENT, ensure_ascii = False
                )

        doc.append_h2('Статистические характеристики результатов разметки')

        doc.append_paragraph(f'Количество размеченных документов: {stats.n_documents}')
        doc.append_paragraph(f'Количество размеченных таблиц: {sum(stats.n_tables)}')
        doc.append_paragraph(f'Количество размеченных параграфов: {sum(stats.n_paragraphs)}')

        doc.append_paragraph(f'Среднее количество размеченных таблиц в документе: {stats.n_tables_average:.3f}')
        doc.append_paragraph(f'Среднеквадратичное отклонение количества размеченных таблиц в документе: {stats.n_tables_stdev:.3f}')

        doc.append_paragraph(f'Среднее количество параграфов контекста (по всем документам): {stats.n_paragraphs_average:.3f}')
        doc.append_paragraph(f'Среднеквадратичное отклонение количества параграфов контекста (по всем документам): {stats.n_paragraphs_stdev:.3f}')

        doc.append_paragraph(f'Среднее количество символов в параграфе контекста (по всем документам): {stats.paragraph_length_average:.3f}')
        doc.append_paragraph(f'Среднеквадратичное отклонение количества символов в параграфе контекста (по всем документам): {stats.paragraph_length_stdev:.3f}')

        n_stats_elements = 10

        for document, document_stats in stats:
            doc.append_h3(f'Документ {document}')

            doc.append_paragraph(f'Среднее количество параграфов контекста (по таблицам документа {document}): {document_stats.n_paragraphs_average:.3f}')
            doc.append_paragraph(f'Среднеквадратичное отклонение количества параграфов контекста (по таблицам документа {document}): {document_stats.n_paragraphs_stdev:.3f}')

            doc.append_paragraph(f'Среднее количество символов в параграфе контекста (по таблицам документа {document}): {document_stats.paragraph_length_average:.3f}')
            doc.append_paragraph(f'Среднеквадратичное отклонение количества символов в параграфе контекста (по таблицам документа {document}): {document_stats.paragraph_length_stdev:.3f}')

            n_stats_elements += 5

        doc.move_last_n_elements(n_stats_elements, 4)

        with open(output_file, 'w', encoding = 'utf-8') as file:
            dump(
                {
                    'elements': records
                }, file, indent = INDENT, ensure_ascii = False
            )

        doc.to_docx(annotations_document_path)


if __name__ == '__main__':
    main()
