from os import walk, path as os_path, mkdir
from pathlib import Path
from queue import Queue
from json import dump, load
from logging import getLogger

from click import group, argument, option

from .util import get_comments, get_elements, get_xml  # , get_paragraph_style
from .document import Paragraph, Table, Document, INDENT
from .embedder import EmbedderType, BaseModel, FlatEmbedder, StructuredEmbedder
from .evaluation import evaluate as run_evaluation, average


RAW_DATA_PATH = 'assets/data/raw'
PREPARED_DATA_PATH = 'assets/data/prepared'
REPORT_PATH = 'assets/data/report.tsv'
ANNOTATIONS_DOCUMENT_PATH = 'assets/data/annotations.docx'


logger = getLogger(__name__)


@group()
def main():
    pass


@main.command()
@argument('input-path', type = str, default = PREPARED_DATA_PATH)
@argument('output-path', type = str, default = REPORT_PATH)
def evaluate(input_path: str, output_path: str):
    dfs = []

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
@option('--architecture', '-a', type = EmbedderType, default = EmbedderType.FLAT)
@option('--model', '-m', type = BaseModel, default = BaseModel.E5_LARGE)
@option('--cpu', '-c', is_flag = True)
def embed(input_path: str, output_path: str, architecture: EmbedderType, model: BaseModel, cpu: bool):
    if architecture == EmbedderType.FLAT:
        embedder = FlatEmbedder(model, cuda = not cpu)
    elif architecture == EmbedderType.STRUCTURED:
        embedder = StructuredEmbedder(model, cuda = not cpu)
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
def prepare(input_path: str, output_path: str, annotations_document_path: str):
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
                            for comment in comments:
                                if (label := comment.body.target) is not None:
                                    if (paragraph_ids := label_to_paragraph_ids.get(comment.body.target)) is None:
                                        label_to_paragraph_ids[label] = [(paragraph.id, comment.body.score)]
                                    else:
                                        paragraph_ids.append((paragraph.id, comment.body.score))

                                    paragraph_id_to_xml[paragraph.id] = get_xml(element)

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

            doc.to_docx(annotations_document_path)


if __name__ == '__main__':
    main()
