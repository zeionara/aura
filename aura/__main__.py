from os import walk, path as os_path, mkdir
from pathlib import Path
from json import dump, load

from click import group, argument, option

from .util import get_comments, get_elements  # , get_paragraph_style
from .document import Paragraph, Table, INDENT
from .embedder import EmbedderType, BaseModel, FlatEmbedder


RAW_DATA_PATH = 'assets/data/raw'
PREPARED_DATA_PATH = 'assets/data/prepared'


@group()
def main():
    pass


@main.command()
@argument('input-path', type = str, default = PREPARED_DATA_PATH)
@argument('output-path', type = str, default = PREPARED_DATA_PATH)
@option('--architecture', '-a', type = EmbedderType, default = EmbedderType.FLAT)
@option('--model', '-m', type = BaseModel, default = BaseModel.E5_LARGE)
@option('--cpu', '-c', is_flag = True)
def embed(input_path: str, output_path: str, architecture: EmbedderType, model: BaseModel, cpu: bool):
    if architecture == EmbedderType.FLAT:
        embedder = FlatEmbedder(model, cuda = not cpu)

    for root, _, files in walk(input_path):
        for file in files:
            print()
            print(file)
            print()

            with open(os_path.join(root, file), 'r') as file:
                data = load(file)

            embedder.embed(data['elements'])


@main.command()
@argument('input-path', type = str, default = RAW_DATA_PATH)
@argument('output-path', type = str, default = PREPARED_DATA_PATH)
def prepare(input_path: str, output_path: str):
    for root, _, files in walk(input_path):
        for file in files:
            path = os_path.join(root, file)

            output_file = os_path.join(output_path, f'{Path(file).stem}.json')

            if not os_path.isdir(output_path):
                mkdir(output_path)

            comments, content = get_comments(path)
            elements = get_elements(content, comments)

            records = []

            label_to_paragraph_ids = {}

            for element, comments in elements:
                if element.tag.endswith('}p'):
                    paragraph = Paragraph.from_xml(element)

                    if paragraph:
                        if comments is not None:
                            for comment in comments:
                                if (label := comment.body.target) is not None:
                                    if (paragraph_ids := label_to_paragraph_ids.get(comment.body.target)) is None:
                                        label_to_paragraph_ids[label] = [(paragraph.id, comment.body.score)]
                                    else:
                                        paragraph_ids.append((paragraph.id, comment.body.score))

                        records.append(paragraph.json)

            for element, comments in elements:
                if element.tag.endswith('}p'):
                    paragraph = Paragraph.from_xml(element)

                    if paragraph:
                        records.append(paragraph.json)
                else:
                    label = None

                    if comments is not None:
                        for comment in comments:
                            if comment.body.id is not None:
                                label = comment.body.id

                    table = Table.from_xml(
                        element,
                        label,
                        [
                            paragraph
                            for paragraph, _ in sorted(
                                label_to_paragraph_ids.get(label),
                                key = lambda entry: entry[1],
                                reverse = True
                            )
                        ] if label in label_to_paragraph_ids else None
                    )

                    records.append(table.json)

            with open(output_file, 'w') as file:
                dump(
                    {
                        'elements': records
                    }, file, indent = INDENT, ensure_ascii = False
                )


if __name__ == '__main__':
    main()
