from os import walk, path as os_path, mkdir
from pathlib import Path
from json import dump

from click import group, argument

from .util import get_comments, get_elements  # , get_paragraph_style
from .document import Paragraph, Table, INDENT


RAW_DATA_PATH = 'assets/data/raw'
PREPARED_DATA_PATH = 'assets/data/prepared'


@group()
def main():
    pass


@main.command()
@argument('input-path', type = str, default = RAW_DATA_PATH)
@argument('output-path', type = str, default = PREPARED_DATA_PATH)
def prepare_corpus(input_path: str, output_path: str):
    for root, _, files in walk(input_path):
        for file in files:
            path = os_path.join(root, file)

            output_folder = os_path.join(Path(root).parent, 'prepared')
            output_path = os_path.join(output_folder, f'{Path(file).stem}.json')

            if not os_path.isdir(output_folder):
                mkdir(output_folder)

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

            with open(output_path, 'w') as file:
                dump(
                    {
                        'elements': records
                    }, file, indent = INDENT, ensure_ascii = False
                )


if __name__ == '__main__':
    main()
