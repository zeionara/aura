import re
from os import walk, path as os_path

from click import group, argument

from .util import get_comments, get_elements


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

            comments, content = get_comments(path)
            elements = get_elements(content, comments)

            print()
            print(path)
            print()

            for element, comments in elements:
                print(element, 0 if comments is None else len(comments))


if __name__ == '__main__':
    main()
