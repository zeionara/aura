from os import path as os_path
from json import dumps

from .string import read


PROMPTS_PATH = 'assets/prompt'


def make_system_prompt():
    return read(
        os_path.join(
            PROMPTS_PATH,
            'system.md'
        )
    )


def make_annotation_prompt(table: dict):
    prompt_template = read(
        os_path.join(
            PROMPTS_PATH,
            'annotation.md'
        )
    )

    return prompt_template.format(
        table = dumps(
            table,
            indent = 2,
            ensure_ascii = False
        )
    )
