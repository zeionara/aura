import re

SPACE = re.compile(r'\s+')

PUNCTUATION_WITH_LEADING_SPACE = re.compile(r'\s+([.,;])')
PUNCTUATION_WITH_TRAILING_SPACE = re.compile(r'([\[])\s+')


def normalize_spaces(string: str):
    return SPACE.sub(' ', string).strip()


def drop_space_around_punctuation(string: str):
    return PUNCTUATION_WITH_TRAILING_SPACE.sub(
        r'\g<1>',
        PUNCTUATION_WITH_LEADING_SPACE.sub(r'\g<1>', string)
    )


def read(path: str):
    with open(path, 'r', encoding = 'utf-8') as file:
        return file.read()
