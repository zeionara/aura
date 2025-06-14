import re

SPACE = re.compile(r'\s+')


def normalize_spaces(string: str):
    return SPACE.sub(' ', string).strip()
