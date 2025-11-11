import re
from json import dumps, JSONDecoder
from json.decoder import JSONDecodeError


SPACE = re.compile(r'\s+')

PUNCTUATION_WITH_LEADING_SPACE = re.compile(r'\s+([.,;])')
PUNCTUATION_WITH_TRAILING_SPACE = re.compile(r'([\[])\s+')


JSON_DECODER = JSONDecoder()


def dict_to_string(data: dict):
    return dumps(
        data,
        indent = 2,
        ensure_ascii = False
    )


def string_to_dict(data: str):
    idx = 0

    while idx < len(data):
        idx = data.find('{')

        if idx == -1:
            break

        try:
            data, _ = JSON_DECODER.raw_decode(data, idx)
            return data
        except JSONDecodeError:
            idx += 1

    return None


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
