from json import dump


def dict_to_json_file(data: dict, path: str):
    with open(path, 'w', encoding = 'utf-8') as file:
        dump(
            data,
            file,
            indent = 2,
            ensure_ascii = False
        )
