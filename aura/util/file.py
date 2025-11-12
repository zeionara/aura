from json import dump, load


def dict_to_json_file(data: dict, path: str):
    with open(path, 'w', encoding = 'utf-8') as file:
        dump(
            data,
            file,
            indent = 2,
            ensure_ascii = False
        )


def dict_from_json_file(path: str):
    with open(path, 'r', encoding = 'utf-8') as file:
        return load(file)
