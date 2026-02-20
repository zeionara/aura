def to_cuda(data: dict):
    data_on_cuda = {}

    for key, value in data.items():
        data_on_cuda[key] = value.to('cuda')

    return data_on_cuda


def drop_embeddings(cells: list[dict]):
    return [
        {
            'id': cell.get('id'),
            'text': cell.get('text'),
            'rows': cell.get('rows'),
            'cols': cell.get('cols'),
            'updated': cell.get('updated')
        }
        for cell in cells
    ]
