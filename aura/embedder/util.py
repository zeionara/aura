def to_cuda(data: dict):
    data_on_cuda = {}

    for key, value in data.items():
        data_on_cuda[key] = value.to('cuda')

    return data_on_cuda


def describe(table: dict):
    return {
        'rows': [
            [
                {
                    'text': cell.get('text'),
                    'rows': cell.get('rows'),
                    'cols': cell.get('cols'),
                }
                for cell in row
            ]
            for row in table['rows']
        ]
    }
