def to_cuda(data: dict):
    data_on_cuda = {}

    for key, value in data.items():
        data_on_cuda[key] = value.to('cuda')

    return data_on_cuda
