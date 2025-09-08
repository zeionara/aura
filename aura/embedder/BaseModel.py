from enum import Enum


class BaseModel(Enum):
    E5_LARGE = 'intfloat/multilingual-e5-large-instruct'
    QWEN3 = 'Qwen/Qwen3-Embedding-0.6B'
    RUBERT = 'DeepPavlov/rubert-base-cased'
