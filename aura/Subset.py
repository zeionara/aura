from enum import Enum
from random import random


class Subset(Enum):
    TRAIN = 'train'
    TEST = 'test'

    @staticmethod
    def random(train_fraction: float):
        number = random()

        if number > train_fraction:
            return Subset.TEST

        return Subset.TRAIN

    @staticmethod
    def count(items):
        train_count = 0
        test_count = 0
        none_count = 0

        for item in items:
            if item.subset == Subset.TRAIN:
                train_count += 1
            elif item.subset == Subset.TEST:
                test_count += 1
            else:
                none_count += 1

        return train_count, test_count, none_count
