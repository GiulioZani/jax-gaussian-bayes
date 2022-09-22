from abc import ABC


class Model(ABC):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def predict(self, x) -> int:
        raise NotImplementedError
