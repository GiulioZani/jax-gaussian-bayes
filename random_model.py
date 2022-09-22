from model import Model
from jax import numpy as np
import jax


class RandomModel(Model):
    def __init__(self, train_x, train_y):
        super().__init__(train_x, train_y)
        self.rnd_key = jax.random.PRNGKey(123)

    def predict(self, x) -> int:
        return jax.random.randint(self.rnd_key, (1,), 0, 10)[0]
