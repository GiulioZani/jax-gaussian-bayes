import jax.numpy as np
from jax.image import resize
from abc import ABC
from jaxtyping import Array, Float
import pickle
import ipdb

from model import Model


def _compute_mean_var_classes(train_x, train_y):
    mean_images = []
    for i in range(10):
        class_x = train_x[train_y == i]
        mean_images.append(
            (
                np.mean(class_x, axis=0),
                np.var(class_x, axis=0),
            )
        )
    return np.squeeze(np.array(mean_images)).transpose(1, 0, 2)


def _gaussian(x, mean, var):
    return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


# creates a multivariate gaussian function
def gaussian(x, mean, var):
    return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def _compute_likelyhood(
    x,
    mean,
    var,
    prior,
):
    return (_gaussian(x, mean, var)).prod() * prior


def _compute_priors(train_y):
    return np.array(
        tuple((train_y == i).sum() / len(train_y) for i in range(10))
    )


class GaussianNaiveBayes(Model):
    def __init__(self, train_x, train_y):
        self.train_x = resize(train_x, (train_x.shape[0], 3, 1, 1), "nearest")
        self.train_y = train_y
        self._mean_images, self._var_images = _compute_mean_var_classes(
            self.train_x, self.train_y
        )
        self._priors = _compute_priors(self.train_y)

    def predict(self, x) -> int:
        x = np.squeeze(resize(x, (3, 1, 1), "nearest"))
        prob_classes = np.array(
            [
                _compute_likelyhood(
                    x,
                    self._mean_images[i],
                    self._var_images[i],
                    self._priors[i],
                )
                for i in range(10)
            ]
        )
        return np.argmax(prob_classes)
