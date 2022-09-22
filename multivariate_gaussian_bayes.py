import jax.numpy as np
from jax import scipy
from jax.image import resize
from abc import ABC
from jaxtyping import Array, Float
import pickle
import ipdb

from model import Model


def _compute_mean_var_classes(train_x, train_y):
    mean_images = []
    cov_images = []
    for i in range(10):
        class_x = train_x[train_y == i]
        class_x = class_x.reshape(class_x.shape[0], -1).T
        mean_images.append(np.mean(class_x, axis=1))
        cov_images.append(np.cov(class_x))
    return np.array(mean_images), np.array(cov_images)


# creates a multivariate gaussian function
def gaussian(x, means, cov):
    return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def _compute_likelyhood(
    x,
    mean,
    var,
    prior,
):
    return scipy.stats.multivariate_normal.pdf(x, mean, var) * prior


def _compute_priors(train_y):
    return np.array(tuple((train_y == i).sum() / len(train_y) for i in range(10)))


class MultivariateGaussianBayes(Model):
    def __init__(self, train_x, train_y, downsample_size=1):
        self.downsample_size = downsample_size
        self.train_x = np.squeeze(
            resize(
                train_x,
                (train_x.shape[0], 3, downsample_size, downsample_size),
                "nearest",
            )
        )
        self.train_y = train_y
        self._mean_images, self._var_images = _compute_mean_var_classes(
            self.train_x, self.train_y
        )
        self._priors = _compute_priors(self.train_y)

    def predict(self, x) -> int:
        x = np.squeeze(resize(x, (3, self.downsample_size, self.downsample_size), "nearest")).reshape(-1)
        prob_classes = [
            _compute_likelyhood(
                x, self._mean_images[i], self._var_images[i], self._priors[i]
            )
            for i in range(10)
        ]
        return np.argmax(np.array(prob_classes))
