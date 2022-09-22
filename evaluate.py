import jax.numpy as np
from abc import ABC
from jaxtyping import Array, Float
import pickle
import ipdb
from numpy import random
import os

from model import Model
from random_model import RandomModel
from gaussian_naive_bayes import GaussianNaiveBayes
from multivariate_gaussian_bayes import MultivariateGaussianBayes
import jax


def accuracy(test_x, test_y, model: Model) -> float:
    correct = 0
    for i in range(len(test_x)):
        if model.predict(test_x[i]) == test_y[i]:
            correct += 1
    return correct / len(test_y)


def unpickle(file):
    with open(file, "rb") as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


def read_data(location: str):
    files = os.listdir(location)
    data = []
    labels = []
    for file in files:
        if file.startswith("data"):
            datadict = unpickle(os.path.join(location, file))
            data.append(datadict["data"])
            labels.append(np.array(datadict["labels"]))

    test_dict = unpickle(os.path.join(location, "test_batch"))
    test_data = test_dict["data"]
    test_labels = np.array(test_dict["labels"])
    return (np.concatenate(data), np.concatenate(labels)), (
        test_data,
        test_labels,
    )


def main():
    # datadict = unpickle("cifar-10-batches-py/data_batch_1")
    if not os.path.exists("cifar-10-batches-py"):
        try:
            os.system(
                "wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            )
            os.system("tar -xvzf cifar-10-python.tar.gz")
        except:
            print(
                "Could not download data (probably you don't have wget). You can download it yourself and put it in the current directory."
            )
            exit(1)
    (train_x, train_y), (test_x, test_y) = read_data("cifar-10-batches-py")
    train_x = train_x.reshape((len(train_x), 3, 32, 32))
    test_x = test_x.reshape((len(test_x), 3, 32, 32))
    # datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')
    key = jax.random.PRNGKey(123)
    labeldict = unpickle("cifar-10-batches-py/batches.meta")
    # random_model = RandomModel(train_x, train_y)
    # print(f"{accuracy(test_x, test_y, random_model)=}")
    # gaussian_naive_bayes_classifier = GaussianNaiveBayes(train_x, train_y)
    # print(f"{accuracy(test_x, test_y, gaussian_naive_bayes_classifier)=}")
    multivariate_gaussian_bayes_classifier = MultivariateGaussianBayes(
        train_x, train_y, downsample_size=2
    )
    print(
        f"{accuracy(test_x, test_y, multivariate_gaussian_bayes_classifier)=}"
    )


if __name__ == "__main__":
    main()
