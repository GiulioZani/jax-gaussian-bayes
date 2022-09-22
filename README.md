# Multivariate Gaussian Bayes classsifier in JAX

This is a simple implementation of a multivariate Gaussian Bayes classifier in [JAX](https://github.com/google/jax).
The classifier is trained and tested on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
## Run by:
```
python evaluate.py
```
It will print the accuracy of a random classifier vs a Navie Bayes classifier vs a Multivariate Gaussian Bayes classifier.

If the program fais, probably you need to download the dataset first. You can do it by running:
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

## Dependencies
- python >= 3.9
- [JAX](https://github.com/google/jax)

