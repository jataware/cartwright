#testing one class neural network

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

from matplotlib import pyplot as plt

import pdb


def main():
    normal_train, normal_test, anomaly = get_data()
    



def get_data(normal_class=6):
    """get mnist data split into train and test for the normal class, and all others are anomalies"""

    mnist = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # collect the normal and anomaly data
    normal = [img for img, label in mnist if label == normal_class]
    anomaly = [img for img, label in mnist if label != normal_class]

    # create a train test split for the normal data
    num_train = int(len(normal) * 0.5)
    num_test = len(normal) - num_train
    normal_train, normal_test = torch.utils.data.random_split(normal, [num_train, num_test])

    # stack all of the training/testing/anomaly data into single tensors
    normal_train = torch.stack((*normal_train,))
    normal_test = torch.stack((*normal_test,))
    anomaly = torch.stack(anomaly)

    # reshape each image into a vector
    normal_train = normal_train.reshape(len(normal_train), -1)
    normal_test = normal_test.reshape(len(normal_test), -1)
    anomaly = anomaly.reshape(len(anomaly), -1)

    # normalize the data from 0 to 1
    normal_train /= 255.0
    normal_test /= 255.0
    anomaly /= 255.0

    return normal_train, normal_test, anomaly




def vec2img(vector: torch.Tensor) -> torch.Tensor:
    """convert a vector to an image"""
    return vector.view((28, 28))

def vecshow(vector: torch.Tensor) -> None:
    """plot a vector as an image"""
    img = vec2img(vector)
    plt.imshow(img, cmap='gray')
    plt.show()




if __name__ == '__main__':
    main()