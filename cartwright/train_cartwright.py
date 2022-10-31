from __future__ import unicode_literals, print_function, division

import os
from io import open
import sys
import math
import random
import argparse
import operator
import pdb

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from collections import defaultdict
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import pandas as pd
import numpy as np
import re
from string import punctuation
import glob
import unicodedata
import string
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from cartwright.utils import character_tokins
from cartwright.category_helpers import return_all_category_classes_and_labels
from cartwright.LSTM import LSTMClassifier, PaddedTensorDataset
from cartwright.CartWrightBase import CartWrightBase

# This class creates a randomized dataset and splits it into training ,validation and testing for model training and validation
class CartwrightTrainer(CartWrightBase):
    def __init__(self, seed=1, num_epochs=1, learning_rate=.001):
        super().__init__()
        self.category_values = {}

        self.train_split = []
        self.dev_split = []
        self.test_split = []
        self.training_set_size = 100000
        self.dev_set_size = 1000
        self.test_set_size = 4000
        self.traning_data_size = self.get_training_data_size()

        self.seed = seed
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.weight_decay = 1e-4
        self.batch_size = 32

        self.criterion = nn.NLLLoss(size_average=False)

    def get_training_data_size(self):
        return self.training_set_size + self.dev_set_size + self.test_set_size

    def datapoint(self):

        label = np.random.choice(self.all_labels)

        class_ = self.all_classes[label]
        lab, val = class_.generate_training_data()

        return {"hash": lab, "obj": val}

    def dataframe(self, size):
        self.df = pd.DataFrame([self.datapoint() for i in range(size)])
        self.get_category_values()
        return self.df

    def get_category_values(self):

        for category in self.all_labels:
            bool_array = self.df['hash'] == category
            self.category_values[category] = self.df[bool_array]

    def randomChoice(self, values):
        return values[random.randint(0, len(values) - 1)]

    def getRandomSet(self):
        category = self.randomChoice(self.all_labels)
        line = self.randomChoice(list(self.category_values[category]['obj']))
        return (line, category)

    def updateAllCharacters(self):
        for x in (self.df['obj'].values):
            for y in list(str(x)):
                if y in self.all_characters:
                    pass
                else:
                    self.all_characters += y
                    self.n_characters = len(self.all_characters)

    def split_data(self):

        self.train_split = []
        self.dev_split = []
        self.test_split = []

        for _ in range(self.training_set_size):
            self.train_split.append(self.getRandomSet())

        for _ in range(self.dev_set_size):
            self.dev_split.append(self.getRandomSet())

        for _ in range(self.test_set_size):
            self.test_split.append(self.getRandomSet())

    def apply(self, batch, targets, lengths):
        pred = self.model(torch.autograd.Variable(batch), lengths.cpu().numpy())
        loss = self.criterion(pred, torch.autograd.Variable(targets))
        return pred, loss


    def train_model(self, optimizer):

        for epoch in range(self.num_epochs):
            print('Epoch:', epoch)
            y_true = list()
            y_pred = list()
            total_loss = 0
            for batch, targets, lengths, raw_data in self.create_training_dataset(self.train_split,
                                                                                  self.character_tokins, self.label2id,
                                                                                  self.batch_size):
                batch, targets, lengths = self.sort_batch(batch, targets, lengths)
                self.model.zero_grad()
                pred, loss = self.apply(batch, targets, lengths)
                loss.backward()
                optimizer.step()
                pred_idx = torch.max(pred, 1)[1]
                y_true += list(targets.int())
                y_pred += list(pred_idx.data.int())

                total_loss += loss
            acc = accuracy_score(y_true, y_pred)
            val_loss, val_acc = self.evaluate_validation_set()
            print("Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(
                total_loss.data.float() / len(self.train_split), acc,
                val_loss, val_acc))

    def evaluate_validation_set(self):
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in self.create_training_dataset(self.dev_split, self.character_tokins,
                                                                              self.label2id, batch_size=1):
            batch, targets, lengths = self.sort_batch(batch, targets, lengths)
            pred, loss = self.apply(batch, targets, lengths)
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        return total_loss.data.float() / len(self.dev_split), acc

    def evaluate_test_set(self):
        y_true = list()
        y_pred = list()

        for batch, targets, lengths, raw_data in self.create_training_dataset(self.test_split, self.character_tokins,
                                                                              self.label2id, batch_size=1):
            batch, targets, lengths = self.sort_batch(batch, targets, lengths)
            pred = self.model(torch.autograd.Variable(batch), lengths.cpu().numpy())
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())

        confusion = confusion_matrix(y_true, y_pred)

        # Set up plot
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion)
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels(self.label2id, rotation=90)
        ax.set_yticklabels(self.label2id)
        ax.set_xticks(np.arange(len(self.label2id)))
        ax.set_yticks(np.arange(len(self.label2id)))

        plt.show()

    # train the model by creating a pseudo dataset
    def train(self):
        print("Starting training")
        random.seed(self.seed)

        self.dataframe(size=self.get_training_data_size())
        self.split_data()

        print('Training samples:', len(self.train_split))
        print('Valid samples:', len(self.dev_split))
        print('Test samples:', len(self.test_split))

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.train_model(optimizer)

        self.evaluate_test_set()

    def save_model(self, version):
        # If the model performed well you can save it locally
        path = f'cartwright/models/LSTM_RNN_CartWright_v_{version}_dict.pth'
        torch.save(self.model.state_dict(), path)


def default_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="set the version of the model")
    parser.add_argument("--num_epochs",type=int, help="number of epochs for model training")
    parser.add_argument("--new_data",type=bool, help="generate new training data")
    args = parser.parse_args()
    cartwright = CartwrightTrainer(num_epochs=args.num_epochs)
    cartwright.train()
    cartwright.save_model(version=str(args.version))

if __name__ == "__main__":
    default_training()