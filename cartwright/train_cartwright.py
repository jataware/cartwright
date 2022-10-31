from __future__ import unicode_literals, print_function, division

from io import open
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from cartwright.CartwrightBase import CartwrightBase

# This class creates a randomized dataset and splits it into training ,validation and testing for model training and validation
class CartwrightTrainer(CartwrightBase):
    def __init__(self, seed=1,training_set_size=150000, test_set_size=5000, num_epochs=1, learning_rate=.001):
        super().__init__()
        self.category_values = {}

        self.train_split = []
        self.dev_split = []
        self.test_split = []
        self.training_set_size = training_set_size
        self.dev_set_size = 5000
        self.test_set_size = test_set_size
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

    def create_or_load_training_date(self,create_training_data):
        if create_training_data:
            random.seed(self.seed)
            self.dataframe(size=self.get_training_data_size())
            self.split_data()

        else:
            self.test_split=[]
            self.train_split=[]
            self.dev_split=[]

            with open('cartwright/resources/training_data.pickle', 'rb') as f:
                self.train_split = pickle.load(f)

            with open('cartwright/resources/testing_data.pickle', 'rb') as f:
                self.test_split = pickle.load(f)

            with open('cartwright/resources/dev_data.pickle', 'rb') as f:
                self.dev_split = pickle.load(f)

    def save_data(self):
        self.df.to_csv('cartwright/resources/all_training_data.csv')
        with open('cartwright/resources/training_data.pickle', 'wb') as f:
            pickle.dump(self.train_split, f)

        with open('cartwright/resources/testing_data.pickle', 'wb') as f:
            pickle.dump(self.test_split, f)

        with open('cartwright/resources/dev_data.pickle', 'wb') as f:
            pickle.dump(self.dev_split, f)

    # train the model by creating a pseudo dataset
    def train(self, create_training_date=False):
        print("Starting training")
        self.create_or_load_training_date(create_training_data=create_training_date)

        self.save_data()

        print('Training samples:', len(self.train_split))
        print('Valid samples:', len(self.dev_split))
        print('Test samples:', len(self.test_split))

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.train_model(optimizer)

        self.evaluate_test_set()

    def save_model(self, version):
        # If the model performed well you can save it locally
        path = f'cartwright/models/LSTM_RNN_Cartwright_v_{version}_dict.pth'
        torch.save(self.model.state_dict(), path)


def default_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.0.1",help="set the version of the model")
    parser.add_argument("--num_epochs",type=int,default=8, help="number of epochs for model training")
    parser.add_argument("--new_data",  action='store_true', help="generate new training data")
    parser.add_argument("--training_size", type=int,default=250000, help="size of training dataset")
    parser.add_argument("--testing_size", type=int,default=10000, help="size of testing dataset")
    args = parser.parse_args()
    print(args)
    cartwright = CartwrightTrainer(num_epochs=args.num_epochs)
    cartwright.train(create_training_date=args.new_data)
    cartwright.save_model(version=str(args.version))

if __name__ == "__main__":
    default_training()