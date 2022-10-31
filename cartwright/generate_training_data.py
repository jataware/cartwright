from __future__ import unicode_literals, print_function, division


from io import open
import argparse
import pickle
import pandas as pd
import numpy as np
import random

from cartwright.CartwrightBase import CartwrightBase

# This class creates a randomized dataset and splits it into training ,validation and testing for model training and validation
class CartwrightDatasetGenerator(CartwrightBase):
    def __init__(self, seed=1, training_size=150000, test_size=5000 ):
        super().__init__()
        self.category_values = {}

        self.train_split = []
        self.dev_split = []
        self.test_split = []
        self.training_set_size = training_size
        self.dev_set_size = 4000
        self.test_set_size = test_size
        self.traning_data_size = self.get_training_data_size()

        self.seed = seed

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




    def save_data(self):
        self.df.to_csv('cartwright/resources/all_training_data.csv')
        with open('cartwright/resources/training_data.pickle', 'wb') as f:
            pickle.dump(self.train_split, f)

        with open('cartwright/resources/testing_data.pickle', 'wb') as f:
            pickle.dump(self.test_split, f)

        with open('cartwright/resources/dev_data.pickle', 'wb') as f:
            pickle.dump(self.dev_split, f)

    # train the model by creating a pseudo dataset
    def generate_training_data(self):
        print("Generating data")
        random.seed(self.seed)
        self.dataframe(size=self.get_training_data_size())
        self.split_data()
        self.save_data()




def generate_training_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=1, help="set random seed")
    parser.add_argument("--training_size",type=int, default=150000, help="set size of training dataset")
    parser.add_argument("--test_size",type=int, default=5000, help="set size of test dataset")
    args = parser.parse_args()
    print(f'args {args}')
    cartwright = CartwrightDatasetGenerator(seed=args.seed, training_size=args.training_size, test_size=args.test_size)
    cartwright.generate_training_data()

if __name__ == "__main__":
    generate_training_data()