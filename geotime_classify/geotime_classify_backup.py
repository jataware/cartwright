#!/usr/bin/env python
from __future__ import unicode_literals, print_function, division

import os

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

# Kyle's attempt
import pandas as pd
import numpy as np
import re
from string import punctuation
import glob

import string
import random
import time
import dateutil.parser
import datetime
import arrow
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import fuzzywuzzy
import pkgutil
import csv
import pkg_resources
from pydantic import BaseModel, constr, Field
from typing import List, Optional


class fuzzyColumn(BaseModel):
    matchedKey: str = Field(default=None)
    fuzzyCategory: str = Field(default=None)
    ratio: int = Field(default=None)


class Classification(BaseModel):
    column: str = Field(None)
    category: str = None
    subcategory: str = None
    format: str = None
    match_type: List = []
    Parser: str = None
    DayFirst: bool = None
    fuzzyColumn: Optional[fuzzyColumn]


class Classifications(BaseModel):
    classifications: List[Classification]





# Need to have the class of the model in local memory to load a saved model in pytorch
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(LSTMClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.dropout_layer = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        return (
            autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
            autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
        )

    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(-1))

        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)
        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output


class PaddedTensorDataset(Dataset):
    #     """Dataset wrapping data, target and length tensors.

    #     Each sample will be retrieved by indexing both tensors along the first
    #     dimension.

    #     Arguments:
    #         data_tensor (Tensor): contains sample data.
    #         target_tensor (Tensor): contains sample targets (labels).
    #         length (Tensor): contains sample lengths.
    #         raw_data (Any): The data that has been transformed into tensor, useful for debugging
    #     """

    def __init__(self, data_tensor, target_tensor, length_tensor, raw_data):
        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor
        self.raw_data = raw_data

    def __getitem__(self, index):
        return (
            self.data_tensor[index],
            self.target_tensor[index],
            self.length_tensor[index],
            self.raw_data[index],
        )

    def __len__(self):
        return self.data_tensor.size(0)


class GeoTimeClassify:
    def __init__(self, number_of_samples):
        self.model = LSTMClassifier(vocab_size=89, embedding_dim=128, hidden_dim=32, output_size=71)
        self.model.load_state_dict(
            torch.load(pkg_resources.resource_stream(__name__, 'models/LSTM_RNN_Geotime_Classify_v_0.09_dict.pth')))
        self.model.eval()
        self.number_of_random_samples = number_of_samples
        #       prediction tensors with the best match being less than predictionLimit will not be returned
        self.predictionLimit = -4.5
        self.country_lookup = pd.read_csv(pkg_resources.resource_stream(__name__, 'datasets/country.csv'),
                                          encoding='latin-1')
        self.city_lookup = pd.read_csv(pkg_resources.resource_stream(__name__, 'datasets/city.csv'),
                                       encoding='latin-1')
        self.city_lookup = np.asarray(self.city_lookup["city"])
        self.state_lookup = pd.read_csv(pkg_resources.resource_stream(__name__, 'datasets/NA_states_provinces.csv'),
                                        encoding='latin-1')
        self.state_lookup = np.asarray(self.state_lookup["state_name"])
        self.country_name = np.asarray(self.country_lookup["country_name"])
        self.iso3_lookup = np.asarray(self.country_lookup["Alpha-3_Code"])
        self.iso2_lookup = np.asarray(self.country_lookup["Alpha-2_Code"])
        self.cont_lookup = pd.read_csv(pkg_resources.resource_stream(__name__, 'datasets/continent_code.csv'),
                                       encoding='latin-1')
        self.cont_lookup = np.asarray(self.cont_lookup["continent_name"])
        self.FakeData = pd.read_csv(pkg_resources.resource_stream(__name__, 'datasets/Fake_data.csv'),
                                    encoding='latin-1')

        self.day_of_week = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        self.month_of_year = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sept",
            "Oct",
            "Nov",
            "Dec",
        ]
        self.tag2id = defaultdict(
            int,
            {
                "city": 0,
                "first_name": 1,
                "geo": 2,
                "percent": 3,
                "year": 4,
                "ssn": 5,
                "language_name": 6,
                "country_name": 7,
                "phone_number": 8,
                "month": 9,
                "zipcode": 10,
                "iso8601": 11,
                "paragraph": 12,
                "pyfloat": 13,
                "email": 14,
                "prefix": 15,
                "pystr": 16,
                "isbn": 17,
                "boolean": 18,
                "country_code": 19,
                "country_GID": 20,
                "continent": 21,
                "date_%Y-%m-%d": 22,
                "date_%Y_%m_%d": 23,
                "date_%Y/%m/%d": 24,
                "date_%Y.%m.%d": 25,
                "date_%m-%d-%Y": 26,
                "date_%m-%d-%y": 27,
                "date_%m_%d_%Y": 28,
                "date_%m_%d_%y": 29,
                "date_%m/%d/%Y": 30,
                "date_%m/%d/%y": 31,
                "date_%m.%d.%Y": 32,
                "date_%m.%d.%y": 33,
                "date_%d-%m-%Y": 34,
                "date_%d-%m-%y": 35,
                "date_%d_%m_%Y": 36,
                "date_%d_%m_%y": 37,
                "date_%d/%m/%Y": 38,
                "date_%d/%m/%y": 39,
                "date_%d.%m.%Y": 40,
                "date_%d.%m.%y": 41,
                "date_%Y%m%d": 42,
                "date_%Y%d": 43,
                "date_%Y-%m": 44,
                "date_%Y/%m": 45,
                "date_%Y.%m": 46,
                "day_of_month": 47,
                "day_of_week": 48,
                "date_long_dmdy": 49,
                "date_long_mdy": 50,
                "date_long_dmdyt": 51,
                "date_long_mdyt_m": 52,
                "date_long_dmonthY": 53,
                "date_long_dmonthy": 54,
                "city_suffix": 55,
                "month_name": 56,
                "boolean_letter": 57,
                'date_%Y-%m-%d %H:%M:%S': 58,
                'date_%Y/%m/%d %H:%M:%S': 59,
                'date_%Y_%m_%d %H:%M:%S': 60,
                'date_%Y.%m.%d %H:%M:%S': 61,
                'date_%m-%d-%Y %H:%M:%S': 62,
                'date_%m/%d/%Y %H:%M:%S': 63,
                'date_%m_%d_%Y %H:%M:%S': 64,
                'date_%m.%d.%Y %H:%M:%S': 65,
                'date_%d-%m-%Y %H:%M:%S': 66,
                'date_%d/%m/%Y %H:%M:%S': 67,
                'date_%d_%m_%Y %H:%M:%S': 68,
                'date_%d.%m.%Y %H:%M:%S': 69,
                'unix_time': 70
            },
        )
        self.n_categories = len(self.tag2id)
        self.token_set = {
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
            "'",
            ",",
            ".",
            ";",
            "*",
            "!",
            "@",
            "#",
            "$",
            "%",
            "^",
            "&",
            "(",
            ")",
            "_",
            "=",
            "-",
            ":",
            "+",
            "/",
            "\\",
            "*",
        }
        self.token2id = defaultdict(
            int,
            {
                "PAD": 0,
                "UNK": 1,
                "a": 2,
                "b": 3,
                "c": 4,
                "d": 5,
                "e": 6,
                "f": 7,
                "g": 8,
                "h": 9,
                "i": 10,
                "j": 11,
                "k": 12,
                "l": 13,
                "m": 14,
                "n": 15,
                "o": 16,
                "p": 17,
                "q": 18,
                "r": 19,
                "s": 20,
                "t": 21,
                "u": 22,
                "v": 23,
                "w": 24,
                "x": 25,
                "y": 26,
                "z": 27,
                "A": 28,
                "B": 29,
                "C": 30,
                "D": 31,
                "E": 32,
                "F": 33,
                "G": 34,
                "H": 35,
                "I": 36,
                "J": 37,
                "K": 38,
                "L": 39,
                "N": 40,
                "O": 41,
                "P": 42,
                "Q": 43,
                "R": 44,
                "S": 45,
                "T": 46,
                "U": 47,
                "V": 48,
                "W": 49,
                "X": 50,
                "Y": 51,
                "Z": 52,
                "1": 53,
                "2": 54,
                "3": 55,
                "4": 56,
                "5": 57,
                "6": 58,
                "7": 59,
                "8": 60,
                "9": 61,
                "0": 62,
                "'": 63,
                ",": 64,
                ".": 65,
                ";": 66,
                "*": 67,
                "!": 68,
                "@": 68,
                "#": 70,
                "$": 71,
                "%": 72,
                "^": 73,
                "&": 74,
                "(": 75,
                ")": 76,
                "_": 77,
                "=": 78,
                "-": 79,
                ":": 80,
                "+": 81,
                "/": 82,
                "\\": 83,
            },
        )

    class PaddedTensorDataset(Dataset):
        #     """Dataset wrapping data, target and length tensors.

        #     Each sample will be retrieved by indexing both tensors along the first
        #     dimension.

        #     Arguments:
        #         data_tensor (Tensor): contains sample data.
        #         target_tensor (Tensor): contains sample targets (labels).
        #         length (Tensor): contains sample lengths.
        #         raw_data (Any): The data that has been transformed into tensor, useful for debugging
        #     """

        def __init__(self, data_tensor, target_tensor, length_tensor, raw_data):
            assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
            self.data_tensor = data_tensor
            self.target_tensor = target_tensor
            self.length_tensor = length_tensor
            self.raw_data = raw_data

        def __getitem__(self, index):
            return (
                self.data_tensor[index],
                self.target_tensor[index],
                self.length_tensor[index],
                self.raw_data[index],
            )

        def __len__(self):
            return self.data_tensor.size(0)

    def vectorized_string(self, string):
        return [
            self.token2id[token] if token in self.token2id else self.token2id["UNK"]
            for token in str(string)
        ]

    def vectorized_array(self, array):
        vecorized_array = []
        for stringValue in array:
            vecorized_array.append(self.vectorized_string(str(stringValue)))
        return vecorized_array

    def pad_sequences(self, vectorized_seqs, seq_lengths):
        # create a zero matrix
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()

        # fill the index
        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        return seq_tensor

    def create_dataset(self, data, batch_size=1):
        vectorized_seqs = self.vectorized_array(data)
        seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
        seq_tensor = self.pad_sequences(vectorized_seqs, seq_lengths)
        target_tensor = torch.LongTensor([self.tag2id[y] for y in data])
        raw_data = [x for x in data]
        return DataLoader(
            PaddedTensorDataset(seq_tensor, target_tensor, seq_lengths, raw_data),
            batch_size=batch_size,
        )

    def sort_batch(self, batch, targets, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return seq_tensor.transpose(0, 1), target_tensor, seq_lengths

    def evaluate_test_set(self, test):
        y_pred = list()
        all_predictionsforValue = []

        for batch, targets, lengths, raw_data in self.create_dataset(
                test, batch_size=1
        ):
            batch, targets, lengths = self.sort_batch(batch, targets, lengths)
            pred = self.model(torch.autograd.Variable(batch), lengths.cpu().numpy())
            pred_idx = torch.max(pred, 1)[1]

            def get_key(val):
                for key, value in self.tag2id.items():
                    if val == value:
                        return {"top_pred": key, "tensor": pred, "pred_idx": pred_idx}

            all_predictionsforValue.append(get_key(pred_idx[0]))
        return all_predictionsforValue

    def read_in_csv(self, path):
        self.df = pd.read_csv(path)
        df = pd.read_csv(path)
        return df

    def get_arrayOfValues_df(self, df):
        column_value_object = {}

        for column in df.columns:
            guesses = []
            column_value_object[column] = []
            for _ in range(1, self.number_of_random_samples):
                random_values = str(np.random.choice(df[column]))
                random_col = column
                column_value_object[column].append(random_values)

        return column_value_object

    def get_arrayOfValues_df_enhanced(self, df, index_remove):
        column_value_object = {}

        for i, column in enumerate(df.columns):
            guesses = []
            column_value_object[column] = []
            if i in index_remove:

                pass

            else:
                for _ in range(1, self.number_of_random_samples):
                    random_values = str(np.random.choice(df[column].dropna()))
                    random_col = column
                    column_value_object[column].append(random_values)
        print('column_val', column_value_object)
        return column_value_object

    def averaged_predictions(self, all_predictions):
        all_arrays = []
        for pred in all_predictions:
            all_arrays.append(pred["tensor"].detach().numpy())

        out = np.mean(all_arrays, axis=0)
        maxValue = np.amax(out)

        def get_key(val):
            for key, value in self.tag2id.items():
                if val == value:
                    return key

        topcat = get_key(np.argmax(out))

        return {
            "averaged_tensor": out,
            "averaged_top_category": {True: "None", False: topcat}[
                maxValue < self.predictionLimit
                ],
        }

    def return_data(self):
        return self.cont_lookup

    def predictions(self, path_to_csv):
        print('Start LSTM predictions ...')

        class LSTMClassifier(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
                super(LSTMClassifier, self).__init__()

                self.embedding_dim = embedding_dim
                self.hidden_dim = hidden_dim
                self.vocab_size = vocab_size

                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

                self.hidden2out = nn.Linear(hidden_dim, output_size)
                self.softmax = nn.LogSoftmax(dim=1)

                self.dropout_layer = nn.Dropout(p=0.2)

            def init_hidden(self, batch_size):
                return (
                    autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                    autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                )

            def forward(self, batch, lengths):
                self.hidden = self.init_hidden(batch.size(-1))

                embeds = self.embedding(batch)
                packed_input = pack_padded_sequence(embeds, lengths)
                outputs, (ht, ct) = self.lstm(packed_input, self.hidden)
                # ht is the last hidden state of the sequences
                # ht = (1 x batch_size x hidden_dim)
                # ht[-1] = (batch_size x hidden_dim)
                output = self.dropout_layer(ht[-1])
                output = self.hidden2out(output)
                output = self.softmax(output)

                return output

        df = self.read_in_csv(path=path_to_csv)
        print(df.columns)
        # self.model = torch.load(self.cwd + '/geotime_classify/models/LSTM_RNN_Geotime_Classify_v_0.07.pth')
        column_value_object = self.get_arrayOfValues_df(df)
        self.column_value_object = column_value_object
        predictionList = []
        for column in column_value_object:
            try:
                all_predictions = self.evaluate_test_set(
                    column_value_object[column]
                )
                avg_predictions = self.averaged_predictions(all_predictions)
                predictionList.append(
                    {
                        "column": column,
                        "values": column_value_object[column],
                        "avg_predictions": avg_predictions,
                        "model_predictions": all_predictions,
                    }
                )
            except Exception as e:
                print(e)

        self.predictionList = predictionList
        return self.predictionList


    def predictions_enhanced(self, df, index_remove):
        print('Start LSTM predictions ...')

        class LSTMClassifier(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
                super(LSTMClassifier, self).__init__()

                self.embedding_dim = embedding_dim
                self.hidden_dim = hidden_dim
                self.vocab_size = vocab_size

                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

                self.hidden2out = nn.Linear(hidden_dim, output_size)
                self.softmax = nn.LogSoftmax(dim=1)

                self.dropout_layer = nn.Dropout(p=0.2)

            def init_hidden(self, batch_size):
                return (
                    autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                    autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                )

            def forward(self, batch, lengths):
                self.hidden = self.init_hidden(batch.size(-1))

                embeds = self.embedding(batch)
                packed_input = pack_padded_sequence(embeds, lengths)
                outputs, (ht, ct) = self.lstm(packed_input, self.hidden)
                # ht is the last hidden state of the sequences
                # ht = (1 x batch_size x hidden_dim)
                # ht[-1] = (batch_size x hidden_dim)
                output = self.dropout_layer(ht[-1])
                output = self.hidden2out(output)
                output = self.softmax(output)

                return output

        #df = df.drop(df.columns[index_remove], axis=1)
        # self.model = torch.load(self.cwd + '/geotime_classify/models/LSTM_RNN_Geotime_Classify_v_0.07.pth')
        column_value_object = self.get_arrayOfValues_df_enhanced(df, index_remove)
        self.column_value_object = column_value_object
        predictionList = []
        for i, column in enumerate(column_value_object):
            if i in index_remove:
                predictionList.append(
                    {
                        "column": column,
                        "values": "Skipped",
                        "avg_predictions": "Skipped",
                        "model_predictions": "Skipped",
                    }
                )
            else:

                try:
                    all_predictions = self.evaluate_test_set(
                        column_value_object[column]
                    )
                    avg_predictions = self.averaged_predictions(all_predictions)
                    predictionList.append(
                        {
                            "column": column,
                            "values": column_value_object[column],
                            "avg_predictions": avg_predictions,
                            "model_predictions": all_predictions,
                        }
                    )
                except Exception as e:
                    print(e)
        # print('predictionList', predictionList)
        self.predictionList = predictionList
        return self.predictionList

    # match two words return true based on ratio
    def fuzzyMatch(self, word1, word2, ratio=95):
        Ratio = fuzz.ratio(word1.lower(), word2.lower())

        if Ratio > ratio:
            return True

    def fuzzyRatio(self, word1, word2, ratio=95):
        Ratio = fuzz.ratio(word1.lower(), word2.lower())
        if Ratio > ratio:
            return True, Ratio

    def assign_heuristic_function_enhanced(self, predictions, fuzzyMatched):
        c_lookup = self.city_lookup
        state_lookup = self.state_lookup
        country_lookup = self.country_name
        iso3_lookup = self.iso3_lookup
        iso2_lookup = self.iso2_lookup
        cont_lookup = self.cont_lookup

        def build_return_object(format, util, dayFirst):
            classifiedObj=Classification(category='time', subcategory= 'date', format= format,
                    match_type= ['LSTM'], Parser= "Util", DayFirst= dayFirst)
            print('clasobj', classifiedObj)
            #return classifiedObj
            return {'category': 'time', 'subcategory': 'date', 'format': format,
                    "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}

        def build_return_standard_object(category, subcategory, match_type):
            classifiedObj = Classification(category=category, subcategory=subcategory, format=None,
                                           match_type=[match_type], Parser=None, DayFirst=None)
            print('calls 2', classifiedObj)
            #return classifiedObj
            return {'category': category, 'subcategory': subcategory, 'format': None,
                    "match_type": [match_type], "Parser": None, "DayFirst": None}


        def none_f(values):
            return build_return_standard_object(category=None, subcategory=None, match_type=None)


        def Skipped_f(column, fuzzyMatched):
            category=None
            subcategory=None
            match_type=None
            for match in fuzzyMatched:
                if column == match['header']:
                    match_type='fuzzy'
                    category="geo"
                    subcategory=match['value']

            return {'category': category, 'subcategory': subcategory, 'format': None,
                    "match_type": [match_type], "Parser": None, "DayFirst": None}

        def city_f(values):
            print("Start city validation ...")
            city_match_bool = []
            if len(values) < 40:
                subsample = 5
            else:
                # subsample = int(round(.2*len(values),0))
                subsample = 5

            print(subsample)
            count =0
            passed=0
            while passed < 2 and not count >= subsample:
                count += 1
                try:
                    match = fuzzywuzzy.process.extractOne(
                        random.choice(values), c_lookup, scorer=fuzz.token_sort_ratio
                    )
                    if match is not None:
                        if match[1] > 90:
                            city_match_bool.append(True)
                            passed += 1
                            print(passed)
                except Exception as e:
                    print(e)

            # print("city_match_bool", city_match_bool)
            if np.count_nonzero(city_match_bool) >= 2:
                return build_return_standard_object(category='geo', subcategory='city name', match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def state_f(values):

            print("Start state validation ...")
            state_match_bool = []
            subsample=len(values)
            count = 0
            passed = 0
            while passed < 5 and not count >= subsample:
                count += 1
                try:
                    match = fuzzywuzzy.process.extractOne(
                        values[count], state_lookup, scorer=fuzz.token_sort_ratio
                    )
                    if match is not None:
                        if match[1] > 90:
                            state_match_bool.append(True)
                            passed += 1
                            print(passed)
                except Exception as e:
                    print(e)

            if np.count_nonzero(state_match_bool) >= 5:
                return build_return_standard_object(category='geo', subcategory='state name', match_type='LSTM')
            else:
                print("Start cities validation ...")
                return city_f(values)

        def country_f(values):

            print("Start country validation ...")
            country_match_bool = []
            subsample = len(values)
            count = 0
            passed = 0
            while passed < 5 and not count >= subsample:
                count += 1
                try:
                    match = fuzzywuzzy.process.extractOne(
                        values[count], country_lookup, scorer=fuzz.token_sort_ratio
                    )
                    if match is not None:
                        if match[1] > 90:
                            country_match_bool.append(True)
                            passed += 1
                            print(passed)
                except Exception as e:
                    print(e)

            if np.count_nonzero(country_match_bool) >= 5:
                return build_return_standard_object(category='geo', subcategory='country name', match_type='LSTM')
            else:
                return state_f(values)

        def country_iso3(values):
            print("Start iso3 validation ...")
            ISO_in_lookup = []

            for iso in values:
                for cc in iso3_lookup:
                    try:
                        ISO_in_lookup.append(
                            self.fuzzyMatch(str(iso), str(cc), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(ISO_in_lookup) >= (len(values) * 0.65):
                return build_return_standard_object(category='geo', subcategory='ISO3', match_type='LSTM')
            else:
                return country_iso2(values)

        def country_iso2(values):
            print("Start iso2 validation ...")
            ISO2_in_lookup = []
            for iso in values:
                for cc in iso2_lookup:
                    try:

                        ISO2_in_lookup.append(
                            self.fuzzyMatch(str(iso), str(cc), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(ISO2_in_lookup) >= (len(values) * 0.65):

                return build_return_standard_object(category='geo', subcategory='ISO2', match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def continent_f(values):
            print("Start continent validation ...")
            cont_in_lookup = []

            for cont in values:
                for c in cont_lookup:
                    try:
                        cont_in_lookup.append(
                            self.fuzzyMatch(str(cont), str(c), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(cont_in_lookup) >= (len(values) * 0.65):

                return build_return_standard_object(category='geo', subcategory='continent', match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def geo_f(values):
            print("Start geo validation ...")
            geo_valid = []
            percent_array = []
            for geo in values:
                try:
                    if 180 >= float(geo) >= -180:
                        if 90 >= float(geo) >= -90:
                            geo_valid.append("latlng")
                            if 1 >= float(geo) >= -1:
                                percent_array.append("true")

                        else:
                            geo_valid.append("lng")
                    else:
                        geo_valid.append("failed")
                except Exception as e:
                    print(e)

            if "failed" in geo_valid:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)
            elif len(percent_array) >= len(values) * 0.95:
                return build_return_standard_object(category='number/geo',
                                                    subcategory="Unknown-mostly between -1 and 1", match_type='LSTM')
            elif "lng" in geo_valid:
                return build_return_standard_object(category='geo', subcategory="longitude", match_type='LSTM')
            elif "latlng" in geo_valid:
                return build_return_standard_object(category='geo', subcategory="latitude", match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def year_f(values):
            print("Start year validation ...")
            year_values_valid = []
            years_failed = []
            strange_year = []
            for year in values:
                try:
                    if str.isdigit(str(year)):
                        if 1300 < int(year) < 2500:
                            year_values_valid.append("True")
                        else:
                            strange_year.append("Maybe")
                    else:
                        years_failed.append("Failed")
                except Exception as e:
                    print(e)

            if len(years_failed) > len(values) * 0.15:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)
            elif len(strange_year) > len(values) * 15:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)
            elif len(year_values_valid) > len(values) * 0.75:
                return build_return_object(format="%Y", util=None, dayFirst=None)

        def bool_f(values):
            print("Start boolean validation ...")
            bool_arr = ["true", "false", "T", "F"]
            bool_array = []
            for bools in values:
                for b in bool_arr:
                    try:
                        bool_array.append(self.fuzzyMatch(bools, b, ratio=85))
                    except Exception as e:
                        print(e)

            if np.count_nonzero(bool_array) >= (len(values) * 0.85):

                return build_return_standard_object(category='Boolean', subcategory=None, match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def bool_letter_f(values):
            print('Start boolean validation ...')
            bool_arr = ['t', 'f', 'T', 'F']
            bool_array = []
            for bools in values:
                for b in bool_arr:
                    try:
                        bool_array.append(self.fuzzyMatch(bools, b, ratio=98))
                    except Exception as e:
                        print(e)

            if np.count_nonzero(bool_array) >= (len(values) * .85):

                return build_return_standard_object(category='Boolean', subcategory=None, match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)




        def dayFirstCheck(values, separator, shortYear, yearLoc):
            # only works for 4 number year
            for date in values:
                try:
                    arr = date.split(separator)
                    if shortYear:
                        if yearLoc == 0:
                            if int(arr[1]) > 12:
                                return True
                        else:
                            if int(arr[0]) > 12:
                                return True
                    else:
                        if len(arr[0]) == 4:
                            if int(arr[1]) > 12:
                                return True
                        else:
                            if int(arr[0]) > 12:
                                return True
                except Exception as e:
                    print("error occurred", e)

            return False

        def date_arrow(values, separator):
            utils_array = []
            for date in values:
                try:
                    dateArrow = arrow.get(str(date), normalize_whitespace=True).datetime

                    if isinstance(dateArrow, datetime.date):
                        utils_array.append("true")
                    else:
                        print("Not a valid date format")
                except Exception as e:
                    print(e, "Error from Arrow: Date had an error")
            return utils_array

        def date_arrow_1(values):
            array_valid = date_arrow(values, separator="none")
            if len(array_valid) > len(values) * 0.85:
                return build_return_object(format="%Y%d", util='arrow', dayFirst=None)
            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_arrow_2(values):
            # array_valid = date_arrow(values, separator="-")
            monthFormat = month_MMorM(values, separator='-', loc=1)
            allMonthVals = []
            for val in values:
                monthval = val.split('-')[1]
                allMonthVals.append(monthval)
            validMonth = month_day_f(allMonthVals)
            if validMonth["subcategory"] == "date" and validMonth["format"] == "%m" or validMonth[
                "subcategory"] == 'date' and validMonth['format'] == "%-m":
                return build_return_object(format="%Y-" + monthFormat, util='arrow', dayFirst=None)
            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_arrow_3(values):
            array_valid = date_arrow(values, separator="/")
            monthFormat = month_MMorM(values, separator='/', loc=1)
            allMonthVals = []
            for val in values:
                monthval = val.split('/')[1]
                allMonthVals.append(monthval)
            validMonth = month_day_f(allMonthVals)
            if len(array_valid) > len(values) * 0.85:
                return build_return_object(format="%Y/" + monthFormat, util='arrow', dayFirst=None)
            elif validMonth["subcategory"] == 'date' and validMonth['format'] == "%m" or validMonth[
                'subcategory'] == 'date' and validMonth['format'] == '%-m':
                return build_return_object(format="%Y/" + monthFormat, util='arrow', dayFirst=None)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_arrow_4(values):
            array_valid = date_arrow(values, separator=".")
            monthFormat = month_MMorM(values, separator='.', loc=1)
            allMonthVals = []
            for val in values:
                try:
                    monthval = val.split('.')[1]
                    allMonthVals.append(monthval)
                except Exception as e:
                    print(e)

            validMonth = month_day_f(allMonthVals)

            if len(array_valid) > len(values) * 0.75:
                return build_return_object(format="%Y." + monthFormat, util='arrow', dayFirst=None)

            elif validMonth['subcategory'] == 'date' and validMonth['format'] == '%m' or validMonth[
                'subcategory'] == 'date' and validMonth['format'] == '%-m':
                return build_return_object(format="%Y." + monthFormat, util='arrow', dayFirst=None)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util(values, separator, shortyear, yearloc):

            util_dates = []
            if separator != "none":
                dayFirst = dayFirstCheck(values, separator, shortYear=shortyear, yearLoc=yearloc)
            else:
                dayFirst = False

            for date in values:
                try:

                    dateUtil = dateutil.parser.parse(str(date), dayfirst=dayFirst)
                    if isinstance(dateUtil, datetime.date):
                        util_dates.append({"value": date, "standard": dateUtil})
                    else:
                        pass
                except Exception as e:
                    print(e)

            return util_dates, dayFirst

        def day_ddOrd(values, separator, loc):
            dayFormat = '%-d'
            for d in values:
                try:
                    if separator is None:
                        if d[0] == '0':
                            dayFormat = '%d'
                    else:
                        d_value = d.split(separator)[loc]
                        if d_value[0] == '0':
                            dayFormat = '%d'
                except Exception as e:
                    print(e)
            return dayFormat

        def month_MMorM(values, separator, loc):
            monthFormat = '%-m'
            for d in values:
                try:
                    if separator is None:
                        if d[0] == '0':
                            monthFormat = '%m'
                    else:
                        d_value = d.split(separator)[loc]
                        if d_value[0] == '0':
                            monthFormat = '%m'
                except Exception as e:
                    print(e)
            return monthFormat

        def hour_hOrH(values, separator, loc_hms):
            hourFormat = '%-H'
            for d in values:

                if separator is None:
                    if d[0] == '0':
                        hourFormat = '%H'
                else:
                    hms = d.split(' ')[-1]
                    hms = hms.split(separator)[loc_hms]

                    if hms[0] == '0':
                        hourFormat = '%H'
            return hourFormat

        def minute_mOrM(values, separator, loc_hms):
            minuteFormat = '%-M'
            for d in values:
                if separator is None:
                    if d[0] == '0':
                        minuteFormat = '%M'
                else:
                    hms = d.split(' ')[-1]
                    hms = hms.split(separator)[loc_hms]
                    if hms[0] == '0':
                        minuteFormat = '%M'
            return minuteFormat

        def second_sOrS(values, separator, loc_hms):
            secondFormat = '%-S'
            for d in values:
                if separator is None:
                    if d[0] == '0':
                        secondFormat = '%S'
                else:
                    hms = d.split(' ')[-1]
                    hms = hms.split(separator)[loc_hms]
                    if hms[0] == '0':
                        secondFormat = '%S'
            return secondFormat

        # currently can't handle yyyy-dd-mm HH:mm:ss
        def iso_time(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)

            if len(array_valid) > len(values) * 0.85:
                # '1996-03-20T07:46:39'
                # '1998-08-15T08:43:22'
                # '1972-10-03T05:52:26'
                # '1987-08-15T09:51:25'
                return build_return_object(format="%Y-%m-%dT%H%M%S", util='Util', dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_1(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)

            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "-" + monthFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "-" + dayFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_2(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)

            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "-" + monthFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "-" + dayFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_3(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)

            if dayFirst:
                return build_return_object(format=dayFormat + "_" + monthFormat + "_%Y", util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format=monthFormat + "_" + dayFormat + "_%Y", util='Util', dayFirst=dayFirst)

        def date_util_4(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)
            if dayFirst:
                return build_return_object(format=dayFormat + "_" + monthFormat + "_%y", util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format=monthFormat + "_" + dayFormat + "_%y", util='Util', dayFirst=dayFirst)

        def date_util_5(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)

            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "/" + monthFormat + "/%Y", util='Util',
                                               dayFirst=dayFirst)
                else:
                    return build_return_object(format=monthFormat + "/" + dayFormat + "/%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_6(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "/" + monthFormat + "/%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "/" + dayFormat + "/%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_7(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    return build_return_object(format=dayFormat + "." + monthFormat + ".%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "." + dayFormat + ".%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_8(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "." + monthFormat + ".%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "." + dayFormat + ".%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_9(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "-" + monthFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "-" + dayFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_10(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFormat:
                    return build_return_object(format=dayFormat + "-" + monthFormat + "-%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "-" + dayFormat + "-%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_11(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)

            if dayFirst:

                return build_return_object(format=dayFormat + "_" + monthFormat + "_%Y", util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format=monthFormat + "_" + dayFormat + "_%Y", util='Util', dayFirst=dayFirst)

        def date_util_12(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)

            if dayFirst:

                return build_return_object(format=dayFormat + "_" + monthFormat + "_%y", util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format=monthFormat + "_" + dayFormat + "_%y", util='Util', dayFirst=dayFirst)

        def date_util_13(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    return build_return_object(format=dayFormat + '/' + monthFormat + '/' + "/%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + '/' + dayFormat + '/' + "/%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_14(values):

            array_valid, dayFirst = date_util(values, separator="/", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    return build_return_object(format=dayFormat + '/' + monthFormat + "/%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + '/' + dayFormat + "/%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_15(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    return build_return_object(format=dayFormat + '.' + monthFormat + ".%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + '.' + dayFormat + ".%Y", util='Util',
                                               dayFirst=dayFirst)


            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_16(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': dayFormat + '.' + monthFormat + ".%y",
                    #         "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format=dayFormat + '.' + monthFormat + ".%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': monthFormat + '.' + dayFormat + ".%y",
                    # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format=monthFormat + '.' + dayFormat + ".%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_17(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=2)
                monthFormat = month_MMorM(values, separator='_', loc=1)

            if dayFirst:
                #    return {'category': 'time', 'subcategory': 'date', 'format': "%Y_"+ dayFormat+"_"+monthFormat,
                # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                return build_return_object(format="%Y_" + dayFormat + "_" + monthFormat, util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format="%Y_" + monthFormat + "_" + dayFormat, util='Util', dayFirst=dayFirst)

            #    return {'category': 'time', 'subcategory': 'date', 'format': "%Y_" + monthFormat + "_" + dayFormat,
            # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}

        def date_util_18(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=2)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    #    return  {'category': 'time', 'subcategory': 'date', 'format': "%Y."+dayFormat+"."+monthFormat,
                    # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format="%Y." + dayFormat + "." + monthFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format="%Y." + monthFormat + "." + dayFormat, util='Util',
                                               dayFirst=dayFirst)

                #    return  {'category': 'time', 'subcategory': 'date', 'format': "%Y."+monthFormat+"."+dayFormat,
                # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_19(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=2)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format': "%Y-" + dayFormat+"-"+monthFormat,
                    # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format="%Y-" + dayFormat + "-" + monthFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format="%Y-" + monthFormat + "-" + dayFormat, util='Util',
                                               dayFirst=dayFirst)

                #    return  {'category': 'time', 'subcategory': 'date',
                #    'format': "%Y-" + monthFormat + "-" + dayFormat,
                # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_20(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=2)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return  {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y/"+dayFormat+"/"+monthFormat, "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format="%Y/" + dayFormat + "/" + monthFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format="%Y/" + monthFormat + "/" + dayFormat, util='Util',
                                               dayFirst=dayFirst)

                    # return  {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y/" + monthFormat + "/" + dayFormat, "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_21(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=0)

            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=2)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            # hour min sec format
            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(
                        format="%Y-" + dayFormat + "-" + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(
                        format="%Y-" + monthFormat + "-" + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_23(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=0)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=2)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(
                        format="%Y/" + dayFormat + "/" + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(
                        format="%Y/" + monthFormat + "/" + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_24(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=0)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=2)
                monthFormat = month_MMorM(values, separator='_', loc=1)

            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if dayFirst:
                # return {'category': 'time', 'subcategory': 'date',
                # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                # "Parser": "Util", "DayFirst": dayFirst}
                return build_return_object(
                    format="%Y_" + dayFormat + "_" + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                    util='Util',
                    dayFirst=dayFirst)

            else:
                #    return {'category': 'time', 'subcategory': 'date',
                #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                # "Parser": "Util",  "DayFirst": dayFirst}
                return build_return_object(
                    format="%Y_" + monthFormat + "_" + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                    util='Util',
                    dayFirst=dayFirst)

        def date_util_25(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=0)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=2)
                monthFormat = month_MMorM(values, separator='.', loc=1)

            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(
                        format="%Y." + dayFormat + "." + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(
                        format="%Y." + monthFormat + "." + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_26(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)

            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(
                        format=dayFormat + "-" + monthFormat + '-%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(
                        format=monthFormat + "-" + dayFormat + '-%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_27(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)
            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(
                        format=dayFormat + "/" + monthFormat + '/%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(
                        format=monthFormat + "/" + dayFormat + '/%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_28(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=2)
            print('dayf', dayFirst)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)
            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if dayFirst:
                # return {'category': 'time', 'subcategory': 'date',
                # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                # "Parser": "Util", "DayFirst": dayFirst}
                return build_return_object(
                    format=dayFormat + "_" + monthFormat + '_%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                    util='Util',
                    dayFirst=dayFirst)

            else:
                #    return {'category': 'time', 'subcategory': 'date',
                #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                # "Parser": "Util",  "DayFirst": dayFirst}
                return build_return_object(
                    format=monthFormat + "_" + dayFormat + '_%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                    util='Util',
                    dayFirst=dayFirst)

        def date_util_29(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(
                        format=dayFormat + "." + monthFormat + '.%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(
                        format=monthFormat + "." + dayFormat + '.%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
                        util='Util',
                        dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_22(values):
            array_valid = []
            for v in values:
                try:
                    if int(v) < -5364601438 or int(v) > 4102506000:
                        array_valid.append('failed')
                    elif len(v) <= 13:
                        array_valid.append('valid')
                    else:
                        array_valid.append('failed')
                except Exception as e:
                    array_valid.append('failed')
                    print(e)

            if 'failed' in array_valid:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)
            else:
                return build_return_object(format='Unix Timestamp', util=None, dayFirst=None)

                # return {'category': 'time', 'subcategory': 'date', 'format': 'Unix Timestamp',
                # "match_type": ['LSTM'],
                #         "Parser": "Util"}

        def date_long_1(values):
            #              #  01 April 2008
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=0)
            if len(array_valid) > len(values) * 0.85:
                return build_return_object(format=dayFormat + " %B %Y", util=None, dayFirst=None)

                # return {'category': 'time', 'subcategory': 'date', 'format':dayFormat+" %B %Y",
                # "match_type": ['LSTM'],  "Parser": "Util"}

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_2(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=0)
            if len(array_valid) > len(values) * 0.85:
                #                 02 April 20
                #                    dd/LLLL/yy
                return build_return_object(format=dayFormat + " %B %y", util=None, dayFirst=None)

                # return  {'category': 'time', 'subcategory': 'date', 'format': dayFormat+" %B %y",
                # "match_type": ['LSTM'], "Parser": "Util"}

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_3(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=2)
            if len(array_valid) > len(values) * 0.85:
                return build_return_object(format="%A, %B " + dayFormat + ",%y", util=None, dayFirst=None)
                # return  {'category': 'time', 'subcategory': 'date', 'format': "%A, %B "+dayFormat+",%y",
                # "match_type": ['LSTM'], "Parser": "Util"}

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_4(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=1)
            if len(array_valid) > len(values) * 0.85:
                #                 April 10, 2008
                #                 LLLL dd, y
                return build_return_object(format="%B " + dayFormat + ", %Y", util=None, dayFirst=None)

                # return  {'category': 'time', 'subcategory': 'date', 'format':"%B "+dayFormat+", %Y" ,
                # "match_type": ['LSTM'], "Parser": "Util"}

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_5(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=2)
            if len(array_valid) > len(values) * 0.85:
                #  Thursday, April 10, 2008 6:30:00 AM
                #                 EEEE, LLLL dd,yy HH:mm:ss

                return build_return_object(format="%A, %B " + dayFormat + ",%y HH:mm:ss", util=None, dayFirst=None)
                # return {'category': 'time', 'subcategory': 'date','format': "%A, %B "+dayFormat+",%y HH:mm:ss",
                # "match_type": ['LSTM'],"Parser": "Util"}
            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_6(values):

            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)

            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    #         'format': dayFormat + "/" + monthFormat + "/%y HH:mm", "match_type": ['LSTM']}
                    return build_return_object(format=dayFormat + "/" + monthFormat + "/%y HH:mm", util=None,
                                               dayFirst=None)

                #              03/23/21 01:55 PM
                #                 MM/dd/yy HH:mm
                else:
                    return build_return_object(format=monthFormat + "/" + dayFormat + "/%y HH:mm", util=None,
                                               dayFirst=None)
                    # return {'category': 'time', 'subcategory': 'date', 'format':monthFormat+"/"+dayFormat+"/%y HH:mm", "match_type": ['LSTM'] }
            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def month_day_f(values):
            month_day_results = []
            for i, md in enumerate(values):
                try:
                    if str.isdigit(md):
                        if 12 >= int(md) >= 1:
                            month_day_results.append("month_day")
                        elif 12 < int(md) <= 31:
                            month_day_results.append("day")
                        else:
                            month_day_results.append("failed")
                    else:
                        print("Not a valid digit")
                except Exception as e:
                    print(e)

            if "failed" in month_day_results:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)
            elif "day" in month_day_results:
                # return {'category': 'time', 'subcategory': 'date', 'format':day_ddOrd(values, separator=None, loc=None), "match_type": ['LSTM'] }
                return build_return_object(format=day_ddOrd(values, separator=None, loc=None), util=None, dayFirst=None)
            elif "month_day" in month_day_results:
                # return {'category': 'time', 'subcategory': 'date', 'format':month_MMorM(values, separator=None, loc=None), "match_type": ['LSTM'] }
                return build_return_object(format=month_MMorM(values, separator=None, loc=None), util=None,
                                           dayFirst=None)
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def month_name_f(values):
            print("Start month validation ...")
            month_array_valid = []
            for month in values:
                for m in self.month_of_year:
                    try:
                        month_array_valid.append(
                            self.fuzzyMatch(str(month), str(m), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(month_array_valid) >= (len(values) * 0.65):
                # return {'category': 'time', 'subcategory': 'date', 'format':'%B', "match_type": ['LSTM'] }
                return build_return_object('%B', util=None, dayFirst=None)
            else:
                return day_name_f(values)

        def day_name_f(values):
            print("Start day validation ...")
            day_array_valid = []
            for day in values:
                for d in self.day_of_week:
                    try:
                        day_array_valid.append(
                            self.fuzzyMatch(str(day), str(d), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(day_array_valid) >= (len(values) * 0.65):

                # return {'category': 'time', 'subcategory': 'date', 'format':'%A', "match_type": ['LSTM'] }
                return build_return_object('%A', util=None, dayFirst=None)
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        functionlist = defaultdict(
            int,
            {
                "None": none_f,
                "Skipped": Skipped_f,
                "country_name": country_f,
                "city": country_f,
                "language_name": country_f,
                "city_suffix": country_f,
                "first_name": none_f,
                "country_GID": country_iso3,
                "country_code": country_iso2,
                "continent": continent_f,
                "geo": geo_f,
                "pyfloat": none_f,
                "percent": none_f,
                "ssn": none_f,
                "phone_number": none_f,
                "zipcode": none_f,
                "paragraph": none_f,
                "email": none_f,
                "prefix": none_f,
                "pystr": none_f,
                "isbn": none_f,
                "boolean": bool_f,
                'boolean_letter': bool_letter_f,
                "iso8601": iso_time,
                "year": year_f,
                "day_of_month": month_day_f,
                "month": month_day_f,
                "month_name": month_name_f,
                "day_of_week": month_name_f,
                "date_%Y%d": date_arrow_1,
                "date_%Y-%m": date_arrow_2,
                "date_%Y/%m": date_arrow_3,
                "date_%Y.%m": date_arrow_4,
                "date_%m-%d-%Y": date_util_1,
                "date_%m-%d-%y": date_util_2,
                "date_%m_%d_%Y": date_util_3,
                "date_%m_%d_%y": date_util_4,
                "date_%m/%d/%Y": date_util_5,
                "date_%m/%d/%y": date_util_6,
                "date_%m.%d.%Y": date_util_7,
                "date_%m.%d.%y": date_util_8,
                "date_%d-%m-%Y": date_util_9,
                "date_%d-%m-%y": date_util_10,
                "date_%d_%m_%Y": date_util_11,
                "date_%d_%m_%y": date_util_12,
                "date_%d/%m/%Y": date_util_13,
                "date_%d/%m/%y": date_util_14,
                "date_%d.%m.%Y": date_util_15,
                "date_%d.%m.%y": date_util_16,
                "date_%Y_%m_%d": date_util_17,
                "date_%Y.%m.%d": date_util_18,
                "date_%Y-%m-%d": date_util_19,
                "date_%Y/%m/%d": date_util_20,
                "date_%Y-%m-%d %H:%M:%S": date_util_21,
                'date_%Y/%m/%d %H:%M:%S': date_util_23,
                'date_%Y_%m_%d %H:%M:%S': date_util_24,
                'date_%Y.%m.%d %H:%M:%S': date_util_25,
                'date_%m-%d-%Y %H:%M:%S': date_util_26,
                'date_%m/%d/%Y %H:%M:%S': date_util_27,
                'date_%m_%d_%Y %H:%M:%S': date_util_28,
                'date_%m.%d.%Y %H:%M:%S': date_util_29,
                'date_%d-%m-%Y %H:%M:%S': date_util_26,
                'date_%d/%m/%Y %H:%M:%S': date_util_27,
                'date_%d_%m_%Y %H:%M:%S': date_util_28,
                'date_%d.%m.%Y %H:%M:%S': date_util_29,
                'unix_time': date_util_22,
                "date_long_dmonthY": date_long_1,
                "date_long_dmonthy": date_long_2,
                "date_long_dmdy": date_long_3,
                "date_long_mdy": date_long_4,
                "date_long_dmdyt": date_long_5,
                "date_long_mdyt_m": date_long_6,
            },
        )
        final_column_classification = []

        def add_obj(obj, add_objs):
            for property in add_objs:
                obj[property] = add_objs[property]

            return obj

        for pred in predictions:
            print(pred['column'])
            print(pred['values'])
            if pred['values'] == 'Skipped':
                final_column_classification.append(
                    add_obj({"column": pred["column"]}, functionlist['Skipped'](
                        pred['column'],fuzzyMatched
                    ))
                )
            else:
                final_column_classification.append(
                    add_obj({"column": pred["column"]}, functionlist[pred["avg_predictions"]["averaged_top_category"]](
                        self.column_value_object[pred["column"]]
                    )))

        return final_column_classification


    # map function to model prediction category
    def assign_heuristic_function(self, predictions):
        c_lookup = self.city_lookup
        state_lookup = self.state_lookup
        country_lookup = self.country_name
        iso3_lookup = self.iso3_lookup
        iso2_lookup = self.iso2_lookup
        cont_lookup = self.cont_lookup

        def none_f(values):
            return build_return_standard_object(category=None, subcategory=None, match_type=None)


        def city_f(values):
            print("Start city validation ...")
            country_match_bool = []
            if len(values)<40:
                subsample=len(values)
            else:
                # subsample = int(round(.2*len(values),0))
                subsample = 25

            print(subsample)
            for city in values[:subsample]:
                try:
                    match = fuzzywuzzy.process.extractOne(
                        city, c_lookup, scorer=fuzz.token_sort_ratio
                    )
                    if match is not None:
                        if match[1] > 90:
                            country_match_bool.append(True)
                except Exception as e:
                    print(e)
            print("country_match_bool", country_match_bool)

            if np.count_nonzero(country_match_bool) >= (subsample * 0.50):
                return build_return_standard_object(category='geo', subcategory='city name', match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def state_f(values):

            print("Start state validation ...")
            country_match_bool = []

            for state in values:
                for c in state_lookup:
                    try:
                        country_match_bool.append(self.fuzzyMatch(state, c, ratio=85))
                    except Exception as e:
                        print(e)
            if np.count_nonzero(country_match_bool) >= (len(values) * 0.40):
                return build_return_standard_object(category='geo', subcategory='state name', match_type='LSTM')
            else:
                print("Start cities validation ...")
                return city_f(values)

        def country_f(values):

            print("Start country validation ...")
            country_match_bool = []

            for country in values:
                for c in country_lookup:
                    try:
                        country_match_bool.append(self.fuzzyMatch(country, c, ratio=85))
                    except Exception as e:
                        print(e)

            if np.count_nonzero(country_match_bool) >= (len(values) * 0.40):
                return build_return_standard_object(category='geo', subcategory='country name', match_type='LSTM')
            else:
                return state_f(values)

        def country_iso3(values):
            print("Start iso3 validation ...")
            ISO_in_lookup = []

            for iso in values:
                for cc in iso3_lookup:
                    try:
                        ISO_in_lookup.append(
                            self.fuzzyMatch(str(iso), str(cc), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(ISO_in_lookup) >= (len(values) * 0.65):
                return build_return_standard_object(category='geo', subcategory='ISO3',match_type='LSTM')
            else:
                return country_iso2(values)

        def country_iso2(values):
            print("Start iso2 validation ...")
            ISO2_in_lookup = []
            for iso in values:
                for cc in iso2_lookup:
                    try:

                        ISO2_in_lookup.append(
                            self.fuzzyMatch(str(iso), str(cc), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(ISO2_in_lookup) >= (len(values) * 0.65):

                return build_return_standard_object(category='geo', subcategory='ISO2', match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def continent_f(values):
            print("Start continent validation ...")
            cont_in_lookup = []

            for cont in values:
                for c in cont_lookup:
                    try:
                        cont_in_lookup.append(
                            self.fuzzyMatch(str(cont), str(c), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(cont_in_lookup) >= (len(values) * 0.65):

                return build_return_standard_object(category='geo', subcategory='continent',match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def geo_f(values):
            print("Start geo validation ...")
            geo_valid = []
            percent_array = []
            for geo in values:
                try:
                    if 180 >= float(geo) >= -180:
                        if 90 >= float(geo) >= -90:
                            geo_valid.append("latlng")
                            if 1 >= float(geo) >= -1:
                                percent_array.append("true")

                        else:
                            geo_valid.append("lng")
                    else:
                        geo_valid.append("failed")
                except Exception as e:
                    print(e)

            if "failed" in geo_valid:
                return build_return_standard_object(category='number', subcategory=None, match_type=None)
            elif len(percent_array) >= len(values) * 0.95:
                return build_return_standard_object(category='number/geo', subcategory="Unknown-mostly between -1 and 1", match_type='LSTM')
            elif "lng" in geo_valid:
                return build_return_standard_object(category='geo', subcategory="longitude", match_type='LSTM')
            elif "latlng" in geo_valid:
                return build_return_standard_object(category='geo', subcategory="latitude", match_type='LSTM')
            else:
                return build_return_standard_object(category='number', subcategory=None, match_type=None)

        def year_f(values):
            print("Start year validation ...")
            year_values_valid = []
            years_failed = []
            strange_year = []
            for year in values:
                try:
                    if str.isdigit(str(year)):
                        if 1300 < int(year) < 2500:
                            year_values_valid.append("True")
                        else:
                            strange_year.append("Maybe")
                    else:
                        years_failed.append("Failed")
                except Exception as e:
                    print(e)

            if len(years_failed) > len(values) * 0.15:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)
            elif len(strange_year) > len(values) * 15:
                return  build_return_standard_object(category=None, subcategory=None, match_type=None)
            elif len(year_values_valid) > len(values) * 0.75:
                return build_return_object(format="%Y", util=None, dayFirst=None)

        def bool_f(values):
            print("Start boolean validation ...")
            bool_arr = ["true", "false", "T", "F"]
            bool_array = []
            for bools in values:
                for b in bool_arr:
                    try:
                        bool_array.append(self.fuzzyMatch(bools, b, ratio=85))
                    except Exception as e:
                        print(e)

            if np.count_nonzero(bool_array) >= (len(values) * 0.85):

                return  build_return_standard_object(category='Boolean', subcategory=None, match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def bool_letter_f(values):
            print('Start boolean validation ...')
            bool_arr = ['t', 'f', 'T', 'F']
            bool_array = []
            for bools in values:
                for b in bool_arr:
                    try:
                        bool_array.append(self.fuzzyMatch(bools, b, ratio=98))
                    except Exception as e:
                        print(e)

            if np.count_nonzero(bool_array) >= (len(values) * .85):

                return build_return_standard_object(category='Boolean', subcategory=None, match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def build_return_object(format, util, dayFirst):
            return {'category': 'time', 'subcategory': 'date', 'format': format,
                    "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}

        def build_return_standard_object(category, subcategory, match_type):
            return {'category': category, 'subcategory': subcategory, 'format': None,
                    "match_type": [match_type], "Parser": None, "DayFirst": None}

        def build_return_standard_object_skipped(category, subcategory, match_type):
            return {'category': category, 'subcategory': subcategory, 'format': None,
                    "match_type": [match_type], "Parser": None, "DayFirst": None}

        def dayFirstCheck(values, separator, shortYear, yearLoc):
            # only works for 4 number year
            for date in values:
                try:
                    arr = date.split(separator)
                    if shortYear:
                        if yearLoc == 0:
                            if int(arr[1]) > 12:
                                return True
                        else:
                            if int(arr[0]) > 12:
                                return True
                    else:
                        if len(arr[0]) == 4:
                            if int(arr[1]) > 12:
                                return True
                        else:
                            if int(arr[0]) > 12:
                                return True
                except Exception as e:
                    print("error occurred", e)

            return False

        def date_arrow(values, separator):
            utils_array = []
            for date in values:
                try:
                    dateArrow = arrow.get(str(date), normalize_whitespace=True).datetime

                    if isinstance(dateArrow, datetime.date):
                        utils_array.append("true")
                    else:
                        print("Not a valid date format")
                except Exception as e:
                    print(e, "Error from Arrow: Date had an error")
            return utils_array

        def date_arrow_1(values):
            array_valid = date_arrow(values, separator="none")
            if len(array_valid) > len(values) * 0.85:
                return build_return_object(format="%Y%d", util='arrow', dayFirst=None)
            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_arrow_2(values):
            # array_valid = date_arrow(values, separator="-")
            monthFormat = month_MMorM(values, separator='-', loc=1)
            allMonthVals = []
            for val in values:
                monthval = val.split('-')[1]
                allMonthVals.append(monthval)
            validMonth = month_day_f(allMonthVals)
            if validMonth["subcategory"] == "date" and validMonth["format"] == "%m"  or validMonth["subcategory"] ==  'date' and validMonth['format']=="%-m" :
                return build_return_object(format="%Y-" + monthFormat, util='arrow', dayFirst=None)
            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_arrow_3(values):
            array_valid = date_arrow(values, separator="/")
            monthFormat = month_MMorM(values, separator='/', loc=1)
            allMonthVals = []
            for val in values:
                monthval = val.split('/')[1]
                allMonthVals.append(monthval)
            validMonth = month_day_f(allMonthVals)
            if len(array_valid) > len(values) * 0.85:
                return build_return_object(format="%Y/" + monthFormat, util='arrow', dayFirst=None)
            elif validMonth["subcategory"] == 'date' and validMonth['format']=="%m"  or validMonth['subcategory'] == 'date' and validMonth['format']=='%-m':
                return build_return_object(format="%Y/" + monthFormat, util='arrow', dayFirst=None)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_arrow_4(values):
            array_valid = date_arrow(values, separator=".")
            monthFormat = month_MMorM(values, separator='.', loc=1)
            allMonthVals = []
            for val in values:
                monthval = val.split('.')[1]
                allMonthVals.append(monthval)

            validMonth = month_day_f(allMonthVals)

            if len(array_valid) > len(values) * 0.75:
                return build_return_object(format="%Y." + monthFormat, util='arrow', dayFirst=None)

            elif validMonth['subcategory'] == 'date' and validMonth['format']=='%m' or validMonth['subcategory'] == 'date' and validMonth['format']== '%-m':
                return build_return_object(format="%Y." + monthFormat, util='arrow', dayFirst=None)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util(values, separator, shortyear, yearloc):

            util_dates = []
            if separator != "none":
                dayFirst = dayFirstCheck(values, separator, shortYear=shortyear, yearLoc=yearloc)
            else:
                dayFirst = False

            for date in values:
                try:

                    dateUtil = dateutil.parser.parse(str(date), dayfirst=dayFirst)
                    if isinstance(dateUtil, datetime.date):
                        util_dates.append({"value": date, "standard": dateUtil})
                    else:
                        pass
                except Exception as e:
                    print(e)

            return util_dates, dayFirst

        def day_ddOrd(values, separator, loc):
            dayFormat = '%-d'
            for d in values:
                try:
                    if separator is None:
                        if d[0] == '0':
                            dayFormat = '%d'
                    else:
                        d_value = d.split(separator)[loc]
                        if d_value[0] == '0':
                            dayFormat = '%d'
                except Exception as e:
                    print(e)
            return dayFormat

        def month_MMorM(values, separator, loc):
            monthFormat = '%-m'
            for d in values:
                try:
                    if separator is None:
                        if d[0] == '0':
                            monthFormat = '%m'
                    else:
                        d_value = d.split(separator)[loc]
                        if d_value[0] == '0':
                            monthFormat = '%m'
                except Exception as e:
                    print(e)
            return monthFormat

        def hour_hOrH(values,separator, loc_hms):
            hourFormat = '%-H'
            for d in values:

                if separator is None:
                    if d[0] == '0':
                        hourFormat = '%H'
                else:
                    hms = d.split(' ')[-1]
                    hms = hms.split(separator)[loc_hms]

                    if hms[0] == '0':
                        hourFormat = '%H'
            return hourFormat

        def minute_mOrM(values,separator, loc_hms):
            minuteFormat = '%-M'
            for d in values:
                if separator is None:
                    if d[0] == '0':
                        minuteFormat = '%M'
                else:
                    hms = d.split(' ')[-1]
                    hms = hms.split(separator)[loc_hms]
                    if hms[0] == '0':
                        minuteFormat = '%M'
            return minuteFormat

        def second_sOrS(values, separator, loc_hms):
            secondFormat = '%-S'
            for d in values:
                if separator is None:
                    if d[0] == '0':
                        secondFormat = '%S'
                else:
                    hms = d.split(' ')[-1]
                    hms = hms.split(separator)[loc_hms]
                    if hms[0] == '0':
                        secondFormat = '%S'
            return secondFormat

        # currently can't handle yyyy-dd-mm HH:mm:ss
        def iso_time(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)

            if len(array_valid) > len(values) * 0.85:
                # '1996-03-20T07:46:39'
                # '1998-08-15T08:43:22'
                # '1972-10-03T05:52:26'
                # '1987-08-15T09:51:25'
                return build_return_object(format="%Y-%m-%dT%H%M%S", util='Util', dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_1(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)

            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "-" + monthFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "-" + dayFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_2(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)

            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "-" + monthFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "-" + dayFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_3(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)

            if dayFirst:
                return build_return_object(format=dayFormat + "_" + monthFormat + "_%Y", util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format=monthFormat + "_" + dayFormat + "_%Y", util='Util', dayFirst=dayFirst)

        def date_util_4(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)
            if dayFirst:
                return build_return_object(format=dayFormat + "_" + monthFormat + "_%y", util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format=monthFormat + "_" + dayFormat + "_%y", util='Util', dayFirst=dayFirst)

        def date_util_5(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)

            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "/" + monthFormat + "/%Y", util='Util',
                                               dayFirst=dayFirst)
                else:
                    return build_return_object(format=monthFormat + "/" + dayFormat + "/%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_6(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "/" + monthFormat + "/%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "/" + dayFormat + "/%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_7(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    return build_return_object(format=dayFormat + "." + monthFormat + ".%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "." + dayFormat + ".%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_8(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "." + monthFormat + ".%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "." + dayFormat + ".%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_9(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    return build_return_object(format=dayFormat + "-" + monthFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "-" + dayFormat + "-%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_10(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFormat:
                    return build_return_object(format=dayFormat + "-" + monthFormat + "-%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + "-" + dayFormat + "-%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_11(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)

            if dayFirst:

                return build_return_object(format=dayFormat + "_" + monthFormat + "_%Y", util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format=monthFormat + "_" + dayFormat + "_%Y", util='Util', dayFirst=dayFirst)

        def date_util_12(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)

            if dayFirst:

                return build_return_object(format=dayFormat + "_" + monthFormat + "_%y", util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format=monthFormat + "_" + dayFormat + "_%y", util='Util', dayFirst=dayFirst)

        def date_util_13(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    return build_return_object(format=dayFormat + '/' + monthFormat + '/' + "/%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + '/' + dayFormat + '/' + "/%Y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_14(values):

            array_valid, dayFirst = date_util(values, separator="/", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    return build_return_object(format=dayFormat + '/' + monthFormat + "/%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + '/' + dayFormat + "/%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_15(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    return build_return_object(format=dayFormat + '.' + monthFormat + ".%Y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format=monthFormat + '.' + dayFormat + ".%Y", util='Util',
                                               dayFirst=dayFirst)


            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_16(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=True, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:

                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': dayFormat + '.' + monthFormat + ".%y",
                    #         "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format=dayFormat + '.' + monthFormat + ".%y", util='Util',
                                               dayFirst=dayFirst)

                else:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': monthFormat + '.' + dayFormat + ".%y",
                    # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format=monthFormat + '.' + dayFormat + ".%y", util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_17(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=2)
                monthFormat = month_MMorM(values, separator='_', loc=1)

            if dayFirst:
                #    return {'category': 'time', 'subcategory': 'date', 'format': "%Y_"+ dayFormat+"_"+monthFormat,
                # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                return build_return_object(format="%Y_" + dayFormat + "_" + monthFormat, util='Util', dayFirst=dayFirst)

            else:
                return build_return_object(format="%Y_" + monthFormat + "_" + dayFormat, util='Util', dayFirst=dayFirst)

            #    return {'category': 'time', 'subcategory': 'date', 'format': "%Y_" + monthFormat + "_" + dayFormat,
            # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}

        def date_util_18(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=2)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    #    return  {'category': 'time', 'subcategory': 'date', 'format': "%Y."+dayFormat+"."+monthFormat,
                    # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format="%Y." + dayFormat + "." + monthFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format="%Y." + monthFormat + "." + dayFormat, util='Util',
                                               dayFirst=dayFirst)

                #    return  {'category': 'time', 'subcategory': 'date', 'format': "%Y."+monthFormat+"."+dayFormat,
                # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_19(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=2)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format': "%Y-" + dayFormat+"-"+monthFormat,
                    # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format="%Y-" + dayFormat + "-" + monthFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format="%Y-" + monthFormat + "-" + dayFormat, util='Util',
                                               dayFirst=dayFirst)

                #    return  {'category': 'time', 'subcategory': 'date',
                #    'format': "%Y-" + monthFormat + "-" + dayFormat,
                # "match_type": ['LSTM'], "Parser": "Util", "DayFirst": dayFirst}

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_20(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=2)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return  {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y/"+dayFormat+"/"+monthFormat, "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format="%Y/" + dayFormat + "/" + monthFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    return build_return_object(format="%Y/" + monthFormat + "/" + dayFormat, util='Util',
                                               dayFirst=dayFirst)

                    # return  {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y/" + monthFormat + "/" + dayFormat, "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_21(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=0)

            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=2)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            # hour min sec format
            hourFormat= hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':',loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format="%Y-" + dayFormat + "-" + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(format="%Y-" + monthFormat + "-" + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_23(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=0)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=2)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format="%Y/" + dayFormat + "/" + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(format="%Y/" + monthFormat + "/" + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_24(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=0)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=2)
                monthFormat = month_MMorM(values, separator='_', loc=1)

            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if dayFirst:
                # return {'category': 'time', 'subcategory': 'date',
                # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                # "Parser": "Util", "DayFirst": dayFirst}
                return build_return_object(format="%Y_" + dayFormat + "_" + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                           dayFirst=dayFirst)

            else:
                #    return {'category': 'time', 'subcategory': 'date',
                #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                # "Parser": "Util",  "DayFirst": dayFirst}
                return build_return_object(format="%Y_" + monthFormat + "_" + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                           dayFirst=dayFirst)


        def date_util_25(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=0)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=2)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=2)
                monthFormat = month_MMorM(values, separator='.', loc=1)

            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format="%Y." + dayFormat + "." + monthFormat +  ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(format="%Y." + monthFormat + "." + dayFormat +  ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_26(values):
            array_valid, dayFirst = date_util(values, separator="-", shortyear=False, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='-', loc=0)
                monthFormat = month_MMorM(values, separator='-', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='-', loc=1)
                monthFormat = month_MMorM(values, separator='-', loc=0)

            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format=dayFormat + "-" + monthFormat + '-%Y'+ ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(format= monthFormat + "-" + dayFormat + '-%Y'+ ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_util_27(values):
            array_valid, dayFirst = date_util(values, separator="/", shortyear=False, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)
            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format=dayFormat + "/" + monthFormat + '/%Y'+' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(format= monthFormat + "/" + dayFormat + '/%Y'+' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)
        def date_util_28(values):
            array_valid, dayFirst = date_util(values, separator="_", shortyear=False, yearloc=2)
            print('dayf', dayFirst)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='_', loc=0)
                monthFormat = month_MMorM(values, separator='_', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='_', loc=1)
                monthFormat = month_MMorM(values, separator='_', loc=0)
            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if dayFirst:
                # return {'category': 'time', 'subcategory': 'date',
                # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                # "Parser": "Util", "DayFirst": dayFirst}
                return build_return_object(format=dayFormat + "_" + monthFormat + '_%Y' +' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                           dayFirst=dayFirst)

            else:
                #    return {'category': 'time', 'subcategory': 'date',
                #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                # "Parser": "Util",  "DayFirst": dayFirst}
                return build_return_object(format= monthFormat + "_" + dayFormat + '_%Y'+' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                           dayFirst=dayFirst)


        def date_util_29(values):
            array_valid, dayFirst = date_util(values, separator=".", shortyear=False, yearloc=2)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='.', loc=0)
                monthFormat = month_MMorM(values, separator='.', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='.', loc=1)
                monthFormat = month_MMorM(values, separator='.', loc=0)
            hourFormat = hour_hOrH(values, separator=':', loc_hms=0)
            minFormat = minute_mOrM(values, separator=':', loc_hms=1)
            secFormat = second_sOrS(values, separator=':', loc_hms=2)
            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    # 'format': "%Y-" + dayFormat + "-" + monthFormat +' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util", "DayFirst": dayFirst}
                    return build_return_object(format=dayFormat + "." + monthFormat + '.%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

                else:
                    #    return {'category': 'time', 'subcategory': 'date',
                    #    'format':"%Y-" + monthFormat + "-" + dayFormat+ ' %H%M%S', "match_type": ['LSTM'],
                    # "Parser": "Util",  "DayFirst": dayFirst}
                    return build_return_object(format= monthFormat + "." + dayFormat + '.%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat, util='Util',
                                               dayFirst=dayFirst)

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)


        def date_util_22(values):
            array_valid = []
            for v in values:
                try:
                    if int(v) < -5364601438 or int(v) > 4102506000:
                        array_valid.append('failed')
                    elif len(v) <= 13:
                        array_valid.append('valid')
                    else:
                        array_valid.append('failed')
                except Exception as e:
                    array_valid.append('failed')
                    print(e)

            if 'failed' in array_valid:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)
            else:
                return build_return_object(format='Unix Timestamp', util=None, dayFirst=None)

                # return {'category': 'time', 'subcategory': 'date', 'format': 'Unix Timestamp',
                # "match_type": ['LSTM'],
                #         "Parser": "Util"}

        def date_long_1(values):
            #              #  01 April 2008
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=0)
            if len(array_valid) > len(values) * 0.85:
                return build_return_object(format=dayFormat + " %B %Y", util=None, dayFirst=None)

                # return {'category': 'time', 'subcategory': 'date', 'format':dayFormat+" %B %Y",
                # "match_type": ['LSTM'],  "Parser": "Util"}

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_2(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=0)
            if len(array_valid) > len(values) * 0.85:
                #                 02 April 20
                #                    dd/LLLL/yy
                return build_return_object(format=dayFormat + " %B %y", util=None, dayFirst=None)

                # return  {'category': 'time', 'subcategory': 'date', 'format': dayFormat+" %B %y",
                # "match_type": ['LSTM'], "Parser": "Util"}

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_3(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=2)
            if len(array_valid) > len(values) * 0.85:
                return build_return_object(format="%A, %B " + dayFormat + ",%y", util=None, dayFirst=None)
                # return  {'category': 'time', 'subcategory': 'date', 'format': "%A, %B "+dayFormat+",%y",
                # "match_type": ['LSTM'], "Parser": "Util"}

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_4(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=1)
            if len(array_valid) > len(values) * 0.85:
                #                 April 10, 2008
                #                 LLLL dd, y
                return build_return_object(format="%B " + dayFormat + ", %Y", util=None, dayFirst=None)

                # return  {'category': 'time', 'subcategory': 'date', 'format':"%B "+dayFormat+", %Y" ,
                # "match_type": ['LSTM'], "Parser": "Util"}

            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_5(values):
            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            dayFormat = day_ddOrd(values, separator=' ', loc=2)
            if len(array_valid) > len(values) * 0.85:
                #  Thursday, April 10, 2008 6:30:00 AM
                #                 EEEE, LLLL dd,yy HH:mm:ss

                return build_return_object(format="%A, %B " + dayFormat + ",%y HH:mm:ss", util=None, dayFirst=None)
                # return {'category': 'time', 'subcategory': 'date','format': "%A, %B "+dayFormat+",%y HH:mm:ss",
                # "match_type": ['LSTM'],"Parser": "Util"}
            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def date_long_6(values):

            array_valid, dayFirst = date_util(values, separator="none", shortyear=False, yearloc=None)
            if dayFirst:
                dayFormat = day_ddOrd(values, separator='/', loc=0)
                monthFormat = month_MMorM(values, separator='/', loc=1)
            else:
                dayFormat = day_ddOrd(values, separator='/', loc=1)
                monthFormat = month_MMorM(values, separator='/', loc=0)

            if len(array_valid) > len(values) * 0.85:
                if dayFirst:
                    # return {'category': 'time', 'subcategory': 'date',
                    #         'format': dayFormat + "/" + monthFormat + "/%y HH:mm", "match_type": ['LSTM']}
                    return build_return_object(format=dayFormat + "/" + monthFormat + "/%y HH:mm", util=None,
                                               dayFirst=None)

                #              03/23/21 01:55 PM
                #                 MM/dd/yy HH:mm
                else:
                    return build_return_object(format=monthFormat + "/" + dayFormat + "/%y HH:mm", util=None,
                                               dayFirst=None)
                    # return {'category': 'time', 'subcategory': 'date', 'format':monthFormat+"/"+dayFormat+"/%y HH:mm", "match_type": ['LSTM'] }
            else:
                return  build_return_standard_object(category='unknown date', subcategory=None, match_type=None)

        def month_day_f(values):
            month_day_results = []
            for i, md in enumerate(values):
                try:
                    if str.isdigit(md):
                        if 12 >= int(md) >= 1:
                            month_day_results.append("month_day")
                        elif 12 < int(md) <= 31:
                            month_day_results.append("day")
                        else:
                            month_day_results.append("failed")
                    else:
                        print("Not a valid digit")
                except Exception as e:
                    print(e)

            if "failed" in month_day_results:
                return  build_return_standard_object(category=None, subcategory=None, match_type=None)
            elif "day" in month_day_results:
                # return {'category': 'time', 'subcategory': 'date', 'format':day_ddOrd(values, separator=None, loc=None), "match_type": ['LSTM'] }
                return build_return_object(format=day_ddOrd(values, separator=None, loc=None), util=None, dayFirst=None)
            elif "month_day" in month_day_results:
                # return {'category': 'time', 'subcategory': 'date', 'format':month_MMorM(values, separator=None, loc=None), "match_type": ['LSTM'] }
                return build_return_object(format=month_MMorM(values, separator=None, loc=None), util=None,
                                           dayFirst=None)
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        def month_name_f(values):
            print("Start month validation ...")
            month_array_valid = []
            for month in values:
                for m in self.month_of_year:
                    try:
                        month_array_valid.append(
                            self.fuzzyMatch(str(month), str(m), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(month_array_valid) >= (len(values) * 0.65):
                # return {'category': 'time', 'subcategory': 'date', 'format':'%B', "match_type": ['LSTM'] }
                return build_return_object('%B', util=None, dayFirst=None)
            else:
                return day_name_f(values)

        def day_name_f(values):
            print("Start day validation ...")
            day_array_valid = []
            for day in values:
                for d in self.day_of_week:
                    try:
                        day_array_valid.append(
                            self.fuzzyMatch(str(day), str(d), ratio=85)
                        )
                    except Exception as e:
                        print(e)

            if np.count_nonzero(day_array_valid) >= (len(values) * 0.65):

                # return {'category': 'time', 'subcategory': 'date', 'format':'%A', "match_type": ['LSTM'] }
                return build_return_object('%A', util=None, dayFirst=None)
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)

        functionlist = defaultdict(
            int,
            {
                "None": none_f,
                "country_name": country_f,
                "city": country_f,
                "language_name": country_f,
                "city_suffix": country_f,
                "first_name": none_f,
                "country_GID": country_iso3,
                "country_code": country_iso2,
                "continent": continent_f,
                "geo": geo_f,
                "pyfloat": none_f,
                "percent": none_f,
                "ssn": none_f,
                "phone_number": none_f,
                "zipcode": none_f,
                "paragraph": none_f,
                "email": none_f,
                "prefix": none_f,
                "pystr": none_f,
                "isbn": none_f,
                "boolean": bool_f,
                'boolean_letter': bool_letter_f,
                "iso8601": iso_time,
                "year": year_f,
                "day_of_month": month_day_f,
                "month": month_day_f,
                "month_name": month_name_f,
                "day_of_week": month_name_f,
                "date_%Y%d": date_arrow_1,
                "date_%Y-%m": date_arrow_2,
                "date_%Y/%m": date_arrow_3,
                "date_%Y.%m": date_arrow_4,
                "date_%m-%d-%Y": date_util_1,
                "date_%m-%d-%y": date_util_2,
                "date_%m_%d_%Y": date_util_3,
                "date_%m_%d_%y": date_util_4,
                "date_%m/%d/%Y": date_util_5,
                "date_%m/%d/%y": date_util_6,
                "date_%m.%d.%Y": date_util_7,
                "date_%m.%d.%y": date_util_8,
                "date_%d-%m-%Y": date_util_9,
                "date_%d-%m-%y": date_util_10,
                "date_%d_%m_%Y": date_util_11,
                "date_%d_%m_%y": date_util_12,
                "date_%d/%m/%Y": date_util_13,
                "date_%d/%m/%y": date_util_14,
                "date_%d.%m.%Y": date_util_15,
                "date_%d.%m.%y": date_util_16,
                "date_%Y_%m_%d": date_util_17,
                "date_%Y.%m.%d": date_util_18,
                "date_%Y-%m-%d": date_util_19,
                "date_%Y/%m/%d": date_util_20,
                "date_%Y-%m-%d %H:%M:%S": date_util_21,
                'date_%Y/%m/%d %H:%M:%S': date_util_23,
                'date_%Y_%m_%d %H:%M:%S': date_util_24,
                'date_%Y.%m.%d %H:%M:%S': date_util_25,
                'date_%m-%d-%Y %H:%M:%S': date_util_26,
                'date_%m/%d/%Y %H:%M:%S': date_util_27,
                'date_%m_%d_%Y %H:%M:%S': date_util_28,
                'date_%m.%d.%Y %H:%M:%S': date_util_29,
                'date_%d-%m-%Y %H:%M:%S': date_util_26,
                'date_%d/%m/%Y %H:%M:%S': date_util_27,
                'date_%d_%m_%Y %H:%M:%S': date_util_28,
                'date_%d.%m.%Y %H:%M:%S': date_util_29,
                'unix_time': date_util_22,
                "date_long_dmonthY": date_long_1,
                "date_long_dmonthy": date_long_2,
                "date_long_dmdy": date_long_3,
                "date_long_mdy": date_long_4,
                "date_long_dmdyt": date_long_5,
                "date_long_mdyt_m": date_long_6,
            },
        )
        final_column_classification = []

        def add_obj(obj, add_obj):

            for property in add_obj:
                obj[property] = add_obj[property]
            return obj

        for pred in predictions:
            print(pred['column'])
            print(pred['values'])

            final_column_classification.append(
                    add_obj({"column": pred["column"]}, functionlist[pred["avg_predictions"]["averaged_top_category"]](
                            self.column_value_object[pred["column"]]
                        )))

        return final_column_classification

    def fuzzymatchColumns(self, classifications):
        predictions = classifications
        words_to_check = [
            {"Date":"Date"},
            {"Datetime":"Datetime"},
            {"Timestamp":"Timestamp"},
            {"Epoch":"Epoch"},
            {"Time":"Time"},
            {"Year":"Year"},
            {"Month":"Month"},
            {"Lat":"Latitude"},
            {"Latitude":"Latitude"},
            {"lng":"Latitude"},
            {"lon":"Longitude"},
            {"long":"Longitude"},
            {"Longitude":"Longitude"},
            {"Geo":"Geo"},
            {"Coordinates":"Coordinates"},
            {"Location":"Location"},
            {"West":"West"},
            {"South":"South"},
            {"East":"East"},
            {"North":"North"},
            {"Country":"Country"},
            {"CountryName":"CountryName"},
            {"CC":"CC"},
            {"CountryCode":"CountryCode"},
            {"State":"State"},
            {"City":"City"},
            {"Town":"Town"},
            {"Region": "Region"},
            {"Province": "Province"},
            {"Territory": "Territory"},
            {"Address":"Address"},
            {"ISO2":"ISO2"},
            {"ISO3": "ISO3"},
            {"ISO_code":"ISO_code"},
            {"Results": "Results"},
        ]

        for i, pred in enumerate(predictions):
            for y, keyValue in enumerate(words_to_check):
                try:
                  for key in keyValue:
                      if self.fuzzyMatch(str(pred["column"]), str(key), 85):
                            T, ratio = self.fuzzyRatio(str(pred["column"]), str(key),85)
                            predictions[i]['match_type'].append('fuzzy')
                            predictions[i]["fuzzyColumn"]=[]
                            predictions[i]["fuzzyColumn"].append({"matchedKey":str(key), "fuzzyCategory":words_to_check[y][key], "ratio":ratio})
                      else:
                            pass
                except Exception as e:
                    print(e)

        for pred2 in predictions:
            try:
                if len(pred2["fuzzyColumn"])>1:
                    ind = 0
                    for i, fmatch in pred2['fuzzyColumn']:

                        bestRatio=0
                        if fmatch["ratio"]>bestRatio:
                            bestRatio=fmatch["ratio"]
                            ind=i
                    pred2['fuzzyColumn']=pred2['fuzzyColumn'][ind]
                else:
                    pred2['fuzzyColumn']=pred2['fuzzyColumn'][0]
            except Exception as e:
                print(e)

        return predictions


    def fuzzymatchColumns_enhanced(self,df):

        words_to_check = [
            {"Lat":"latitude"},
            {"Latitude":"latitude"},
            {"lng":"latitude"},
            {"lon":"longitude"},
            {"long":"longitude"},
            {"Longitude":"longitude"},
            {"ISO2":"ISO2"},
            {"ISO3": "ISO3"}
        ]
        array_of_columnMatch_index=[]
        index_to_not_process=[]
        for i, header in enumerate(df.columns):
            # print(i,header)
            for y, keyValue in enumerate(words_to_check):
                for key in keyValue:
                    if self.fuzzyMatch(header, str(key), 90):
                        T, ratio = self.fuzzyRatio(header, str(key), 90)
                        index_to_not_process.append(i)
                        array_of_columnMatch_index.append({'index':i, 'header':header, 'key':key, 'value':keyValue[key], 'ratio': ratio})
                    else:
                        pass

        return index_to_not_process , array_of_columnMatch_index


    def standard_dateColumns(self, fuzzyOutput, formats='default'):
        df = self.df
        for i, out in enumerate(fuzzyOutput):
            try:
                #             print(out['classification'][0]['Category'])
                if out['subcategory'] == 'date' or out['fuzzyColumn']["fuzzyCategory"] == 'Date' or out['fuzzyColumn']["fuzzyCategory"] == 'Timestamp' or out['fuzzyColumn']["fuzzyCategory"] == 'Datetime':
                    new_column = 'ISO_8601_' + str(i)
                    if 'DayFirst' in out:
                        if out['DayFirst'] is not None:
                            dayFirst = out['DayFirst']
                    else:
                        dayFirst = False
                    if formats != 'default':

                        df = df.assign(
                            **{new_column: lambda dataframe: dataframe[out['column']].map(
                                lambda date: datetime.datetime.strftime(
                                    dateutil.parser.parse(str(date), dayfirst=dayFirst), formats))}
                        )
                    else:
                        df = df.assign(
                            **{new_column: lambda dataframe: dataframe[out['column']].map(
                                lambda date: dateutil.parser.parse(str(date), dayfirst=dayFirst))}
                        )
            except Exception as e:
                print(e)
        return df

    def standard_dateColumn(self, fuzzyOutput, columnName):
        df = self.df
        for out in fuzzyOutput:
            if out["column"] == columnName:
                dayFirst = out["classification"][0]["DayFirst"]

                df = df.assign(
                    ISO_8601=lambda dataframe: dataframe[columnName].map(
                        lambda date: dateutil.parser.parse(str(date), dayfirst=dayFirst)
                    )
                )
        return df

    def final_step(self, t):
        classifiedObjs=[]
        for tstep in t:
            tstep['match_type']=list(set(tstep['match_type']))
            tstep['match_type']= [i for i in tstep['match_type'] if i]
            fuzzyCol = None
            try:
                fuzzyCol = tstep['fuzzyColumn']
            except Exception as e:
                print(e)
            classifiedObj = Classification(column=tstep['column'], category=tstep["category"], subcategory=tstep['subcategory'],
                                           format=tstep['format'],
                                           match_type=tstep['match_type'], Parser=tstep['Parser'], DayFirst=tstep['DayFirst'],
                                           fuzzyColumn=fuzzyCol)
            classifiedObjs.append(classifiedObj)
        return Classifications(classifications = classifiedObjs)

    def findNANsColumns(self, df):
        array_columns_nans=[]
        index_nan=[]
        for index,column in enumerate(df.columns):
            if df[column].count() > 0:
                print('column test', column)

            else:
                print('failed col', column)
                index_nan.append(index)

        print(index_nan)
        return index_nan

    def columns_classified(self, path):
        preds = self.predictions(path_to_csv=path)
        for pred in preds:
            try:
                print('here')
                print(pred['avg_predictions']["averaged_top_category"])
            except Exception as e:
                print(e)
        output = self.assign_heuristic_function(preds)
        output_fuz = self.fuzzymatchColumns(output)
        return output_fuz

    def columns_classifed_enhanced(self, path):
        df = self.read_in_csv(path)
        index_remove, fuzzyMatchColumns = self.fuzzymatchColumns_enhanced(df)
        columns_na = self.findNANsColumns(df)
        index_remove=index_remove+columns_na
        preds = self.predictions_enhanced(df, index_remove)
        output = self.assign_heuristic_function_enhanced(preds, fuzzyMatchColumns)
        print('out', output)
        t=self.fuzzymatchColumns(output)
        print(fuzzyMatchColumns)
        print('t',t)
        final = self.final_step(t)

        return final


    def add_iso8601_columns(self, path, formats):
        preds = self.predictions(path_to_csv=path)
        output = self.assign_heuristic_function(preds)
        output_fuz = self.fuzzymatchColumns(output)
        output_col = self.standard_dateColumns(output_fuz, formats)

        return output_col

    def get_Fake_Data(self):
        return self.FakeData

import time
start_time=time.time()
gc=GeoTimeClassify(20)
#pred=gc.columns_classified("/home/kyle/Desktop/blank.csv")

preds=gc.columns_classifed_enhanced("/home/kyle/Desktop/test3.csv")
print("final", preds.dict())
print(preds)
print("%s seconds: " % (time.time()-start_time))

