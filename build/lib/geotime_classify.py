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

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
        self.cwd = os.getcwd()
        self.number_of_random_samples = number_of_samples
        #       prediction tensors with the best match being less than predictionLimit will not be returned
        self.predictionLimit = -4.5
        self.country_lookup = pd.read_csv(self.cwd + "/src/datasets/lookups/country.csv")
        self.city_lookupcsv = pd.read_csv(self.cwd + "/src/datasets/lookups/city.csv")
        self.city_lookup = np.asarray(self.city_lookupcsv["city"])
        self.state_lookupcsv = pd.read_csv(self.cwd + "/src/datasets/lookups/NA_states_provinces.csv")
        self.state_lookup = np.asarray(self.state_lookupcsv["state_name"])
        self.country_lookupcsv = pd.read_csv(self.cwd + "/src/datasets/lookups/country.csv")
        self.country_lookup=np.asarray(self.country_lookupcsv["country_name"])
        self.iso3_lookup = np.asarray(self.country_lookupcsv["Alpha-3_Code"])
        self.iso2_lookup =np.asarray(self.country_lookupcsv["Alpha-2_Code"])
        self.cont_lookupcsv = pd.read_csv(self.cwd + "/src/datasets/lookups/continent_code.csv")
        self.cont_lookup = np.asarray(self.cont_lookupcsv["continent_name"])

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
                "boolean_letter":57
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
        # print("test", test)
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

    def predictions(self, path_to_csv):
        print('LSTM')

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
        self.model = LSTMClassifier(vocab_size=89, embedding_dim=128, hidden_dim=32, output_size=58)
        self.model.load_state_dict(torch.load(self.cwd + '/src/models/LSTM_RNN_Geotime_Classify_v_0.07_dict.pth'))
        self.model.eval()
        # self.model = torch.load(self.cwd + '/src/models/LSTM_RNN_Geotime_Classify_v_0.07.pth')
        column_value_object = self.get_arrayOfValues_df(df)
        self.column_value_object = column_value_object
        predictionList= []
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

    # match two words return true based on ratio
    def fuzzyMatch(self, word1, word2, ratio=95):
        Ratio = fuzz.ratio(word1.lower(), word2.lower())

        if Ratio > ratio:
            return True

    # map function to model prediction category
    def assign_heuristic_function(self, predictions):
        c_lookup=self.city_lookup
        state_lookup=self.state_lookup
        country_lookup=self.country_lookup
        iso3_lookup=self.iso3_lookup
        iso2_lookup=self.iso2_lookup
        cont_lookup=self.cont_lookup
        def none_f(values):
            return {"Category": "None"}

        def city_f( values):
            print("start city lookup")
            country_match_bool = []
            for city in values:
                try:
                    match = fuzzywuzzy.process.extractOne(
                        city, c_lookup, scorer=fuzz.token_sort_ratio
                    )
                    if match is not None:
                        if match[1] > 85:
                            country_match_bool.append(True)
                except Exception as e:
                    print(e)

            if np.count_nonzero(country_match_bool) >= (len(values) * 0.30):
                return {"Category": "City Name"}
            else:
                return {"Category": "Proper Noun"}

        def state_f(values):
            print("start state lookup")
            country_match_bool = []

            for state in values:
                for c in state_lookup:
                    try:
                        country_match_bool.append(self.fuzzyMatch(state, c, ratio=85))
                    except Exception as e:
                        print(e)

            if np.count_nonzero(country_match_bool) >= (len(values) * 0.40):
                return {"Category": "State Name"}
            else:
                print("Starting fuzzy match on cities...")
                return city_f(values)

        def country_f(values):

            print("start country lookup")
            country_match_bool = []


            for country in values:
                for c in country_lookup:
                    try:
                        country_match_bool.append(self.fuzzyMatch(country, c, ratio=85))
                    except Exception as e:
                        print(e)

            if np.count_nonzero(country_match_bool) >= (len(values) * 0.40):
                return {"Category": "Country Name"}
            else:
                return state_f(values)

        def country_iso3(values):
            print("start iso3 lookup")
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
                return {"Category": "ISO3"}
            else:
                return country_iso2(values)

        def country_iso2(values):
            print("start iso2 lookup")
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

                return {"Category": "ISO2"}
            else:
                return {"Category": "Unknown code"}

        def continent_f(values):
            print("start continent lookup")
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

                return {"Category": "Continent"}
            else:
                return {"Category": "Proper Noun"}

        def geo_f(values):
            print("start geo test")
            geo_valid = []
            percent_array = []
            for geo in values:
                try:
                    if float(geo) <= 180 and float(geo) >= -180:
                        if float(geo) <= 90 and float(geo) >= -90:
                            geo_valid.append("latlng")
                            if float(geo) <= 1 and float(geo) >= -1:
                                percent_array.append("true")

                        else:
                            print("lng", geo)
                            geo_valid.append("lng")
                    else:
                        geo_valid.append("failed")
                except Exception as e:
                    print(e)

            if "failed" in geo_valid:
                return {"Category": "Number"}
            elif len(percent_array) >= len(values) * 0.95:
                return {
                    "Category": "Number/Geo",
                    "type": "Unknown-mostly between -1 and 1",
                }
            elif "lng" in geo_valid:
                return {"Category": "Geo", "type": "Longitude (number)"}
            elif "latlng" in geo_valid:
                return {"Category": "Geo", "type": "Latitude (number)"}
            else:
                return {"Category": "Number"}

        def year_f(values):
            print("start year test")
            year_values_valid = []
            years_failed = []
            strange_year = []
            for year in values:
                try:
                    if str.isdigit(str(year)):
                        if int(year) > 1300 and int(year) < 2500:
                            year_values_valid.append("True")
                        else:
                            strange_year.append("Maybe")
                    else:
                        years_failed.append("Failed")
                except Exception as e:
                    print(e)

            if len(years_failed) > len(values) * 0.15:
                return {"Category": "None"}
            elif len(strange_year) > len(values) * 15:
                return {"Category": "None"}
            elif len(year_values_valid) > len(values) * 0.75:
                return {"Category": "Year"}

        def bool_f(values):
            print("start boolian test")
            bool_arr = ["true", "false", "T", "F"]
            bool_array = []
            for bools in values:
                for b in bool_arr:
                    try:
                        bool_array.append(self.fuzzyMatch(bools, b, ratio=85))
                    except Exception as e:
                        print(e)

            if np.count_nonzero(bool_array) >= (len(values) * 0.85):

                return {"Category": "Boolian"}
            else:
                return {"Category": "None"}

        def bool_letter_f(values):
            print('start boolian letter test')
            bool_arr = ['t', 'f', 'T', 'F']
            bool_array = []
            for bools in values:
                for b in bool_arr:
                    try:
                        bool_array.append(self.fuzzyMatch(bools, b, ratio=98))
                    except Exception as e:
                        print(e)

            if np.count_nonzero(bool_array) >= (len(values) * .85):

                return {'Category': 'Boolian'}
            else:
                return {'Category': 'None'}

        def dayFirstCheck(Values, seperator):

            for date in Values:
                try:
                    arr = date.split(seperator)
                    if len(arr[0]) == 4:
                        if int(arr[1]) > 12:
                            return True
                    else:
                        if int(arr[0]) > 12:
                            return True
                except:
                    print("error occured")

            return False

        def date_arrow(values, seperator):
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
            array_valid = date_arrow(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                return {"Category": "Date", "Format": "ydd", "Parser": "arrow"}
            else:
                return {"Category": "Unknown Date"}

        def date_arrow_2(values):
            array_valid = date_arrow(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                return {"Category": "Date", "Format": "y-MM", "Parser": "arrow"}
            else:
                return {"Category": "Unknown Date"}

        def date_arrow_3(values):
            array_valid = date_arrow(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                return {"Category": "Date", "Format": "y/MM", "Parser": "arrow"}
            else:
                return {"Category": "Unknown Date"}

        def date_arrow_4(values):
            array_valid = date_arrow(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                return {"Category": "Date", "Format": "y.MM", "Parser": "arrow"}
            else:
                return {"Category": "Unknown Date"}

        def date_util(values, seperator):
            util_dates = []
            if seperator != "none":
                dayFirst = dayFirstCheck(values, seperator)
            else:
                dayFirst = False

            for date in values:
                try:

                    dateUtil = dateutil.parser.parse(str(date), dayfirst=dayFirst)
                    if isinstance(dateUtil, datetime.date):
                        util_dates.append({"value": date, "standard": dateUtil})
                    else:
                        print("failed")
                except Exception as e:
                    print(e)

            return util_dates, dayFirst

        # currently can't handle yyyy-dd-mm HH:mm:ss
        def iso_time(values):
            array_valid, dayFirst = date_util(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "iso8601",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date Format"}

        def date_util_1(values):
            array_valid, dayFirst = date_util(values, seperator="-")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "MM-dd-y",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_2(values):
            array_valid, dayFirst = date_util(values, seperator="-")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "MM-dd-y",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_3(values):
            array_valid, dayFirst = date_util(values, seperator="_")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "MM_dd_y",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_4(values):
            array_valid, dayFirst = date_util(values, seperator="_")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "MM_dd_yy",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_5(values):
            array_valid, dayFirst = date_util(values, seperator="/")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "MM/dd/y",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_6(values):
            array_valid, dayFirst = date_util(values, seperator="/")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "MM/dd/yy",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_7(values):
            array_valid, dayFirst = date_util(values, seperator=".")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "MM.dd.y",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_8(values):
            array_valid, dayFirst = date_util(values, seperator=".")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "MM.dd.yy",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_9(values):
            array_valid, dayFirst = date_util(values, seperator="-")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "d-MM-y",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_10(values):
            array_valid, dayFirst = date_util(values, seperator="-")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "d-MM-yy",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_11(values):
            array_valid, dayFirst = date_util(values, seperator="_")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "d_MM_y",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_12(values):
            array_valid, dayFirst = date_util(values, seperator="_")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "d_MM_yy",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_13(values):
            array_valid, dayFirst = date_util(values, seperator="/")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "d/MM/y",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_14(values):

            array_valid, dayFirst = date_util(values, seperator="/")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "d/MM/yy",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_15(values):
            array_valid, dayFirst = date_util(values, seperator=".")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "d.MM.y",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_16(values):
            array_valid, dayFirst = date_util(values, seperator=".")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "d.MM.yy",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_17(values):
            array_valid, dayFirst = date_util(values, seperator="_")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "y_MM_dd",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_18(values):
            array_valid, dayFirst = date_util(values, seperator=".")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "y.MM.dd",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_19(values):
            array_valid, dayFirst = date_util(values, seperator="-")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "y-MM-dd",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_util_20(values):
            array_valid, dayFirst = date_util(values, seperator="/")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "y/MM/dd",
                    "Parser": "Util",
                    "DayFirst": dayFirst,
                }
            else:
                return {"Category": "Unknown Date"}

        def date_long_1(values):
            #              #  01 April 2008
            array_valid, dayFirst = date_util(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                return {"Category": "Date", "Format": "dd LLLL y", "Parser": "Util"}
            else:
                return {"Category": "Unknown Date"}

        def date_long_2(values):
            array_valid, dayFirst = date_util(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                #                 02 April 20
                #                    dd/LLLL/yy
                return {"Category": "Date", "Format": "dd LLLL yy", "Parser": "Util"}
            else:
                return {"Category": "Unknown Date"}

        def date_long_3(values):
            array_valid, dayFirst = date_util(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                return {
                    "Category": "Date",
                    "Format": "EEEE, LLLL dd,yy",
                    "Parser": "Util",
                }
            else:
                return {"Category": "Unknown Date"}

        def date_long_4(values):
            array_valid, dayFirst = date_util(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                #                 April 10, 2008
                #                 LLLL dd, y
                return {"Category": "Date", "Format": "LLLL dd, y", "Parser": "Util"}
            else:
                return {"Category": "Unknown Date"}

        def date_long_5(values):
            array_valid, dayFirst = date_util(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                #  Thursday, April 10, 2008 6:30:00 AM
                #                 EEEE, LLLL dd,yy HH:mm:ss
                return {
                    "Category": "Date",
                    "Format": "EEEE, LLLL dd,yy HH:mm:ss",
                    "Parser": "Util",
                }
            else:
                return {"Category": "Unknown Date"}

        def date_long_6(values):

            array_valid, dayFirst = date_util(values, seperator="none")
            if len(array_valid) > len(values) * 0.85:
                #              03/23/21 01:55 PM
                #                 MM/dd/yy HH:mm
                return {"Category": "Date", "Format": "MM/dd/yy HH:mm"}
            else:
                return {"Category": "Unknown Date"}

        def month_day_f(values):
            month_day_results = []
            for i, md in enumerate(values):
                try:
                    if str.isdigit(md):
                        if int(md) <= 12 and int(md) >= 1:
                            month_day_results.append("month_day")
                        elif int(md) > 12 and int(md) <= 31:
                            month_day_results.append("day")
                        else:
                            month_day_results.append("failed")
                    else:
                        print("not a valid digit")
                except Exception as e:
                    print(e)

            if "failed" in month_day_results:
                return {"Category": "None"}
            elif "day" in month_day_results:
                return {"Category": "Day Number"}
            elif "month_day" in month_day_results:
                return {"Category": "Month Number"}
            else:
                return {"Category": "None"}

        def month_name_f(values):
            print("start month lookup")
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
                return {"Category": "Month Name"}
            else:
                return day_name_f(values)

        def day_name_f(values):
            print("start day lookup")
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
                return {"Category": "Day Name"}
            else:
                return {"Category": "None"}

        functionlist = defaultdict(
            int,
            {
                "None": none_f,
                "country_name": country_f,
                "city": country_f,
                "language_name": country_f,
                "city_suffix": country_f,
                "country_GID": country_iso3,
                "country_code": country_iso2,
                "continent": continent_f,
                "geo": geo_f,
                "pyfloat": geo_f,
                "percent": geo_f,
                "first_name": none_f,
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
                "date_long_dmonthY": date_long_1,
                "date_long_dmonthy": date_long_2,
                "date_long_dmdy": date_long_3,
                "date_long_mdy": date_long_4,
                "date_long_dmdyt": date_long_5,
                "date_long_mdyt_m": date_long_6,
            },
        )
        final_column_classification = []

        for pred in predictions:

            fun = []
            try:
                fun.append(
                    functionlist[pred["avg_predictions"]["averaged_top_category"]](
                        self.column_value_object[pred["column"]]
                    )
                )
            except Exception as e:
                print(e)

            final_column_classification.append(
                {"column": pred["column"], "classification": fun}
            )

        return final_column_classification

    def fuzzymatchColumns(self, classifications):
        predictions = classifications
        words_to_check = [
            "Date",
            "Datetime",
            "Timestamp",
            "Epoch",
            "Time",
            "Year",
            "Month",
            "Lat",
            "Latitude",
            "lng",
            "Longitude",
            "Geo",
            "Coordinates",
            "Location",
            "location",
            "West",
            "South",
            "East",
            "North",
            "Country",
            "CountryName",
            "CC",
            "CountryCode",
            "State",
            "City",
            "Town",
            "Region",
            "Province",
            "Territory",
            "Address",
            "ISO2",
            "ISO3",
            "ISO_code",
            "Results",
        ]

        for i, pred in enumerate(predictions):
            for cc in words_to_check:
                try:
                    if self.fuzzyMatch(str(pred["column"]), str(cc), 85):
                        predictions[i]["fuzzyColumn"] = cc
                    else:
                        pass
                except Exception as e:
                    print(e)
        return predictions

    def standard_dateColumns(self, fuzzyOutput, formats='default'):
        df = self.df
        for i, out in enumerate(fuzzyOutput):
            try:
                #             print(out['classification'][0]['Category'])
                if out['classification'][0]['Category'] == 'Date' or out['fuzzyColumn'] == 'Date' or out[
                    'fuzzyColumn'] == 'Timestamp' or out['fuzzyColumn'] == 'Datetime':

                    new_column = 'ISO_8601_' + str(i)
                    if 'DayFirst' in out['classification'][0]:

                        dayFirst = out['classification'][0]['DayFirst']
                        print('dayFirst', dayFirst)
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

    def columns_classified(self, path):
        preds = self.predictions(path_to_csv=path)
        output = self.assign_heuristic_function(preds)
        output_fuz = self.fuzzymatchColumns(output)
        return output_fuz

    def add_iso8601_columns(self, path, formats):
        preds = self.predictions( path_to_csv=path)
        output = self.assign_heuristic_function(preds)
        output_fuz = self.fuzzymatchColumns(output)
        output_col = self.standard_dateColumns(output_fuz, formats)

        return output_col

