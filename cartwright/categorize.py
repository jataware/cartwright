#!/usr/bin/env python
from __future__ import unicode_literals, print_function, division

import torch
import torch.autograd as autograd
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from collections import defaultdict

# Kyle's attempt
import pandas as pd
import numpy as np

import random
import dateutil.parser
import datetime
import arrow
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import fuzzywuzzy
import pkg_resources
from . import schemas
from . import time_resolution
import time

import logging

from .LSTM import PaddedTensorDataset
from .CartWrightBase import CartWrightBase

from .utils import days_of_the_week , \
                  months_of_the_year , \
                  columns_to_classify_and_skip_if_found , \
                  columns_to_classify_if_found , \
                  character_tokins, \
                  fuzzy_match,fuzzy_ratio,\
                  build_return_standard_object
from cartwright.category_helpers import return_all_category_classes_and_labels, generate_label_id


# Set log level and formatter
logging.getLogger().setLevel(level="ERROR")
logging.basicConfig(format='%(levelname)s - %(asctime)s %(message)s')


def timeout():
    return build_return_standard_object(category="timeout", subcategory=None, match_type=[])


def skipped( column, fuzzy_matched):
    category = None
    subcategory = None
    match_type = None
    try:
        for match in fuzzy_matched:
            if column == match['header']:
                match_type = 'fuzzy'
                category = "geo"
                subcategory = match['value']

        return build_return_standard_object(category=category, subcategory=subcategory, match_type=match_type)
    except Exception as e:
        logging.error(f'Skipped validation error: {e}')
        return build_return_standard_object(category=None, subcategory=None, match_type=None)


class CartwrightClassify(CartWrightBase):
    def __init__(self, number_of_samples, seconds_to_finish=40):
        super().__init__()
        self.model.load_state_dict(
            torch.load(pkg_resources.resource_stream(__name__, 'models/LSTM_RNN_CartWright_v_0.0.0.1_dict.pth')))
        self.model.eval()
        self.number_of_random_samples = number_of_samples
        #       prediction tensors with the best match being less than predictionLimit will not be returned
        self.predictionLimit = -4.5
        self.country_lookup = pd.read_csv(pkg_resources.resource_stream(__name__, 'resources/country_lookup.csv'),
                                          encoding='latin-1')
        self.city_lookup = pd.read_csv(pkg_resources.resource_stream(__name__, 'resources/city.csv'),
                                       encoding='latin-1')
        self.city_lookup = np.asarray(self.city_lookup["city"])
        self.state_lookup = pd.read_csv(pkg_resources.resource_stream(__name__, 'resources/states_provinces_lookup.csv'),
                                        encoding='latin-1')
        self.state_lookup = np.asarray(self.state_lookup["state_name"])
        self.country_name = np.asarray(self.country_lookup["country_name"])
        self.iso3_lookup = np.asarray(self.country_lookup["Alpha-3_Code"])
        self.iso2_lookup = np.asarray(self.country_lookup["Alpha-2_Code"])
        self.cont_lookup = pd.read_csv(pkg_resources.resource_stream(__name__, 'resources/continent_lookup.csv'),
                                       encoding='latin-1')
        self.cont_lookup = np.asarray(self.cont_lookup["continent_name"])
        self.fake_data = pd.read_csv(pkg_resources.resource_stream(__name__, 'datasets/Fake_data.csv'), #cartwright/datasets/Fake_data.csv
                                    encoding='latin-1')
        self.seconds_to_finish = seconds_to_finish
        self.days_of_week = days_of_the_week
        self.month_of_year = months_of_the_year
        self.all_classes = return_all_category_classes_and_labels()
        self.all_labels = np.array(list(self.all_classes.keys()))
        self.label2id = generate_label_id(self.all_labels)
        self.n_categories = len(self.label2id)


    def evaluate_test_set(self, test):
        predictions = []

        for batch, targets, lengths, raw_data in self.create_dataset(
                test, batch_size=1
        ):
            batch, targets, lengths = self.sort_batch(batch, targets, lengths)
            # print(batch,targets,lengths)
            # print(torch.autograd.Variable(batch))
            pred = self.model(torch.autograd.Variable(batch), lengths.cpu().numpy())
            pred_idx = torch.max(pred, 1)[1]
            def get_key(val):
                for key, value in self.label2id.items():
                    if val == value:
                        return {"top_pred": key, "tensor": pred, "pred_idx": pred_idx}

            predictions.append(get_key(pred_idx[0]))
        return predictions

    def read_in_csv(self, path):
        self.df = pd.read_csv(path)
        return self.df

    def generate_column_values_dict(self, index_remove):
        column_value_object = {}

        for i, column in enumerate(self.df.columns):
            column_value_object[column] = []
            if i in index_remove:
                pass
            else:
                for _ in range(1, self.number_of_random_samples):
                    random_values = str(np.random.choice(self.df[column].dropna()))
                    column_value_object[column].append(random_values)
        return column_value_object

    def averaged_predictions(self, all_predictions):
        all_arrays = []
        for pred in all_predictions:
            all_arrays.append(pred["tensor"].detach().numpy())

        out = np.mean(all_arrays, axis=0)
        maxValue = np.amax(out)

        def get_key(val):
            for key, value in self.label2id.items():
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

    def predictions(self, index_remove):
        logging.info('Start LSTM predictions ...')
        print("starting lstm")
        column_value_object = self.generate_column_values_dict( index_remove)
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
                    logging.error(f"predictions {column}: {e}")

        return predictionList

    def assign_heuristic_function(self, predictions, fuzzy_matched):
        final_column_classification = []

        def add_obj(obj, add_objs):
            for property in add_objs:
                obj[property] = add_objs[property]

            return obj

        currentTime = time.perf_counter()
        count = 0
        timesUp = time.perf_counter()
        while currentTime - timesUp < self.seconds_to_finish and count < len(predictions):
            category=None
            class_=None
            print(predictions[count])


            try:
                if predictions[count]['values'] == 'Skipped':
                    pass
                else:
                    category = predictions[count]["avg_predictions"]["averaged_top_category"]
                    class_ = self.all_classes[category]
                    final_column_classification.append(
                        add_obj({"column": predictions[count]["column"]},
                                class_.validate(
                                    self.column_value_object[predictions[count]["column"]]
                                )))

            except Exception as e:
                logging.error(f"While loop failed: {e}")
            count += 1
            currentTime = time.perf_counter()

        # if the model ends before it is finished we want to make sure we are still classifying the skipped values.
        # also we want to keep track of what index are the skipped columns
        additionalColumnClassified = []
        for i, pred in enumerate(predictions):
            try:
                if pred['values'] == 'Skipped':
                    final_column_classification.append(
                        add_obj({"column": pred["column"]}, skipped(
                            pred['column'], fuzzy_matched
                        ))
                    )
                    if i > count:
                        additionalColumnClassified.append(i)

            except Exception as e:
                logging.error(f"assign_heuristic_function - {pred}: {e}")

        # if we skipped a column return 'timeout' for category
        for i, p in enumerate(predictions):
            if i < count:
                pass
            elif i in additionalColumnClassified:
                pass
            else:
                final_column_classification.append(
                    add_obj({"column": p["column"]}, timeout())
                )

        return final_column_classification

    def fuzzy_match_columns(self, predictions):

        for i, pred in enumerate(predictions):
            for y, keyValue in enumerate(columns_to_classify_if_found):
                try:
                    for key in keyValue:
                        if fuzzy_match(str(pred["column"]), str(key), 85):
                            T, ratio = fuzzy_ratio(str(pred["column"]), str(key), 85)
                            predictions[i]['match_type'].append('fuzzy')
                            predictions[i]["fuzzyColumn"] = []
                            predictions[i]["fuzzyColumn"].append(
                                {"matchedKey": str(key), "fuzzyCategory": columns_to_classify_if_found[y][key], "ratio": ratio})

                except Exception as e:
                    logging.error(f"fuzzy_match_columns - {keyValue}: {e}")
        # return only the hightest fuzzy match value
        for pred2 in predictions:
            try:
                if len(pred2["fuzzyColumn"]) > 1:
                    ind = 0
                    bestRatio = 0
                    for i, fmatch in pred2['fuzzyColumn']:
                        if fmatch["ratio"] > bestRatio:
                            bestRatio = fmatch["ratio"]
                            ind = i
                    pred2['fuzzyColumn'] = pred2['fuzzyColumn'][ind]
                else:
                    pred2['fuzzyColumn'] = pred2['fuzzyColumn'][0]
            except Exception as e:
                logging.warning(pred2['column'], f"fuzzy_match_columns - Column has no fuzzy match: {e}")

        return predictions

    def skip_matched_columns(self):
        array_of_columnMatch_index = []
        index_to_not_process = []
        for i, header in enumerate(self.df.columns):

            for y, keyValue in enumerate(columns_to_classify_and_skip_if_found):
                for key in keyValue:
                    if fuzzy_match(header, str(key), 90):
                        T, ratio = fuzzy_ratio(header, str(key), 90)
                        index_to_not_process.append(i)
                        array_of_columnMatch_index.append(
                            {'index': i, 'header': header, 'key': key, 'value': keyValue[key], 'ratio': ratio})
                    else:
                        pass

        return index_to_not_process, array_of_columnMatch_index

    def final_step(self, t):
        classifiedObjs = []
        for tstep in t:

            tstep['match_type'] = list(set(tstep['match_type']))
            tstep['match_type'] = [i for i in tstep['match_type'] if i]
            categoryValue=tstep["category"]
            subcategoryValue=tstep["subcategory"]
            fuzzyCol = None
            try:
                fuzzyCol = tstep['fuzzyColumn']
                if categoryValue == None:
                    if fuzzyCol['fuzzyCategory'] in ["Year", "Date", "Datetime", "Timestamp", "Epoch", "Time", "Month"]:
                        categoryValue="time"
                        subcategoryValue="date"
                    elif fuzzyCol["fuzzyCategory"] in ["Geo", "Coordinates", "Location", "Address"]:
                        categoryValue ="geo"
                        subcategoryValue=None
                    elif fuzzyCol["fuzzyCategory"] in ["Country", "CountryName", "CountryCode"]:
                        categoryValue = "geo"
                        subcategoryValue = "country"
                    elif fuzzyCol["fuzzyCategory"] in ["State", "Town", "City", "Region", "Province", "Territory"]:
                        categoryValue="geo"
                        subcategoryValue=fuzzyCol["fuzzyCategory"].lower()
                    else:
                        pass

            except Exception as e:
                logging.info(tstep, f"final_step - Column has no fuzzy match:{e}")

            classifiedObj = schemas.Classification(column=tstep['column'], category=categoryValue,
                                                          subcategory=subcategoryValue,
                                                          format=tstep['format'],
                                                          match_type=tstep['match_type'], Parser=tstep['Parser'],
                                                          DayFirst=tstep['DayFirst'],
                                                          fuzzyColumn=fuzzyCol)
            classifiedObjs.append(classifiedObj)
        return schemas.Classifications(classifications=classifiedObjs)

    def find_NANs(self):
        index_nan = []
        for index, column in enumerate(self.df.columns):
            if self.df[column].count() > 0:
                pass
            else:
                index_nan.append(index)

        return index_nan

    def predict_temporal_resolution(self, final: schemas.Classifications) -> schemas.Classifications:
        found_time = False
        for classification in final.classifications:
            try:

                if classification.category != schemas.category.time:
                    continue
                if classification.format is None:
                    continue
                found_time = True

                #convert the datetime strings in the dataframe to unix timestamps using the classification format
                times = self.df[classification.column].to_list()
                times = [datetime.datetime.strptime(str(time_), classification.format).replace(tzinfo=datetime.timezone.utc).timestamp() for time_ in times]
                times = np.array(times)

                classification.time_resolution = time_resolution.detect_resolution(times)
            except Exception as e:
                print(f'error {e}')

        if not found_time:
            logging.warning("No time columns found to predict temporal resolution")

        return final
    
    def columns_classified(self, path):
        logging.info('starting classification')
        self.read_in_csv(path)
        index_remove, fuzzyMatchColumns = self.skip_matched_columns()
        columns_na = self.find_NANs()
        index_remove = index_remove + columns_na
        preds = self.predictions(index_remove)
        output = self.assign_heuristic_function(preds, fuzzyMatchColumns)
        fuzzyMatch = self.fuzzy_match_columns(output)
        final = self.final_step(fuzzyMatch)
        final = self.predict_temporal_resolution( final)
        return final


    def show_values(self):
        print(self.model)
        # print(self.number_of_random_samples)
        # print(self.predictionLimit)
        # print(self.country_lookup)
        # print(self.city_lookup)
        # print(self.state_lookup)
        # print(self.country_name)
        # print(self.number_of_random_samples)
        # print(self.iso3_lookup)
        # print(self.iso2_lookup)
        # print(self.cont_lookup)
        # print(self.fake_data)
        # print(self.seconds_to_finish )
        # print(self.days_of_week)
        # print(self.month_of_year)
        # print(self.all_classes)
        # print(self.all_labels)
        # print(self.label2id.items())
        # print(self.n_categories)

if __name__ == '__main__':
    gc=CartwrightClassify(10)
    preds=gc.columns_classified('datasets/Fake_data.csv')
    # preds=gc.vectorized_array(['hi','there'])
    print(preds)
    # print(gc.pad_sequences(preds, torch.LongTensor([len(s) for s in preds])) )
    # print(torch.LongTensor([len(s) for s in preds]))
    gc.show_values()