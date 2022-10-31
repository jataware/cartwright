#!/usr/bin/env python
from __future__ import unicode_literals, print_function, division

import torch
import pandas as pd
import numpy as np
import argparse
import datetime
import pkg_resources
import time
import logging

from . import schemas
from . import time_resolution
from .CartwrightBase import CartwrightBase

from .utils import (
    columns_to_classify_and_skip_if_found,
    columns_to_classify_if_found,
    fuzzy_match,
    fuzzy_ratio,
    build_return_standard_object,
)
from cartwright.category_helpers import (
    return_all_category_classes_and_labels,
    generate_label_id,
)

# Set log level and formatter
logging.getLogger().setLevel(level="ERROR")
logging.basicConfig(format="%(levelname)s - %(asctime)s %(message)s")


def timeout():
    return build_return_standard_object(
        category="timeout", subcategory=None, match_type=[]
    )

def skipped(column, fuzzy_matched):
    category = None
    subcategory = None
    match_type = None
    try:
        for match in fuzzy_matched:
            if column == match["header"]:
                match_type = "fuzzy"
                category = "geo"
                subcategory = match["value"]

        return build_return_standard_object(
            category=category, subcategory=subcategory, match_type=match_type
        )
    except Exception as e:
        logging.error(f"Skipped validation error: {e}")
        return build_return_standard_object(
            category=None, subcategory=None, match_type=None
        )

class CartwrightClassify(CartwrightBase):
    def __init__(self,model_version='0.0.1', number_of_samples=100, seconds_to_finish=40):
        super().__init__()
        self.model_version=model_version
        self.model.load_state_dict(
            torch.load(
                pkg_resources.resource_stream(
                    __name__, f"models/LSTM_RNN_Cartwright_v_{self.model_version}_dict.pth"
                )
            )
        )
        self.model.eval()
        self.number_of_random_samples = number_of_samples
        #       prediction tensors with the best match being less than predictionLimit will not be returned
        self.predictionLimit = -4.5
        self.fake_data = pd.read_csv(
            pkg_resources.resource_stream(
                __name__, "datasets/fake_data.csv"
            ),  # cartwright/datasets/fake_data.csv
            encoding="latin-1",
        )
        self.seconds_to_finish = seconds_to_finish
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
        backup_out=out.copy()[0]
        maxValue = np.amax(out)
        backup_ind = np.argpartition(backup_out, -5)[-5:]
        # # remove max value
        # backup_ind=np.delete(backup_ind,np.where(backup_ind == np.argmax(backup_out)))
        def get_key(val):
            for key, value in self.label2id.items():
                if val == value:
                    return key

        topcat = get_key(np.argmax(out))

        top_categories= {}
        for ind in backup_ind:
            if backup_out[ind]>self.predictionLimit:
                top_categories[get_key(ind)]= backup_out[ind]

        sorted_top_categories=sorted(top_categories.items(), key=lambda x: x[1], reverse=True)

        # print(f'sorted {sorted_top_categories}')
        return {
            "averaged_tensor": out,
            "averaged_top_category": {True: "None", False: topcat}[
                maxValue < self.predictionLimit
            ],
            "top_categories":sorted_top_categories
        }

    def return_data(self):
        return self.cont_lookup

    def predictions(self, index_remove):
        logging.info("Start LSTM predictions ...")
        print("starting lstm")
        column_value_object = self.generate_column_values_dict(index_remove)
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
            category_name = None
            category_obj = None
            try:
                if predictions[count]["values"] == "Skipped":
                    pass
                else:
                    series = self.column_value_object[predictions[count]["column"]]
                    total_sample_count = len(series)
                    final_categorization=None
                    for category_name, _ in predictions[count]["avg_predictions"]["top_categories"]:
                        # print(f'predictions {category_name}')
                        category_obj = self.all_classes[category_name]
                        valid_sample_count = category_obj.validate_series(series=series)

                        try:
                            final_categorization = category_obj.pass_validation(
                                valid_sample_count,
                                total_sample_count,
                            )
                            break
                        except Exception as e:
                            print(e)

                    if final_categorization is None:
                        final_categorization = build_return_standard_object(
                            category=None, subcategory=None, match_type=None
                        )
                    final_column_classification.append(
                        add_obj(
                            {"column": predictions[count]["column"]},
                            final_categorization,
                        )
                    )
            except Exception as e:
                logging.error(f"While loop failed: {e}")
            count += 1
            currentTime = time.perf_counter()

        # if the model ends before it is finished we want to make sure we are still classifying the skipped values.
        # also we want to keep track of what index are the skipped columns
        additionalColumnClassified = []
        for i, pred in enumerate(predictions):
            try:
                if pred["values"] == "Skipped":
                    final_column_classification.append(
                        add_obj(
                            {"column": pred["column"]},
                            skipped(pred["column"], fuzzy_matched),
                        )
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
                            predictions[i]["match_type"].append("fuzzy")
                            predictions[i]["fuzzyColumn"] = []
                            predictions[i]["fuzzyColumn"].append(
                                {
                                    "matchedKey": str(key),
                                    "fuzzyCategory": columns_to_classify_if_found[y][
                                        key
                                    ],
                                    "ratio": ratio,
                                }
                            )

                except Exception as e:
                    logging.error(f"fuzzy_match_columns - {keyValue}: {e}")
        # return only the hightest fuzzy match value
        for pred2 in predictions:
            try:
                if len(pred2["fuzzyColumn"]) > 1:
                    ind = 0
                    bestRatio = 0
                    for i, fmatch in pred2["fuzzyColumn"]:
                        if fmatch["ratio"] > bestRatio:
                            bestRatio = fmatch["ratio"]
                            ind = i
                    pred2["fuzzyColumn"] = pred2["fuzzyColumn"][ind]
                else:
                    pred2["fuzzyColumn"] = pred2["fuzzyColumn"][0]
            except Exception as e:
                logging.warning(
                    pred2["column"],
                    f"fuzzy_match_columns - Column has no fuzzy match: {e}",
                )

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
                            {
                                "index": i,
                                "header": header,
                                "key": key,
                                "value": keyValue[key],
                                "ratio": ratio,
                            }
                        )
                    else:
                        pass

        return index_to_not_process, array_of_columnMatch_index

    def build_skipped_categorization(self, t):
        classifiedObjs = []
        for tstep in t:
            # print(f't {tstep}')
            tstep["match_type"] = list(set(tstep["match_type"]))
            tstep["match_type"] = [i for i in tstep["match_type"] if i]
            categoryValue = tstep["category"]
            subcategoryValue = tstep["subcategory"]
            fuzzyCol = None
            try:
                fuzzyCol = tstep["fuzzyColumn"]
                if categoryValue == None:
                    if fuzzyCol["fuzzyCategory"] in [
                        "Year",
                        "Date",
                        "Datetime",
                        "Timestamp",
                        "Epoch",
                        "Time",
                        "Month",
                    ]:
                        categoryValue = "time"
                        subcategoryValue = "date"
                    elif fuzzyCol["fuzzyCategory"] in [
                        "Geo",
                        "Coordinates",
                        "Location",
                        "Address",
                    ]:
                        categoryValue = "geo"
                        subcategoryValue = None
                    elif fuzzyCol["fuzzyCategory"] in [
                        "Country",
                        "CountryName",
                        "CountryCode",
                    ]:
                        categoryValue = "geo"
                        subcategoryValue = "country"
                    elif fuzzyCol["fuzzyCategory"] in [
                        "State",
                        "Town",
                        "City",
                        "Region",
                        "Province",
                        "Territory",
                    ]:
                        categoryValue = "geo"
                        subcategoryValue = fuzzyCol["fuzzyCategory"].lower()
                    else:
                        pass

            except Exception as e:
                logging.info(tstep, f"final_step - Column has no fuzzy match:{e}")

            classifiedObj = schemas.Classification(
                column=tstep["column"],
                category=categoryValue,
                subcategory=subcategoryValue,
                format=tstep["format"],
                match_type=tstep["match_type"],
                Parser=tstep["Parser"],
                fuzzyColumn=fuzzyCol,
            )
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

    def predict_temporal_resolution(
        self, final: schemas.Classifications
    ) -> schemas.Classifications:
        found_time = False
        for classification in final.classifications:
            try:

                if classification.category != schemas.Category.time:
                    continue
                if classification.format is None:
                    continue
                found_time = True

                # convert the datetime strings in the dataframe to unix timestamps using the classification format
                times = self.df[classification.column].to_list()
                times = [
                    datetime.datetime.strptime(str(time_), classification.format)
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                    for time_ in times
                ]
                times = np.array(times)

                classification.time_resolution = time_resolution.detect_resolution(
                    times
                )
            except Exception as e:
                print(f"error {e}")

        if not found_time:
            logging.warning("No time columns found to predict temporal resolution")

        return final

    def columns_classified(self, df=None, path=None):
        logging.info("starting classification")
        if path is not None:
            self.read_in_csv(path)
        if df is not None:
            self.df = df
        index_remove, fuzzy_matched_columns = self.skip_matched_columns()
        columns_na = self.find_NANs()
        index_remove = index_remove + columns_na
        preds = self.predictions(index_remove)
        output = self.assign_heuristic_function(preds, fuzzy_matched_columns)
        fuzzyMatch = self.fuzzy_match_columns(output)
        final = self.build_skipped_categorization(fuzzyMatch)
        final = self.predict_temporal_resolution(final)
        return final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="cartwright/datasets/fake_data.csv", help="path to csv")
    parser.add_argument("--num_samples",type=int, default=100, help="number of samples to test from each column")
    parser.add_argument("--model_version",default="0.0.1", help='model version you would like to run')
    args = parser.parse_args()
    cartwright = CartwrightClassify(model_version=args.model_version, number_of_samples=args.num_samples)
    preds = cartwright.columns_classified(path=args.path)
    print(preds)
    return preds


if __name__ == "__main__":
    main()
