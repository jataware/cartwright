import numpy as np
from faker import Faker
import logging
import pandas as pd
import pkg_resources
import random
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import fuzzywuzzy
import arrow
import datetime
import dateutil.parser
from collections import defaultdict

from cartwright.utils import (
    build_return_date_object,
    build_return_standard_object,
    build_return_timespan,
)


class CategoryBase:
    #global/static variables
    country_lookup = pd.read_csv(
        pkg_resources.resource_stream(__name__, "resources/country_lookup.csv"),
        encoding="latin-1",
    )
    city_lookup = pd.read_csv(
        pkg_resources.resource_stream(__name__, "resources/city_lookup.csv"),
        encoding="latin-1",
    )
    state_lookup = pd.read_csv(
        pkg_resources.resource_stream(__name__, "resources/states_provinces_lookup.csv"),
        encoding="latin-1",
    )
    cont_lookup = pd.read_csv(
        pkg_resources.resource_stream(__name__, "resources/continent_lookup.csv"),
        encoding="latin-1",
    )
    fake = Faker()

    def __init__(self):
        self.city_lookup = np.asarray(self.city_lookup["city"])
        self.state_lookup = np.asarray(self.state_lookup["state_name"])
        self.country_name = np.asarray(self.country_lookup["country_name"])
        self.iso3_lookup = np.asarray(self.country_lookup["Alpha-3_Code"])
        self.iso2_lookup = np.asarray(self.country_lookup["Alpha-2_Code"])
        self.cont_names = np.asarray(self.cont_lookup["continent_name"])
        self.cont_codes=self.cont_lookup['continent_code'].unique()
        self.cont_codes[1]='NA' #fix continent code NA for north america converted to nan
        self.threshold=.85
        self.category = None

    def validate_series(self, series):
        valid_samples=0
        for sample in series:
            try:
                if self.validate(sample):
                    valid_samples+=1
            except Exception as e:
                print(e)
        return valid_samples


    def class_name(self):
        return self.__class__.__name__

    def get_fake_date(self, lab):
        return str(self.fake.date(pattern=lab))

    def return_label(self):
        try:
            if self.format:
                return self.format
        except:
            return self.class_name()

    def space_seperator(self, val):
        options = [True, False]
        if np.random.choice(options):
            return " " + val + " "
        return val

    # helpers for validation
    def exception_category(self, e):
        logging.error(f"{self.return_label()} validation error: {e}")
        return build_return_standard_object(
            category=None, subcategory=None, match_type=None
        )

    def not_classified(self):
        return build_return_standard_object(
            category=None, subcategory=None, match_type=None
        )
    def threshold_check(self, number_validated, number_of_samples, threshold):
        if number_validated >= number_of_samples * threshold:
            return build_return_standard_object(category=self.category, subcategory=str(self.return_label()), match_type="LSTM")
        raise
    def pass_validation(self, number_validated, number_of_samples):
        return self.threshold_check(number_validated,number_of_samples,self.threshold)

class MiscBase(CategoryBase):
    def __init__(self):
        super().__init__()


    def validate_series(self,series):
        return 0




class GeoBase(CategoryBase):
    def __init__(self):
        super().__init__()
        self.category="geo"



class DateBase(CategoryBase):
    def __init__(self):
        super().__init__()
        self.threshold = 0.85
        self.format = None
        self.category="date"

    def generate_training_data(self):
        assert self.format is not None, "Format must be set before generating training data. Or this method should be overridden."
        return self.format, self.get_fake_date(self.format)


    def validate(self, value):
        return self.is_date_(value)

    def pass_validation(self, number_validated, number_of_samples):
        return self.threshold_check(number_validated,number_of_samples,self.threshold)

    def validate_years(self,series):
        valid_years = 0
        for year in series:
            if str.isdigit(str(year).strip()):
                if 1700 < int(year) < 2200:
                    valid_years += 1
        if valid_years == len(series):
            return True


    def is_date_(self, date):
        #try to parse the date according to the format
        #if it parsed, return True, otherwise an exception will be raised
        datetime.datetime.strptime(date, self.format)
        return True 


    def is_date_arrow(self,date):
        dateArrow = arrow.get(str(date), normalize_whitespace=True).datetime

        if isinstance(dateArrow, datetime.date):
            return True

    def threshold_check(self, number_validated, number_of_samples, threshold):
        if number_validated >= number_of_samples * threshold:
            return build_return_date_object(format=self.return_label())
        raise

class TimespanBase(DateBase):
    def __init__(self):
        super().__init__()

    def threshold_check(self, number_validated, number_of_samples, threshold):
        if number_validated >= number_of_samples * threshold:
            return build_return_timespan(format=self.return_label())
        raise