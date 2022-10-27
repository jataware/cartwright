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

from cartwright.utils import (
    build_return_date_object,
    build_return_standard_object,
    fuzzy_match,
)


class CategoryBase:
    #global/static variables
    country_lookup = pd.read_csv(
        pkg_resources.resource_stream(__name__, "../resources/country_lookup.csv"),
        encoding="latin-1",
    )
    city_lookup = pd.read_csv(
        pkg_resources.resource_stream(__name__, "../resources/city_lookup.csv"),
        encoding="latin-1",
    )
    state_lookup = pd.read_csv(
        pkg_resources.resource_stream(__name__, "../resources/states_provinces_lookup.csv"),
        encoding="latin-1",
    )
    cont_lookup = pd.read_csv(
        pkg_resources.resource_stream(__name__, "../resources/continent_lookup.csv"),
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


class GeoBase(CategoryBase):
    def __init__(self):
        super().__init__()

    def validate_country(self, values):
        try:
            logging.info("Start country validation ...")
            country_match_bool = []
            subsample = len(values)
            count = 0
            passed = 0
            while passed < 5 and not count >= subsample:
                count += 1
                try:
                    match = fuzzywuzzy.process.extractOne(
                        values[count], self.country_name, scorer=fuzz.token_sort_ratio
                    )
                    if match is not None:
                        if match[1] > 90:
                            country_match_bool.append(True)
                            passed += 1
                except Exception as e:
                    logging.error(f"country_f - {values}: {e}")

            if np.count_nonzero(country_match_bool) >= 5:
                logging.info("country validated")
                return build_return_standard_object(
                    category="geo", subcategory="country_name", match_type="LSTM"
                )
            else:
                return self.validate_state(values)
        except Exception as e:
            logging.error(f"country error: {e}")
            return build_return_standard_object(
                category=None, subcategory=None, match_type=None
            )

    def validate_state(self, values):
        try:
            logging.info("Start state validation ...")
            state_match_bool = []
            subsample = len(values)
            count = 0
            passed = 0
            while passed < 5 and not count >= subsample:
                count += 1
                try:
                    match = fuzzywuzzy.process.extractOne(
                        values[count], self.state_lookup, scorer=fuzz.token_sort_ratio
                    )
                    if match is not None:
                        if match[1] > 90:
                            state_match_bool.append(True)
                            passed += 1
                except Exception as e:
                    logging.error(f"state_f -{values}: {e}")

            if np.count_nonzero(state_match_bool) >= 5:
                logging.info("state validated")
                return build_return_standard_object(
                    category="geo", subcategory="state_name", match_type="LSTM"
                )
            else:
                return self.validate_city(values)
        except Exception as e:
            logging.error(f"state error: {e}")
            return build_return_standard_object(
                category=None, subcategory=None, match_type=None
            )

    def validate_city(self, values):
        try:
            logging.info("Start city validation ...")
            city_match_bool = []
            subsample = 5

            count = 0
            passed = 0
            while passed < 2 and not count >= subsample:
                count += 1
                try:
                    match = fuzzywuzzy.process.extractOne(
                        random.choice(values),
                        self.city_lookup,
                        scorer=fuzz.token_sort_ratio,
                    )
                    if match is not None:
                        if match[1] > 90:
                            city_match_bool.append(True)
                            passed += 1
                except Exception as e:
                    logging.error(f"city_f - {values}: {e}")

            if np.count_nonzero(city_match_bool) >= 2:
                logging.info("city validated")
                return build_return_standard_object(
                    category="geo", subcategory="city_name", match_type="LSTM"
                )
            else:
                return build_return_standard_object(
                    category=None, subcategory=None, match_type=None
                )
        except Exception as e:
            logging.error(f"city error: {e}")
            return build_return_standard_object(
                category=None, subcategory=None, match_type=None
            )

    def validate_iso2(self, values):
        try:
            logging.info("Start iso2 validation ...")
            ISO2_in_lookup = []
            for iso in values:
                for cc in self.iso2_lookup:
                    try:
                        ISO2_in_lookup.append(fuzzy_match(str(iso), str(cc), ratio=85))
                    except Exception as e:
                        logging.error(f"country_iso2 - {values}: {e}")

            if np.count_nonzero(ISO2_in_lookup) >= (len(values) * 0.65):

                return build_return_standard_object(
                    category="geo", subcategory="ISO2", match_type="LSTM"
                )
            else:
                return build_return_standard_object(
                    category=None, subcategory=None, match_type=None
                )
        except Exception as e:
            logging.error(f"country_iso2 error: {e}")
            return build_return_standard_object(
                category=None, subcategory=None, match_type=None
            )

    def validate_geos(self, values):
        try:
            logging.info("Start geo validation ...")
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
                    logging.error(f"geo_f - {values}: {e}")

            if "failed" in geo_valid:
                return build_return_standard_object(
                    category=None, subcategory=None, match_type=None
                )
            elif len(percent_array) >= len(values) * 0.95:
                return build_return_standard_object(
                    category=None, subcategory=None, match_type=None
                )
            elif "lng" in geo_valid:
                return build_return_standard_object(
                    category="geo", subcategory="longitude", match_type="LSTM"
                )
            elif "latlng" in geo_valid:
                return build_return_standard_object(
                    category="geo", subcategory="latitude", match_type="LSTM"
                )
            else:
                return build_return_standard_object(
                    category=None, subcategory=None, match_type=None
                )
        except Exception as e:
            logging.error(f"geo error: {e}")
            return build_return_standard_object(
                category=None, subcategory=None, match_type=None
            )


class DateBase(CategoryBase):
    def __init__(self):
        super().__init__()
        # self.days_of_the_week = [
        #     "Monday",
        #     "Tuesday",
        #     "Wednesday",
        #     "Thursday",
        #     "Friday",
        #     "Saturday",
        #     "Sunday",
        # ]
        self.threshold = 0.85
        self.format = None

    def return_label(self):
        if self.format:
            return f'date_{self.format}'
        return self.class_name()

    def generate_training_data(self):
        assert self.format is not None, "Format must be set before generating training data. Or this method should be overridden."
        return self.format, self.get_fake_date(self.format)
    
    def validate_series(self, series):
        valid_samples=0
        for sample in series:
            try:
                if self.validate(sample):
                    valid_samples+=1

            except Exception as e:
                print(e)
        return valid_samples,len(series)

    def validate(self, value):
        return self.is_date_(value)

    def pass_validation(self, number_validated, number_of_samples):
        return self.threshold_check(number_validated,number_of_samples,self.threshold, self.day_first)

    def month_day_format(self, values, dayFirst=True, separator="-", day_month_locs=[]):

        ## for if we want to figure out dd or d mm or m, but right now d or m is fine.
        # if dayFirst:
        #     dayFormat = self.day_ddOrd(values, separator=separator, loc=day_month_locs[0])
        #     monthFormat = self.month_MMorM(values, separator=separator, loc=day_month_locs[1])
        # else:
        #     dayFormat = self.day_ddOrd(values, separator=separator, loc=day_month_locs[1])
        #     monthFormat = self.month_MMorM(values, separator=separator, loc=day_month_locs[0])

        return self.day_ddOrd(values), self.month_MMorM(values)

    def day_ddOrd(self, values, separator="-", loc=0):
        return "%d"
        # dayFormat = '%-d'
        # for d in values:
        #     try:
        #         if separator is None:
        #             if d[0] == '0':
        #                 dayFormat = '%d'
        #         else:
        #             d_value = d.split(separator)[loc]
        #             if d_value[0] == '0':
        #                 dayFormat = '%d'
        #     except Exception as e:
        #         logging.error(f"day_ddOrd - {d}: {e}")
        #
        # return dayFormat

    # Month Format
    def month_MMorM(self, values, separator="-", loc=0):
        return "%m"
        # monthFormat = '%-m'
        # for d in values:
        #     try:
        #         if separator is None:
        #             if d[0] == '0':
        #                 monthFormat = '%m'
        #         else:
        #             d_value = d.split(separator)[loc]
        #             if d_value[0] == '0':
        #                 monthFormat = '%m'
        #     except Exception as e:
        #         logging.error(f"month_MMorM - {d}:{e}")
        #
        # return monthFormat

    # Hour format
    def hour_hOrH(self, values, separator, loc_hms):
        return "%H"
        # hourFormat = '%-H'
        # for d in values:
        #
        #     if separator is None:
        #         if d[0] == '0':
        #             hourFormat = '%H'
        #     else:
        #         hms = d.split(' ')[-1]
        #         hms = hms.split(separator)[loc_hms]
        #
        #         if hms[0] == '0':
        #             hourFormat = '%H'
        # return hourFormat

    # Minute format
    def minute_mOrM(self, values, separator, loc_hms):
        return "%M"
        # minuteFormat = '%-M'
        # for d in values:
        #     if separator is None:
        #         if d[0] == '0':
        #             minuteFormat = '%M'
        #     else:
        #         hms = d.split(' ')[-1]
        #         hms = hms.split(separator)[loc_hms]
        #         if hms[0] == '0':
        #             minuteFormat = '%M'
        # return minuteFormat

    # Second format
    def second_sOrS(self, values, separator, loc_hms):
        return "%S"
        # secondFormat = '%-S'
        # for d in values:
        #     if separator is None:
        #         if d[0] == '0':
        #             secondFormat = '%S'
        #     else:
        #         hms = d.split(' ')[-1]
        #         hms = hms.split(separator)[loc_hms]
        #         if hms[0] == '0':
        #             secondFormat = '%S'
        # return secondFormat

    def day_first_check(self, values, separator, shortYear, yearLoc):
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
                logging.error(f"dayFirstCheck - {date}: {e}")

        return False

    def date_arrow(self, values, separator):
        utils_array = []
        for date in values:
            try:
                dateArrow = arrow.get(str(date), normalize_whitespace=True).datetime

                if isinstance(dateArrow, datetime.date):
                    utils_array.append("true")
                else:
                    logging.info(f"{date}: Not valid format")
            except Exception as e:
                logging.error(f"date_arrow - {date}: {e}")

        return utils_array

    def date_util(self, values, separator, shortyear, yearloc):
        util_dates = []
        if separator != "none":
            dayFirst = self.day_first_check(
                values, separator, shortYear=shortyear, yearLoc=yearloc
            )
        else:
            dayFirst = False

        for date in values:
            try:
                dateUtil = dateutil.parser.parse(str(date), dayfirst=dayFirst)
                if isinstance(dateUtil, datetime.date):
                    util_dates.append({"value": date, "standard": dateUtil})

            except Exception as e:
                logging.error(f"date_util - {date}: {e}")
        return util_dates, dayFirst

    def is_date_(self, date):
        #try to parse the date according to the format
        #if it parsed, return True, otherwise an exception will be raised
        datetime.datetime.strptime(date, self.format)
        return True 


    def is_date_arrow(self,date):
        dateArrow = arrow.get(str(date), normalize_whitespace=True).datetime

        if isinstance(dateArrow, datetime.date):
            return True

    def threshold_check(self, number_validated, number_of_samples, threshold, day_first):
        if number_validated >= number_of_samples * threshold:
            return build_return_date_object(format=self.return_label(), dayFirst=day_first)()
        raise

    # def validate_month_day(self, values):
    #     try:
    #         month_day_results = []
    #         for i, md in enumerate(values):
    #             try:
    #                 if str.isdigit(md):
    #                     if 12 >= int(md) >= 1:
    #                         month_day_results.append("month_day")
    #                     elif 12 < int(md) <= 31:
    #                         month_day_results.append("day")
    #                     else:
    #                         month_day_results.append("failed")
    #                 else:
    #                     logging.warning("Month_day test: Not a valid digit")
    #             except Exception as e:
    #                 logging.error(f"month_day_f - {md}: {e}")
    #
    #         if "failed" in month_day_results:
    #             return build_return_standard_object(
    #                 category=None, subcategory=None, match_type=None
    #             )
    #         elif "day" in month_day_results:
    #             return build_return_date_object(
    #                 format=self.day_ddOrd(values), dayFirst=None
    #             )
    #         elif "month_day" in month_day_results:
    #             return build_return_date_object(
    #                 format=self.month_MMorM(values), dayFirst=None
    #             )
    #         else:
    #             return build_return_standard_object(
    #                 category=None, subcategory=None, match_type=None
    #             )
    #     except Exception as e:
    #         return self.exception_category(e)

    # def validate_day_name(self, values):
    #     try:
    #         logging.info("Start day validation ...")
    #         day_array_valid = []
    #         for day in values:
    #             for d in self.days_of_the_week:
    #                 try:
    #                     day_array_valid.append(fuzzy_match(str(day), str(d), ratio=85))
    #                 except Exception as e:
    #                     logging.error(f"day_name_f - {d}: {e}")

    #         if np.count_nonzero(day_array_valid) >= (len(values) * 0.65):
    #             return build_return_date_object("%A", dayFirst=None)
    #         else:
    #             return build_return_standard_object(
    #                 category=None, subcategory=None, match_type=None
    #             )
    #     except Exception as e:
    #         return self.exception_category(e)

    def validate_month_name(self, values):
        try:
            logging.info("Start month validation ...")
            month_array_valid = []
            for month in values:
                for m in self.month_of_year:
                    try:
                        month_array_valid.append(
                            fuzzy_match(str(month), str(m), ratio=85)
                        )
                    except Exception as e:
                        logging.error(f"month_name_f - {m}: {e}")

            if np.count_nonzero(month_array_valid) >= (len(values) * 0.65):
                return build_return_date_object("%B", util=None, dayFirst=None)
            else:
                return self.validate_day_name(values)
        except Exception as e:
            return self.exception_category(e)
