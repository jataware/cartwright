import numpy as np
import fuzzywuzzy
from fuzzywuzzy import process

import random

from cartwright.CategoryBases import GeoBase
from cartwright.utils import *

#Sarahbury
class city(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate_series(self, series):
        subsample = 5
        count = 0
        passed = 0
        while passed < 2 and count <= subsample:
            count += 1
            match = fuzzywuzzy.process.extractOne(
                random.sample(series, 1),
                self.city_lookup,
                scorer=fuzz.token_sort_ratio,
            )
            if match is not None:
                if match[1] > 90:
                    passed += 1

        if passed>=2:
            return len(series)


#'port'
class city_suffix(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate_series(self, series):
        subsample = 5
        count = 0
        passed = 0
        while passed < 2 and count <= subsample:
            count += 1
            match = fuzzywuzzy.process.extractOne(
                random.sample(series, 1),
                self.city_lookup,
                scorer=fuzz.token_sort_ratio,
            )
            if match is not None:
                if match[1] > 90:
                    passed += 1

        if passed >= 2:
            return len(series)


# Luxembourg
class country_name(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, "country")())

    def validate_series(self, series):
        subsample = 5
        count = 0
        passed = 0
        while passed < 2 and count <= subsample:
            count += 1
            match = fuzzywuzzy.process.extractOne(
                random.sample(series, 1),
                self.country_name,
                scorer=fuzz.token_sort_ratio,
            )
            if match is not None:
                if match[1] > 90:
                    passed += 1

        if passed>=2:
            return len(series)


#TJ
class country_code(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self, value):
        return value.upper() in self.iso2_lookup


#'UZ'
class country_GID(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(np.random.choice(self.iso3_lookup))
    def validate(self, value):
        return value.upper() in self.iso3_lookup


#'Antarctica'
class continent(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        cont_type = np.random.choice(['cont_code','cont_name'])
        if cont_type == 'cont_code':
            val = np.random.choice(self.cont_codes)
        else:
            val = np.random.choice(self.cont_names)
        return self.class_name(), val

    def validate(self, value):

        for continent in self.cont_lookup:
            if fuzzy_match(str(value), str(continent), ratio_=int(100*self.threshold)):
                return True



#'8.166433'
class latitude(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,value):
        if 90 >= float(value) >= -90:
            return True


#'141.645223'
class longitude(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())
    def validate(self,value):
        if 180 >= float(value) >= -180:
            return True

#'74.2533, 179.643'
class latlong(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        remove_or_add_digits = np.random.choice([15, 15, 15, 13, -1, -2, -3])
        val = str(getattr(self.fake, "latitude")())[:remove_or_add_digits] + ", " + str(
            getattr(self.fake, "longitude")())[:remove_or_add_digits]
        return self.class_name(), val


    def validate(self,value):
        values=value.split(',')
        lat=values[0].strip()
        lng=values[1].strip()
        if 180 >= float(lng) >= -180:
            if 90 >= float(lat) >= -90:
                return True

