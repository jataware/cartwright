import numpy as np
import pandas as pd
import logging


# from cartwright.utils import
from cartwright.categories.CategoryBases import GeoBase
from cartwright.utils import *



#Sarahbury
class city(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self, values):
        return self.validate_city(values)

#'port'
class city_suffix(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self, values):
        return self.validate_country(values)


# Luxembourg
class country_name(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, "country")())

    def validate(self, values):
        return self.validate_country(values)


#TJ
class country_code(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.validate_iso2(values)

#'UZ'
class country_GID(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(np.random.choice(self.iso3_lookup))

    def validate(self, values):
        try:
            logging.info("Start iso3 validation ...")
            ISO_in_lookup = []

            for iso in values:
                for cc in self.iso3_lookup:
                    try:
                        ISO_in_lookup.append(
                            fuzzy_match(str(iso), str(cc), ratio=85)
                        )
                    except Exception as e:
                        logging.error(f"country_iso3 - {values}: {e}")

            if np.count_nonzero(ISO_in_lookup) >= (len(values) * 0.65):
                return build_return_standard_object(category='geo', subcategory='ISO3', match_type='LSTM')
            else:
                return self.validate_iso2(values)
        except Exception as e:
            logging.error(f'country_iso3 error: {e}')
            return build_return_standard_object(category=None, subcategory=None, match_type=None)


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

    def validate(self,values):
        try:
            logging.info("Start continent validation ...")
            cont_in_lookup = []

            for cont in values:
                for c in self.cont_lookup:
                    try:
                        cont_in_lookup.append(
                            fuzzy_match(str(cont), str(c), ratio=85)
                        )
                    except Exception as e:
                        logging.error(f"continent_f - {c} - {cont}: {e}")

            if np.count_nonzero(cont_in_lookup) >= (len(values) * 0.65):

                return build_return_standard_object(category='geo', subcategory='continent', match_type='LSTM')
            else:
                return build_return_standard_object(category=None, subcategory=None, match_type=None)
        except Exception as e:
            logging.error(f'continent error: {e}')
            return build_return_standard_object(category=None, subcategory=None, match_type=None)


#'8.166433'
class latitude(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.validate_geos(values)

#'141.645223'
class longitude(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.validate_geos(values)

#'74.2533, 179.643'
class latlong(GeoBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        remove_or_add_digits = np.random.choice([15, 15, 15, 13, -1, -2, -3])
        val = str(getattr(self.fake, "latitude")())[:remove_or_add_digits] + ", " + str(
            getattr(self.fake, "longitude")())[:remove_or_add_digits]
        return self.class_name(), val

    def validate(self, values):
        return self.not_classified()


