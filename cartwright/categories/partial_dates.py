import logging
from cartwright.utils import build_return_date_object, build_return_standard_object
from cartwright.categories.CategoryBases import DateBase

#'2022'
class year(DateBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        try:
            logging.info("Start year validation ...")
            year_values_valid = []

            for year in values:
                try:
                    if str.isdigit(str(year)):
                        if 1800 < int(year) < 2100:
                            year_values_valid.append("True")
                except Exception as e:
                    logging.error(f"year_f - {values}: {e}")

            if len(year_values_valid) > len(values) * 0.75:
                return build_return_date_object(format="%Y", dayFirst=None)
            return build_return_standard_object(category=None, subcategory=None, match_type=None)
        except Exception as e:
            return self.exception_category(e)
#'05'
class month(DateBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.validate_month_day(values)
#'May'
class month_name(DateBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.validate_month_name(values)

#'15'
class day_of_month(DateBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self,values):
        return self.validate_month_day(values)

#'Wednesday'
class day_of_week(DateBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self, values):
        return self.validate_month_day(values)
