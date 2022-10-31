from cartwright.CategoryBases import TimespanBase
import datetime


#'2014-1989'
#'2014 - 1989'
class timespan_1(TimespanBase):
    def __init__(self):
        super().__init__()
        self.format="%Y-%Y"

    def generate_training_data(self):
        return self.format, str(self.fake.date(pattern="%Y") + self.space_seperator("-") + self.fake.date(pattern="%Y"))

    def validate(self, value):
        series=value.split('-')
        return self.validate_years(series)

    # def validate(self, values):
    #     try:
    #         year_array_valid = []
    #         for yearspan in values:
    #             try:
    #                 years = yearspan.split("-")
    #                 for year in years:
    #                     try:
    #                         if str.isdigit(str(year).strip()):
    #                             if 1800 < int(year) < 2100:
    #                                 year_array_valid.append("True")
    #
    #                     except Exception as e:
    #                         logging.error(f"{self.return_label()} error - {values}: {e}")
    #             except Exception as e:
    #                 logging.error(f"{self.return_label()} error: {e}")
    #         if np.count_nonzero(year_array_valid) >= (len(values) * 0.65) * 2:
    #             return build_return_timespan('%Y-%Y', dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         self.exception_category(e)


#'1974:1992'
# '1974 : 1992'
class timespan_2(TimespanBase):
    def __init__(self):
        super().__init__()
        self.format='%Y:%Y'

    def generate_training_data(self):
        return self.format, str(self.fake.date(pattern="%Y") + self.space_seperator(":") + self.fake.date(pattern="%Y"))

    def validate(self, value):
        series = value.split(':')
        return self.validate_years(series)
    # def validate(self, values):
    #     try:
    #         year_array_valid = []
    #         for yearspan in values:
    #             try:
    #                 years = yearspan.split(":")
    #                 for year in years:
    #                     try:
    #                         if str.isdigit(str(year).strip()):
    #                             if 1800 < int(year) < 2100:
    #                                 year_array_valid.append("True")
    #                     except Exception as e:
    #                         logging.error(f"{self.return_label()} - {values}: {e}")
    #             except Exception as e:
    #                 logging.error(f"{self.return_label()} error: {e}")
    #         if np.count_nonzero(year_array_valid) >= (len(values) * 0.65) * 2:
    #             return build_return_timespan('%Y:%Y', dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #
    #     except Exception as e:
    #         self.exception_category(e)


#'April 02, 1973 - June 21, 2015'
#'April 02, 1973-June 21, 2015'
class timespan_3(TimespanBase):
    def __init__(self):
        super().__init__()
        self.format='%B %d, %Y - %B %d, %Y'

    def generate_training_data(self):
        return self.format, str(self.fake.date(pattern="%B %d, %Y")+ self.space_seperator("-")+self.fake.date(pattern="%B %d, %Y"))

    def validate(self, value):
        dates=value.split('-')
        valid_count=0
        for date in dates:
            if datetime.datetime.strptime(date,"%B %d, %Y"):
                valid_count+=1
        if valid_count==len(dates):
            return True

    # def validate(self, values):
    #     try:
    #         array_valid=timespan_valid_array(values=values,separator="-", category=self.return_label())
    #         if len(array_valid) > (len(values) * 0.85) * 2:
    #             return build_return_timespan(format="%B %d, %Y - %B %d, %Y", dayFirst=False)
    #         return build_return_standard_object(category=None, subcategory=None,match_type=None)
    #     except Exception as e:
    #         self.exception_category(e)


#'22-08-2004:24-10-1993'
#'22-08-2004 : 24-10-1993'
class timespan_4(TimespanBase):
    def __init__(self):
        super().__init__()
        self.format='%d-%m-%Y:%d-%m-%Y'

    def generate_training_data(self):
        return self.format, str(self.fake.date(pattern="%d-%m-%Y")+ self.space_seperator(":") +self.fake.date(pattern="%d-%m-%Y"))

    def validate(self, value):
        dates=value.split(':')
        valid_count=0
        for date in dates:
            if datetime.datetime.strptime(date,"%d-%m-%Y"):
                valid_count+=1
        if valid_count==len(dates):
            return True
    # def validate(self, values):
    #     try:
    #         array_valid = timespan_valid_array(values=values, separator=":", category=self.return_label())
    #         if len(array_valid) > (len(values) * 0.85) * 2:
    #             return build_return_timespan(format="%d-%m-%Y:%d-%m-%Y", dayFirst=False)
    #         return build_return_standard_object(category=None, subcategory=None,match_type=None)
    #     except Exception as e:
    #         self.exception_category(e)

#'22-08-2004:24-10-1993'
#'22-08-2004 : 24-10-1993'
class timespan_5(TimespanBase):
    def __init__(self):
        super().__init__()
        self.format='%d/%m/%Y:%d/%m/%Y'

    def generate_training_data(self):
        return self.format, str(self.fake.date(pattern="%d/%m/%Y")+ self.space_seperator(":")+ self.fake.date(pattern="%d/%m/%Y"))

    def validate(self, value):
        dates=value.split(':')
        valid_count=0
        for date in dates:
            if datetime.datetime.strptime(date,"%d/%m/%Y"):
                valid_count+=1
        if valid_count==len(dates):
            return True
    # def validate(self, values):
    #     try:
    #         array_valid = timespan_valid_array(values=values, separator=":", category=self.return_label())
    #         if len(array_valid) > (len(values) * 0.85) * 2:
    #             return build_return_timespan(format="%d/%m/%Y:%d/%m/%Y", dayFirst=False)
    #         return build_return_standard_object(category=None, subcategory=None,match_type=None)
    #     except Exception as e:
    #         self.exception_category(e)

class timespan_6(TimespanBase):
    def __init__(self):
        super().__init__()
        self.format='%d/%m/%Y-%d/%m/%Y'

    def generate_training_data(self):
        return self.format, str(self.fake.date(pattern="%d/%m/%Y")+ self.space_seperator("-")+ self.fake.date(pattern="%d/%m/%Y"))

    def validate(self, value):
        dates=value.split('-')
        valid_count=0
        for date in dates:
            if datetime.datetime.strptime(date,"%d/%m/%Y"):
                valid_count+=1
        if valid_count==len(dates):
            return True
    # def validate(self,values):
    #     try:
    #         array_valid = timespan_valid_array(values=values, separator="-", category=self.return_label())
    #         if len(array_valid) > (len(values) * 0.85) * 2:
    #             return build_return_timespan(format="%d/%m/%Y-%d/%m/%Y", dayFirst=False)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         self.exception_category(e)



