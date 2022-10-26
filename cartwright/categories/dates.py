import logging
from cartwright.utils import build_return_date_object, build_return_standard_object
from cartwright.categories.CategoryBases import DateBase

#'1073674421'
class unix_time(DateBase):
    def __init__(self):
        super().__init__()

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())

    def validate(self, value):
            # does the timestamp pass these conditions
            if int(value) < -5364601438 or int(value) > 4102506000 or len(value) >13:
                pass
            else:
                return value

    def threshold(self,number_validated, number_of_samples):
        if number_validated >= number_of_samples * self.threshold:
            return build_return_date_object(format='Unix Timestamp', dayFirst=None)
        raise


    #             return build_return_standard_object(category=None, subcategory=None, match_type=None)
    # def validate(self, values):
    #     try:
    #         array_valid = []
    #         for v in values:
    #             try:
    #                 if int(v) < -5364601438 or int(v) > 4102506000:
    #                     array_valid.append('failed')
    #                 elif len(v) <= 13:
    #                     array_valid.append('valid')
    #                 else:
    #                     array_valid.append('failed')
    #             except Exception as e:
    #                 array_valid.append('failed')
    #                 logging.error(f"date_util_22 - {v}: {e}")
    #
    #         if 'failed' in array_valid:
    #             return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #         else:
    #             return build_return_date_object(format='Unix Timestamp', dayFirst=None)
    #     except Exception as e:
    #         return self.exception_category(e)

## ymd
#'2020-08-10'
class date_Ymd_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y-%m-%d"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="-", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format="%Y-" + dayFormat + "-" + monthFormat, dayFirst=dayFirst)
    #             return build_return_date_object(format="%Y-" + monthFormat + "-" + dayFormat,dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'2020_06_29'
class date_Ymd_2(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y_%m_%d"
        self.day_first=False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)



    # def validate(self,values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="_", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if dayFirst:
    #             return build_return_date_object(format="%Y_" + dayFormat + "_" + monthFormat, dayFirst=dayFirst)
    #         return build_return_date_object(format="%Y_" + monthFormat + "_" + dayFormat, dayFirst=dayFirst)
    #     except Exception as e:
    #         return self.exception_category(e)

#'1987/11/11'
class date_Ymd_3(DateBase):
    def __init__(self):
        super().__init__()
        self.format = "date_%Y/%m/%d"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="/", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format="%Y/" + dayFormat + "/" + monthFormat, dayFirst=dayFirst)
    #             return build_return_date_object(format="%Y/" + monthFormat + "/" + dayFormat, dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'1989.11.21'
class date_Ymd_4(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y.%m.%d"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)



    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator=".", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format="%Y." + dayFormat + "." + monthFormat, dayFirst=dayFirst)
    #             return build_return_date_object(format="%Y." + monthFormat + "." + dayFormat, dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'19760605'
class date_Ymd_5(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y%m%d"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    def validate(self, value):
        return self.is_date_arrow(value)


    # def validate(self, values):
    #     try:
    #         array_valid = self.date_arrow(values, separator="none")
    #         if len(array_valid) > len(values) * 0.85:
    #             return build_return_date_object(format="%Y%m%d", util='arrow', dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'1980-01-12 17:02:57'
class date_Ymd_6(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y-%m-%d %H:%M:%S"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)



    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="-", shortyear=False, yearloc=0)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(
    #                     format="%Y-" + dayFormat + "-" + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #             return build_return_date_object(
    #                     format="%Y-" + monthFormat + "-" + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'2021/06/01 14:22:56'
class date_Ymd_7(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y/%m/%d %H:%M:%S"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="/", shortyear=False, yearloc=0)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(
    #                     format="%Y/" + dayFormat + "/" + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #             return build_return_date_object(
    #                     format="%Y/" + monthFormat + "/" + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #
    #     except Exception as e:
    #         return self.exception_category(e)


#'1985_11_21 18:19:24'
class date_Ymd_8(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y_%m_%d %H:%M:%S"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="_", shortyear=False, yearloc=0)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if dayFirst:
    #             return build_return_date_object(
    #                 format="%Y_" + dayFormat + "_" + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                 dayFirst=dayFirst)
    #         return build_return_date_object(
    #                 format="%Y_" + monthFormat + "_" + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                 dayFirst=dayFirst)
    #     except Exception as e:
    #         return self.exception_category(e)

#'1995.04.15 12:38:54'
class date_Ymd_9(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y.%m.%d %H:%M:%S"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self,values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator=".", shortyear=False, yearloc=0)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(
    #                     format="%Y." + dayFormat + "." + monthFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #             return build_return_date_object(
    #                     format="%Y." + monthFormat + "." + dayFormat + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

### mdy
#'03-10-2017'
class date_mdy_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m-%d-%Y"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="-", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + "-" + monthFormat + "-%Y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + "-" + dayFormat + "-%Y", dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'03-10-17'
class date_mdy_2(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m-%d-%y"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="-", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + "-" + monthFormat + "-%Y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + "-" + dayFormat + "-%Y", dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'03_10_2017'
class date_mdy_3(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m_%d_%Y"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="_", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if dayFirst:
    #             return build_return_date_object(format=dayFormat + "_" + monthFormat + "_%Y", dayFirst=dayFirst)
    #         return build_return_date_object(format=monthFormat + "_" + dayFormat + "_%Y", dayFirst=dayFirst)
    #     except Exception as e:
    #         return self.exception_category(e)

#'03_10_17'
class date_mdy_4(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m_%d_%y"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="_", shortyear=True, yearloc=2)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if dayFirst:
    #             return build_return_date_object(format=dayFormat + "_" + monthFormat + "_%y", dayFirst=dayFirst)
    #         return build_return_date_object(format=monthFormat + "_" + dayFormat + "_%y", dayFirst=dayFirst)
    #     except Exception as e:
    #         return self.exception_category(e)


#'03/10/2017'
class date_mdy_5(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m/%d/%Y"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="/", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + "/" + monthFormat + "/%Y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + "/" + dayFormat + "/%Y", dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'03/10/17'
class date_mdy_6(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m/%d/%y"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="/", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + "/" + monthFormat + "/%y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + "/" + dayFormat + "/%y", dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'03.10.2017'
class date_mdy_7(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m.%d.%Y"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator=".", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + "." + monthFormat + ".%Y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + "." + dayFormat + ".%Y", dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'03.10.17'
class date_mdy_8(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m.%d.%y"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator=".", shortyear=True, yearloc=2)
    #         dayFormat, monthFormat=self.month_day_format( values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + "." + monthFormat + ".%y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + "." + dayFormat + ".%y", dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'03-10-17 10:28:37'
class date_mdy_9(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m-%d-%Y %H:%M:%S"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="-", shortyear=False, yearloc=2)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(
    #                     format=dayFormat + "-" + monthFormat + '-%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #             return build_return_date_object(
    #                     format=monthFormat + "-" + dayFormat + '-%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'03/10/17 10:28:37'
class date_mdy_10(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%m/%d/%Y %H:%M:%S"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="/", shortyear=False, yearloc=2)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(
    #                     format=dayFormat + "/" + monthFormat + '/%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #             return build_return_date_object(
    #                     format=monthFormat + "/" + dayFormat + '/%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

# '03_10_17 10:28:37'
class date_mdy_11(DateBase):
    def __init__(self):
        super().__init__()
        self.format = "date_%m_%d_%Y %H:%M:%S"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="_", shortyear=False, yearloc=2)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if dayFirst:
    #             return build_return_date_object(
    #                 format=dayFormat + "_" + monthFormat + '_%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                 dayFirst=dayFirst)
    #         return build_return_date_object(
    #                 format=monthFormat + "_" + dayFormat + '_%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                 dayFirst=dayFirst)
    #     except Exception as e:
    #         return self.exception_category(e)

# '03.10.17 10:28:37'
class date_mdy_12(DateBase):
    def __init__(self):
        super().__init__()
        self.format = "date_%m.%d.%Y %H:%M:%S"
        self.day_first = False

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator=".", shortyear=False, yearloc=2)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(
    #                     format=dayFormat + "." + monthFormat + '.%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #             return build_return_date_object(
    #                     format=monthFormat + "." + dayFormat + '.%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#### dmy
#'28-02-1996'
class date_dmy_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d-%m-%Y"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="-", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + "-" + monthFormat + "-%Y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + "-" + dayFormat + "-%Y",dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'28-02-96'
class date_dmy_2(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d-%m-%y"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="-", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + "-" + monthFormat + "-%y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + "-" + dayFormat + "-%y",dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'28_02_1996'
class date_dmy_3(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d_%m_%Y"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="_", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         if dayFirst:
    #             return build_return_date_object(format=dayFormat + "_" + monthFormat + "_%Y", dayFirst=dayFirst)
    #         return build_return_date_object(format=monthFormat + "_" + dayFormat + "_%Y", dayFirst=dayFirst)
    #     except Exception as e:
    #         return self.exception_category(e)


#'28_02_96'
class date_dmy_4(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d_%m_%y"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="_", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         if dayFirst:
    #             return build_return_date_object(format=dayFormat + "_" + monthFormat + "_%y", dayFirst=dayFirst)
    #         return build_return_date_object(format=monthFormat + "_" + dayFormat + "_%y", dayFirst=dayFirst)
    #     except Exception as e:
    #         return self.exception_category(e)

#'28/02/1996'
class date_dmy_5(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d/%m/%Y"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="/", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + '/' + monthFormat + "/%Y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + '/' + dayFormat + "/%Y",dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'28/02/96'
class date_dmy_6(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d/%m/%y"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="/", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + '/' + monthFormat + "/%y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + '/' + dayFormat + "/%y",dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'28.02.1996'
class date_dmy_7(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d.%m.%Y"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator=".", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + '.' + monthFormat + ".%Y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + '.' + dayFormat + ".%Y", dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'28.02.96'
class date_dmy_8(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d.%m.%y"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator=".", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + '.' + monthFormat + ".%y", dayFirst=dayFirst)
    #             return build_return_date_object(format=monthFormat + '.' + dayFormat + ".%y", dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'28-02-1996 08:20:47'
class date_dmy_9(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d-%m-%Y %H:%M:%S"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="-", shortyear=False, yearloc=2)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(
    #                     format=dayFormat + "-" + monthFormat + '-%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #             return build_return_date_object(
    #                     format=monthFormat + "-" + dayFormat + '-%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'28/02/1996 08:20:47'
class date_dmy_10(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d/%m/%Y %H:%M:%S"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="/", shortyear=False, yearloc=2)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(
    #                     format=dayFormat + "/" + monthFormat + '/%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #             return build_return_date_object(
    #                     format=monthFormat + "/" + dayFormat + '/%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'28_02_1996 08:20:47'
class date_dmy_11(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d_%m_%Y %H:%M:%S"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="_", shortyear=False, yearloc=2)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if dayFirst:
    #             return build_return_date_object(
    #                 format=dayFormat + "_" + monthFormat + '_%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                 dayFirst=dayFirst)
    #         return build_return_date_object(
    #                 format=monthFormat + "_" + dayFormat + '_%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                 dayFirst=dayFirst)
    #     except Exception as e:
    #         return self.exception_category(e)


#'28.02.1996 08:20:47'
class date_dmy_12(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%d.%m.%Y %H:%M:%S"
        self.day_first = True

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator=".", shortyear=False, yearloc=2)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         hourFormat = self.hour_hOrH(values, separator=':', loc_hms=0)
    #         minFormat = self.minute_mOrM(values, separator=':', loc_hms=1)
    #         secFormat = self.second_sOrS(values, separator=':', loc_hms=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(
    #                     format=dayFormat + "." + monthFormat + '.%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #             return build_return_date_object(
    #                     format=monthFormat + "." + dayFormat + '.%Y' + ' ' + hourFormat + ':' + minFormat + ':' + secFormat,
    #                     dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#### Yd / Ym
class date_yd_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y%d"
        self.day_first=None

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    def validate(self, value):
        return self.is_date_arrow(value)

    # def validate(self, values):
    #     try:
    #         array_valid = self.date_arrow(values, separator="none")
    #         if len(array_valid) > len(values) * 0.85:
    #             return build_return_date_object(format="%Y%d", util='arrow', dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'2008-12'
class date_ym_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y-%m"
        self.day_first = None

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    def validate(self, value):
        year_month = value.split('-')
        monthval=year_month[1]
        yearval=year_month[0]
        if 12 >= int(monthval) >= 1:
            if str.isdigit(str(yearval)):
                if 1800 < int(yearval) < 2100:
                    return value


    # def validate(self, values):
    #     try:
    #         monthFormat = self.month_MMorM(values, separator='-', loc=1)
    #         allMonthVals = []
    #         for val in values:
    #             monthval = val.split('-')[1]
    #             allMonthVals.append(monthval)
    #         validMonth = self.validate_month_day(allMonthVals)
    #         if validMonth["subcategory"] == "date" and validMonth["format"] == "%m" or validMonth[
    #             "subcategory"] == 'date' and validMonth['format'] == "%-m":
    #             return build_return_date_object(format="%Y-" + monthFormat, util='arrow', dayFirst=None)
    #
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'2008/12'
class date_ym_2(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y/%m"

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    def validate(self, value):
        year_month = value.split('/')
        monthval=year_month[1]
        yearval=year_month[0]
        if self.is_date_arrow(value):
            if 12 >= int(monthval) >= 1:
                if str.isdigit(str(yearval)):
                    if 1800 < int(yearval) < 2100:
                        return value

    # def validate(self, values):
    #     try:
    #         array_valid = self.date_arrow(values, separator="/")
    #         monthFormat = self.month_MMorM(values, separator='/', loc=1)
    #         allMonthVals = []
    #         for val in values:
    #             monthval = val.split('/')[1]
    #             allMonthVals.append(monthval)
    #         validMonth = self.validate_month_day(allMonthVals)
    #         if len(array_valid) > len(values) * 0.85:
    #             return build_return_date_object(format="%Y/" + monthFormat, util='arrow', dayFirst=None)
    #         elif validMonth["subcategory"] == 'date' and validMonth['format'] == "%m" or validMonth[
    #             'subcategory'] == 'date' and validMonth['format'] == '%-m':
    #             return build_return_date_object(format="%Y/" + monthFormat, util='arrow', dayFirst=None)
    #         else:
    #             return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'2008.12'
class date_ym_3(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y.%m"

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    def validate(self, value):
        year_month = value.split('.')
        monthval=year_month[1]
        yearval=year_month[0]
        if self.is_date_arrow(value):
            if 12 >= int(monthval) >= 1:
                if str.isdigit(str(yearval)):
                    if 1800 < int(yearval) < 2100:
                        return value
    # def validate(self, values):
    #     try:
    #         array_valid = self.date_arrow(values, separator=".")
    #         monthFormat = self.month_MMorM(values, separator='.', loc=1)
    #         allMonthVals = []
    #         for val in values:
    #             try:
    #                 monthval = val.split('.')[1]
    #                 allMonthVals.append(monthval)
    #             except Exception as e:
    #                 logging.error(f"{self.return_label()} validate month - {val}: {e}")
    #         validMonth = self.validate_month_day(allMonthVals)
    #         if len(array_valid) > len(values) * 0.75 and validMonth['category'] is not None:
    #             return build_return_date_object(format="%Y." + monthFormat, util='arrow', dayFirst=None)
    #         elif validMonth['subcategory'] == 'date' and validMonth['format'] == '%m' or validMonth[
    #             'subcategory'] == 'date' and validMonth['format'] == '%-m':
    #             return build_return_date_object(format="%Y." + monthFormat, util='arrow', dayFirst=None)
    #         else:
    #             return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'2008.12'
class date_ym_4(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y_%m"

    def generate_training_data(self):
        return self.format, self.get_fake_date(self.format)

    def validate(self, value):
        year_month = value.split('_')
        monthval=year_month[1]
        yearval=year_month[0]
        if self.is_date_arrow(value):
            if 12 >= int(monthval) >= 1:
                if str.isdigit(str(yearval)):
                    if 1800 < int(yearval) < 2100:
                        return value

    # def validate(self, values):
    #     try:
    #         array_valid = self.date_arrow(values, separator="_")
    #         monthFormat = self.month_MMorM(values, separator='_', loc=1)
    #         allMonthVals = []
    #         for val in values:
    #             try:
    #                 monthval = val.split('_')[1]
    #                 allMonthVals.append(monthval)
    #             except Exception as e:
    #                 logging.error(f"date_arrow - {val}: {e}")
    #         validMonth = self.validate_month_day(allMonthVals)
    #         if len(array_valid) > len(values) * 0.75 and validMonth['category'] is not None:
    #             return build_return_date_object(format="%Y_" + monthFormat, util='arrow', dayFirst=None)
    #         elif validMonth['subcategory'] == 'date' and validMonth['format'] == '%m' or validMonth[
    #             'subcategory'] == 'date' and validMonth['format'] == '%-m':
    #             return build_return_date_object(format="%Y_" + monthFormat, util='arrow', dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#2001-05-02T16:40:06
class iso8601(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%Y-%m-%dT%H%M%S"
        self.day_first=False

    def generate_training_data(self):
        return self.class_name(), str(getattr(self.fake, self.class_name())())


    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="none", shortyear=False, yearloc=None)
    #         if len(array_valid) > len(values) * 0.85:
    #             return build_return_date_object(format="%Y-%m-%dT%H%M%S", dayFirst=dayFirst)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

## Long dates
# 'Thursday, November 23, 1999'
class date_long_dmdy(DateBase):
    def __init__(self):
        super().__init__()
        self.format="date_%A, %B %d, %y"
        self.day_first=True

    def generate_training_data(self):
        dayExample = str(getattr(self.fake, "day_of_month")())
        dayExample_name = str(getattr(self.fake, "day_of_week")())
        monthExample = str(getattr(self.fake, "month_name")())
        yearExample = str(getattr(self.fake, "year")())
        val = dayExample_name + ', ' + monthExample + ' ' + dayExample + ', ' + yearExample
        return self.class_name(), val

    # def validation(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="none", shortyear=False, yearloc=None)
    #         dayFormat = self.day_ddOrd(values, separator=' ', loc=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             return build_return_date_object(format="%A, %B " + dayFormat + ",%y", util=None, dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'November 23, 1999'
class date_long_mdy(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%B %d, %Y"

    def generate_training_data(self):
        dayExample = str(getattr(self.fake, "day_of_month")())
        monthExample = str(getattr(self.fake, "month_name")())
        yearExample = str(getattr(self.fake, "year")())
        val = monthExample + ' ' + dayExample + ', ' + yearExample
        return self.class_name(), val

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="none", shortyear=False, yearloc=None)
    #         dayFormat = self.day_ddOrd(values, separator=' ', loc=1)
    #         if len(array_valid) > len(values) * 0.85:
    #             return build_return_date_object(format="%B " + dayFormat + ", %Y", dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'Monday, November 03, 1999, 18:46:22'
class date_long_dmdyt(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%A, %B %d, %Y HH:mm:ss"

    def generate_training_data(self):
        dayExample = str(getattr(self.fake, "day_of_month")())
        dayExample_name = str(getattr(self.fake, "day_of_week")())
        monthExample = str(getattr(self.fake, "month_name")())
        yearExample = str(getattr(self.fake, "year")())
        timeDate = str(getattr(self.fake, "date_time_this_century")())
        time = timeDate.split(' ')[1]
        val = dayExample_name + ', ' + monthExample + ' ' + dayExample + ', ' + yearExample + ', ' + time
        return self.class_name(), val

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="none", shortyear=False, yearloc=None)
    #         dayFormat = self.day_ddOrd(values, separator=' ', loc=2)
    #         if len(array_valid) > len(values) * 0.85:
    #             return build_return_date_object(format="%A, %B " + dayFormat + ",%y HH:mm:ss", dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'11/03/99 18:46:22 PM'
class date_long_mdyt_m(DateBase):
    def __init__(self):
        super().__init__()
        self.format='%m/%d/%y HH:mm:ss'
        self.day_first=False

    def generate_training_data(self):
        dateExample = str(self.fake.date(pattern='%m/%d/%y'))
        timeDate =  str(getattr(self.fake, "date_time_this_century")())
        time = timeDate.split(' ')[1]
        ampm = str(self.fake.am_pm())
        val = dateExample + ' ' + time + ' ' + ampm

        return self.class_name(), val

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="none", shortyear=False, yearloc=None)
    #         dayFormat, monthFormat = self.month_day_format(values)
    #         if len(array_valid) > len(values) * 0.85:
    #             if dayFirst:
    #                 return build_return_date_object(format=dayFormat + "/" + monthFormat + "/%y HH:mm", dayFirst=None)
    #             return build_return_date_object(format=monthFormat + "/" + dayFormat + "/%y HH:mm", dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)


#'03 November 1999'
class date_long_dmonthY(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d %B %Y"
        self.day_first=True

    def generate_training_data(self):
        dayExample = self.fake.day_of_month()
        monthExample = self.fake.month_name()
        yearExample = self.fake.year()
        val = dayExample + ' ' + monthExample + ' ' + yearExample
        return self.class_name(), val

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="none", shortyear=False, yearloc=None)
    #         dayFormat = self.day_ddOrd(values, separator=' ', loc=0)
    #         if len(array_valid) > len(values) * 0.85:
    #             return build_return_date_object(format=dayFormat + " %B %Y", dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)

#'03 November 99'
class date_long_dmonthy(DateBase):
    def __init__(self):
        super().__init__()
        self.format = "%d %B %y"
        self.day_first = True

    def generate_training_data(self):
        dayExample = self.fake.day_of_month()
        monthExample = self.fake.month_name()
        yearExample = self.fake.year()
        val = dayExample + ' ' + monthExample + ' ' + yearExample[:2]
        return self.class_name(), val

    # def validate(self, values):
    #     try:
    #         array_valid, dayFirst = self.date_util(values, separator="none", shortyear=False, yearloc=None)
    #         dayFormat = self.day_ddOrd(values, separator=' ', loc=0)
    #         if len(array_valid) > len(values) * 0.85:
    #             return build_return_date_object(format=dayFormat + " %B %y",  dayFirst=None)
    #         return build_return_standard_object(category=None, subcategory=None, match_type=None)
    #     except Exception as e:
    #         return self.exception_category(e)
