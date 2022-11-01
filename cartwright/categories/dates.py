from cartwright.CategoryBases import DateBase

#'1073674421'
class unix_time(DateBase):
    def __init__(self):
        super().__init__()
        self.format="unix_time"
    
    def generate_training_data(self):
        return self.format, str(getattr(self.fake, self.class_name())())

    def validate(self, value):
            # does the timestamp pass these conditions
            if int(value) < -5364601438 or int(value) > 4102506000 or len(value) >13:
                pass
            else:
                return value


## ymd
#'2020-08-10'
class date_Ymd_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y-%m-%d"



#'2020_06_29'
class date_Ymd_2(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y_%m_%d"



#'1987/11/11'
class date_Ymd_3(DateBase):
    def __init__(self):
        super().__init__()
        self.format = "%Y/%m/%d"


#'1989.11.21'
class date_Ymd_4(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y.%m.%d"




#'19760605'
class date_Ymd_5(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y%m%d"



#'1980-01-12 17:02:57'
class date_Ymd_6(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y-%m-%d %H:%M:%S"




#'2021/06/01 14:22:56'
class date_Ymd_7(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y/%m/%d %H:%M:%S"



#'1985_11_21 18:19:24'
class date_Ymd_8(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y_%m_%d %H:%M:%S"




#'1995.04.15 12:38:54'
class date_Ymd_9(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y.%m.%d %H:%M:%S"



### mdy
#'03-10-2017'
class date_mdy_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m-%d-%Y"



#'03-10-17'
class date_mdy_2(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m-%d-%y"



#'03_10_2017'
class date_mdy_3(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m_%d_%Y"



#'03_10_17'
class date_mdy_4(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m_%d_%y"



#'03/10/2017'
class date_mdy_5(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m/%d/%Y"



#'03/10/17'
class date_mdy_6(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m/%d/%y"



#'03.10.2017'
class date_mdy_7(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m.%d.%Y"


#'03.10.17'
class date_mdy_8(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m.%d.%y"



#'03-10-17 10:28:37'
class date_mdy_9(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m-%d-%Y %H:%M:%S"



#'03/10/17 10:28:37'
class date_mdy_10(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m/%d/%Y %H:%M:%S"




# '03_10_17 10:28:37'
class date_mdy_11(DateBase):
    def __init__(self):
        super().__init__()
        self.format = "%m_%d_%Y %H:%M:%S"




# '03.10.17 10:28:37'
class date_mdy_12(DateBase):
    def __init__(self):
        super().__init__()
        self.format = "%m.%d.%Y %H:%M:%S"



#### dmy
#'28-02-1996'
class date_dmy_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d-%m-%Y"


#'28-02-96'
class date_dmy_2(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d-%m-%y"


#'28_02_1996'
class date_dmy_3(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d_%m_%Y"


#'28_02_96'
class date_dmy_4(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d_%m_%y"


#'28/02/1996'
class date_dmy_5(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d/%m/%Y"


#'28/02/96'
class date_dmy_6(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d/%m/%y"


#'28.02.1996'
class date_dmy_7(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d.%m.%Y"


#'28.02.96'
class date_dmy_8(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d.%m.%y"


#'28-02-1996 08:20:47'
class date_dmy_9(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d-%m-%Y %H:%M:%S"


#'28/02/1996 08:20:47'
class date_dmy_10(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d/%m/%Y %H:%M:%S"


#'28_02_1996 08:20:47'
class date_dmy_11(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d_%m_%Y %H:%M:%S"


#'28.02.1996 08:20:47'
class date_dmy_12(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d.%m.%Y %H:%M:%S"


# 200831 or 197520 or 202208
class date_yd_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y%d"


#'2008-12'
class date_ym_1(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y-%m"


#'2008/12'
class date_ym_2(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y/%m"


#'2008.12'
class date_ym_3(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y.%m"


#'2008_12'
class date_ym_4(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y_%m"


#2001-05-02T16:40:06
class iso8601(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y-%m-%dT%H:%M:%S"

    def generate_training_data(self):
        return self.format, str(getattr(self.fake, self.class_name())())


## Long dates
# 'Thursday, November 23, 1999'
class date_long_dmdy(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%A, %B %d, %Y"

    def generate_training_data(self):
        #TODO: while loop because this can generate invalid dates e.g. Saturday, February 29, 1989
        while True:
            dayExample = str(getattr(self.fake, "day_of_month")())
            dayExample_name = str(getattr(self.fake, "day_of_week")())
            monthExample = str(getattr(self.fake, "month_name")())
            yearExample = str(getattr(self.fake, "year")())
            val = dayExample_name + ', ' + monthExample + ' ' + dayExample + ', ' + yearExample
            try:
                self.validate(val)
                return self.format, val
            except:
                pass

        # return self.format, val



#'November 23, 1999'
class date_long_mdy(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%B %d, %Y"

    def generate_training_data(self):
        while True: #TODO: loop until valid date is generated
            dayExample = str(getattr(self.fake, "day_of_month")())
            monthExample = str(getattr(self.fake, "month_name")())
            yearExample = str(getattr(self.fake, "year")())
            val = monthExample + ' ' + dayExample + ', ' + yearExample
            try:
                self.validate(val)
                return self.format, val
            except:
                pass

        # return self.format, val


#'Monday, November 03, 1999, 18:46:22'
class date_long_dmdyt(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%A, %B %d, %Y, %H:%M:%S"

    def generate_training_data(self):
        #TODO: loop to skip any invalid dates created
        while True:
            dayExample = str(getattr(self.fake, "day_of_month")())
            dayExample_name = str(getattr(self.fake, "day_of_week")())
            monthExample = str(getattr(self.fake, "month_name")())
            yearExample = str(getattr(self.fake, "year")())
            timeDate = str(getattr(self.fake, "date_time_this_century")())
            time = timeDate.split(' ')[1]
            val = dayExample_name + ', ' + monthExample + ' ' + dayExample + ', ' + yearExample + ', ' + time
            try:
                self.validate(val)
                return self.format, val
            except:
                pass

        # return self.format, val


#'11/03/99 18:46:22 PM'
class date_long_mdyt_m(DateBase):
    def __init__(self):
        super().__init__()
        self.format='%m/%d/%y %H:%M:%S %p'

    def generate_training_data(self):
        dateExample = str(self.fake.date(pattern='%m/%d/%y'))
        timeDate =  str(getattr(self.fake, "date_time_this_century")())
        time = timeDate.split(' ')[1]
        ampm = str(self.fake.am_pm())
        val = dateExample + ' ' + time + ' ' + ampm
        return self.format, val



#'03 November 1999'
class date_long_dmonthY(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d %B %Y"

    def generate_training_data(self):
        while True: #TODO: loop to skip any invalid dates created
            dayExample = self.fake.day_of_month()
            monthExample = self.fake.month_name()
            yearExample = self.fake.year()
            val = dayExample + ' ' + monthExample + ' ' + yearExample
            try:
                self.validate(val)
                return self.format, val
            except:
                pass

        # return self.format, val


#'Sat, 31 Oct 1981'
class date_long_a_d_month_Y(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%a, %d %b %Y"



class date_long_dmonthy(DateBase):
    def __init__(self):
        super().__init__()
        self.format = "%d %B %y"

    def generate_training_data(self):
        while True: #TODO: loop to skip any invalid dates created
            dayExample = self.fake.day_of_month()
            monthExample = self.fake.month_name()
            yearExample = self.fake.year()
            val = dayExample + ' ' + monthExample + ' ' + yearExample[:2]
            try:
                self.validate(val)
                return self.format, val
            except:
                pass

        # return self.format, val

