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
            date = date.strip()
            if datetime.datetime.strptime(date,"%B %d, %Y"):
                valid_count+=1
        if valid_count==len(dates):
            return True


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
            date = date.strip()
            if datetime.datetime.strptime(date,"%d-%m-%Y"):
                valid_count+=1
        if valid_count==len(dates):
            return True


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
            date = date.strip()
            if datetime.datetime.strptime(date,"%d/%m/%Y"):
                valid_count+=1
        if valid_count==len(dates):
            return True


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
            date = date.strip()
            if datetime.datetime.strptime(date,"%d/%m/%Y"):
                valid_count+=1
        if valid_count==len(dates):
            return True
