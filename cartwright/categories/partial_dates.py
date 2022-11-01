from cartwright.utils import months_of_the_year_B, months_of_the_year_b, days_of_the_week_A, days_of_the_week_a
from cartwright.CategoryBases import DateBase
import numpy as np
#'2022'
class year(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%Y"
        self.threshold=.99

    def generate_training_data(self):
        return self.format, str(getattr(self.fake, self.class_name())())


# #'05'
class month(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%m"
        self.threshold = .99

    def generate_training_data(self):
        return self.format, str(getattr(self.fake, self.class_name())())


#'May'
class month_name(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%B"

    def generate_training_data(self):
        return self.format, str(getattr(self.fake, self.class_name())())


class month_name_short(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%b"

    def generate_training_data(self):
        return self.format, np.random.choice(months_of_the_year_b)

    #TODO: some disagreement between this and datetime.strptime
    def validate(self,value):
        return value.lower() in months_of_the_year_b


#'15'
class day_of_month(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%d"
        self.threshold = .99

    def generate_training_data(self):
        return self.format, str(getattr(self.fake, self.class_name())())


#'Wednesday'
class day_of_week(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%A"

    def generate_training_data(self):
        return self.format, str(getattr(self.fake, self.class_name())())


#'Wed'
class day_of_week_a(DateBase):
    def __init__(self):
        super().__init__()
        self.format="%a"

    def generate_training_data(self):
        return self.format, np.random.choice(days_of_the_week_a)

    #TODO: some disagreement between this and datetime.strptime
    def validate(self,value):
        return value.lower() in days_of_the_week_a
