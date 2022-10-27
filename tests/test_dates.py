from cartwright.categories import dates
from cartwright.categorize import CartwrightClassify

import pytest


import pdb

t = CartwrightClassify(20)
apple = dates.unix_time()
classes = t.all_classes



date_classes = [
    'date_%Y-%m-%d',
    'date_%Y_%m_%d',
    'date_%Y/%m/%d',
    'date_%Y.%m.%d',
    'date_%Y%m%d',
    'date_%Y-%m-%d %H:%M:%S',
    'date_%Y/%m/%d %H:%M:%S',
    'date_%Y_%m_%d %H:%M:%S',
    'date_%Y.%m.%d %H:%M:%S',
    'date_%d-%m-%Y',
    'date_%d/%m/%Y %H:%M:%S',
    'date_%d_%m_%Y %H:%M:%S',
    'date_%d.%m.%Y %H:%M:%S',
    'date_%d-%m-%y',
    'date_%d_%m_%Y',
    'date_%d_%m_%y',
    'date_%d/%m/%Y',
    'date_%d/%m/%y',
    'date_%d.%m.%Y',
    'date_%d.%m.%y',
    'date_%d-%m-%Y %H:%M:%S',
    'date_%A, %B %d, %y',
    '%A, %B %d, %Y HH:mm:ss',
    '%d %B %Y',
    '%d %B %y',
    '%B %d, %Y',
    '%m/%d/%y HH:mm:ss',
    'date_%m-%d-%Y',
    'date_%m/%d/%Y %H:%M:%S',
    'date_%m_%d_%Y %H:%M:%S',
    'date_%m.%d.%Y %H:%M:%S',
    'date_%m-%d-%y',
    'date_%m_%d_%Y',
    'date_%m_%d_%y',
    'date_%m/%d/%Y',
    'date_%m/%d/%y',
    'date_%m.%d.%Y',
    'date_%m.%d.%y',
    'date_%m-%d-%Y %H:%M:%S',
    'date_%Y%d',
    'date_%Y-%m',
    'date_%Y/%m',
    'date_%Y.%m',
    'date_%Y_%m',
    'date_%Y-%m-%dT%H%M%S',
    'unix_time',
]



@pytest.mark.parametrize('name', date_classes)
def test_generate_single_date(name, num_samples=1000):
    cls = t.all_classes[name]
    examples = [cls.generate_training_data() for _ in range(num_samples)]
    pdb.set_trace()
    cls.validate(example)


if __name__ == '__main__':
    test_generate_single_date(date_classes[0])

#TODO: move to other test files
# 'city'
# 'city_suffix'
# 'continent'
# 'country_GID'
# 'country_code'
# 'country_name'
# 'latitude'
# 'latlong'
# 'longitude'
# 'timespan_%Y-%Y'
# 'timespan_%Y:%Y'
# 'timespan_%B %d, %Y - %B %d, %Y'
# 'timespan_%d-%m-%Y:%d-%m-%Y'
# 'timespan_%d/%m/%Y:%d/%m/%Y'
# 'timespan_%d/%m/%Y-%d/%m/%Y'
# 'boolean'
# 'boolean_letter'
# 'email'
# 'first_name'
# 'language_name'
# 'paragraph'
# 'percent'
# 'phone_number'
# 'prefix'
# 'pyfloat'
# 'pystr'
# 'ssn'
# 'zipcode'
# 'day_of_month'
# 'day_of_week'
# 'month'
# 'month_name'
# 'year'