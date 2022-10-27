from cartwright.categorize import CartwrightClassify
from random import random
import pytest


import pdb

#single instance used for all tests
t = CartwrightClassify(20)


#TODO: pull classes from the .py file itself
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
    'date_%A, %B %d, %Y HH:mm:ss',
    'date_%d %B %Y',
    'date_%d %B %y',
    'date_%B %d, %Y',
    'date_%m/%d/%y HH:mm:ss',
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
    for label, value in examples:
        cls.validate(value)

@pytest.mark.parametrize('name,ratio_valid', [
    (date_class, 1.0) for date_class in date_classes #TODO: for ratio in <some sequence>
])
def test_generate_column(name, ratio_valid, num_samples=1000):
    cls = t.all_classes[name]
    examples = [cls.generate_training_data() if random() < ratio_valid else ('akjdsgjhdg','adhgjahgdj') for _ in range(num_samples)]
    
    #TODO: call the looping validator
    #TODO: categorize.assign_heuristic_function probably needs to be broken out so that the loop validation can be called directly without needing any of the other results/spreadsheet stuff/etc


# #def test_whole_pipeline_synthetic():
# #generate data, save to csv and then have whole pipeline read it

# #def test_whole_pipeline_real():
# #find real data and then have whole pipeline read it

if __name__ == '__main__':
    test_generate_single_date(date_classes[0])
