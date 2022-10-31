from cartwright.categorize import CartwrightClassify
from random import random
import pytest


import pdb

#single instance used for all tests
t = CartwrightClassify('0.0.0.1', 20)


#TODO: pull classes from the .py file itself
date_classes = [
    '%Y-%m-%d',
    '%Y_%m_%d',
    '%Y/%m/%d',
    '%Y.%m.%d',
    '%Y%m%d',
    '%Y-%m-%d %H:%M:%S',
    '%Y/%m/%d %H:%M:%S',
    '%Y_%m_%d %H:%M:%S',
    '%Y.%m.%d %H:%M:%S',
    '%d-%m-%Y',
    '%d/%m/%Y %H:%M:%S',
    '%d_%m_%Y %H:%M:%S',
    '%d.%m.%Y %H:%M:%S',
    '%d-%m-%y',
    '%d_%m_%Y',
    '%d_%m_%y',
    '%d/%m/%Y',
    '%d/%m/%y',
    '%d.%m.%Y',
    '%d.%m.%y',
    '%d-%m-%Y %H:%M:%S',
    '%A, %B %d, %Y',
    '%A, %B %d, %Y, %H:%M:%S',
    '%d %B %Y',
    '%d %B %y',
    '%B %d, %Y',
    '%m/%d/%Y %H:%M:%S',
    '%m-%d-%Y',
    '%m/%d/%Y %H:%M:%S',
    '%m_%d_%Y %H:%M:%S',
    '%m.%d.%Y %H:%M:%S',
    '%m-%d-%y',
    '%m_%d_%Y',
    '%m_%d_%y',
    '%m/%d/%Y',
    '%m/%d/%y',
    '%m.%d.%Y',
    '%m.%d.%y',
    '%m-%d-%Y %H:%M:%S',
    '%Y%d',
    '%Y-%m',
    '%Y/%m',
    '%Y.%m',
    '%Y_%m',
    '%Y-%m-%dT%H:%M:%S',
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
    # test_generate_single_date(date_classes[22])
    # test_generate_single_date('%A, %B %d, %Y')
    cls = t.all_classes['%Y%d']
    for i in range(100):
        print(cls.generate_training_data())
