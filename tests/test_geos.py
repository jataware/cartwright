from cartwright.categorize import CartwrightClassify
from random import random
import pytest
from .utils import dedup_tests


import pdb

#single instance used for all tests
t = CartwrightClassify()

# #TODO: pull classes from the .py file itself
geo_classes = [
    'city',
    'city_suffix',
    'continent',
    'country_GID',
    'ISO2',
    'country_name',
    'latitude',
    'latlong',
    'longitude',
]



@pytest.mark.parametrize('name',
    dedup_tests([
        pytest.param('city', marks=pytest.mark.xfail(reason='city does not yet have a validate() method')),
        pytest.param('city_suffix', marks=pytest.mark.xfail(reason='city_suffix does not yet have a validate() method')),
        pytest.param('country_name', marks=pytest.mark.xfail(reason='Currently broken')),
        *geo_classes
    ])
)
def test_generate_single_geo(name, num_samples=1000):
    cls = t.all_classes[name]
    examples = [cls.generate_training_data() for _ in range(num_samples)]
    for label, value in examples:
        cls.validate(value)

# @pytest.mark.parametrize('name,ratio_valid', [
#     (date_class, 1.0) for date_class in geo_classes #TODO: for ratio in <some sequence>
# ])
# def test_generate_column(name, ratio_valid, num_samples=1000):
#     cls = t.all_classes[name]
#     examples = [cls.generate_training_data() if random() < ratio_valid else ('akjdsgjhdg','adhgjahgdj') for _ in range(num_samples)]
    
#     #TODO: call the looping validator
#     #TODO: categorize.assign_heuristic_function probably needs to be broken out so that the loop validation can be called directly without needing any of the other results/spreadsheet stuff/etc

# #def test_whole_pipeline_synthetic():
# #generate data, save to csv and then have whole pipeline read it

# #def test_whole_pipeline_real():
# #find real data and then have whole pipeline read it

if __name__ == '__main__':
    test_generate_single_geo(geo_classes[0])
