import pytest
from cartwright.categorize import CartwrightClassify


@pytest.mark.parametrize('filepath', ['examples/example_2.csv'])
def test_simple_integration(filepath):
    t = CartwrightClassify(20)
    res = t.columns_classified(filepath)