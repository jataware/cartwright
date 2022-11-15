import pytest
from cartwright.categorize import CartwrightClassify


@pytest.mark.parametrize('filepath', [
    'examples/example_2.csv'
])
def test_simple_integration(filepath):
    t = CartwrightClassify()
    res = t.columns_classified(path=filepath)

#TODO: add tests that take in known CSVs, and check that the columns are correctly classified

if __name__ == '__main__':
    test_simple_integration('examples/example_2.csv')