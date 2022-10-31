import pytest
from cartwright.categorize import CartwrightClassify


@pytest.mark.parametrize('filepath', [pytest.param('examples/example_2.csv')])
def test_simple_integration(filepath):
    t = CartwrightClassify(20)
    res = t.columns_classified(filepath)


if __name__ == '__main__':
    test_simple_integration('examples/example_2.csv')