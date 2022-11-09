import pytest
from cartwright.categorize import CartwrightClassify


@pytest.mark.parametrize('filepath', [
    pytest.param('examples/example_2.csv', marks=pytest.mark.xfail(reason='Currently broken'))
])
def test_simple_integration(filepath):
    t = CartwrightClassify()
    res = t.columns_classified(filepath)


if __name__ == '__main__':
    test_simple_integration('examples/example_2.csv')