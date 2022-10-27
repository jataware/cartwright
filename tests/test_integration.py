import pytest
from cartwright.categorize import CartwrightClassify


@pytest.mark.parametrize('filepath', [
    pytest.param('examples/example_2.csv', marks=pytest.mark.xfail(reason='refactor has broken integration'))])
def test_simple_integration(filepath):
    t = CartwrightClassify(20)
    res = t.columns_classified(filepath)