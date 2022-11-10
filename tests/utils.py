import pytest
from typing import Tuple, Union, List

Param = type(pytest.param()) #TODO: figure out where to import this from


def dedup_tests(tests:List[Union[Tuple, Param]]) -> List[Union[Tuple, Param]]:
    """
        Given a list of tests (either tuples or pytest.param), remove duplicates.
        useful for specifying tests that fail in a parametrized test, without having to also specify the passing tests.

        e.g.

        ```
        @pytest.mark.parametrize("v0, v1", dedup_tests([
            pytest.param(0, 1, marks=pytest.mark.xfail(reason='this test fails')),
            pytest.param(0, 2, marks=pytest.mark.xfail(reason='this test fails')),
            pytest.param(2, 5, marks=pytest.mark.xfail(reason='this test fails')),
            pytest.param(5, 5, marks=pytest.mark.xfail(reason='this test fails')),
            *[(v0, v1) for v0 in range(10) for v1 in range(10)]
        ]))
        def test_foo(v0, v1):
            ...
        ```

        dedup_tests will remove the test cases from the generated list that already occurred in the list of tests that failed.
    """
    seen = set()
    deduped = []
    for test in tests:
        # assert type(test) in (tuple, Param), f'expected test to be a tuple or pytest.param, but got {type(test)}'
        if type(test) is tuple:
            values = test
        elif type(test) is Param:
            values = test.values
        else:
            values = (test,)
        if values in seen:
            continue
        seen.add(values)
        deduped.append(test)

    return deduped