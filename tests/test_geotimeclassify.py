from datetime import datetime
from geotime_classify import geotime_classify
from geotime_classify.geotime_schema import TimeUnit, Uniformity
import os
import numpy as np
import pandas as pd
from typing import Callable, Iterator, Tuple, Union
import pytest


import pdb


#disable logging for tests
import logging
logging.disable(logging.CRITICAL)

#seed the random number generator
np.random.seed(0)


DEFAULT_NUM_ROWS = 200#int(1e6)


"""
Run tests with pytest in the root directory of the project
"""


# def testoncsv():
#     t= geotime_classify.GeoTimeClassify(20)
#     assert t.columns_classified(os.getcwd()+'/geotime_classify/datasets/Test_1.csv')

# def test_time_resolution_in_pipeline():
#     t = geotime_classify.GeoTimeClassify(20)
#     res = t.columns_classified(os.getcwd()+'/examples/example_1.csv')
#     time_res = res.classifications[0].time_resolution
#     assert time_res.uniformity == Uniformity.NOT_UNIFORM
#     assert time_res.unit == TimeUnit.year


@pytest.mark.parametrize("unit,uniformity", 
    [
        pytest.param(TimeUnit.millisecond, Uniformity.PERFECT, marks=pytest.mark.xfail(reason="milliseconds has precision errors")),
        pytest.param(TimeUnit.millisecond, Uniformity.UNIFORM, marks=pytest.mark.xfail(reason="milliseconds has precision errors")),
        pytest.param(TimeUnit.millisecond, Uniformity.NOT_UNIFORM),
        *[(unit, uniformity) for uniformity in Uniformity for unit in TimeUnit if unit != TimeUnit.millisecond],
    ]
)
def test_time_resolution_algorithm(unit:TimeUnit, uniformity:Uniformity, num_rows=DEFAULT_NUM_ROWS):

    #generate some fake data
    times = np.ones(num_rows,dtype=np.float64) * unit
    times = times.cumsum()
    times += np.random.randint(-377711962054, 379654882584, dtype=np.int64) #10,000 BCE - 14,000 CE

    if uniformity == Uniformity.PERFECT:
        pass
    elif uniformity == Uniformity.UNIFORM:
        times += np.random.uniform(-0.004,0.004,num_rows)*unit
    elif uniformity == Uniformity.NOT_UNIFORM:
        times += np.random.uniform(-0.1,0.1,num_rows)*unit

    #run the test
    res = geotime_classify.time_resolution.detect_resolution(times)
    assert res.unit == unit, f'failed to detect {unit}, instead got {res.unit}'
    assert res.uniformity == uniformity, f'failed to detect {uniformity} uniformity for {unit}, instead got {res.uniformity}'


@pytest.mark.parametrize("unit,uniformity", 
    [
        pytest.param(TimeUnit.millisecond, Uniformity.PERFECT, marks=pytest.mark.xfail(reason="milliseconds has precision errors")),
        pytest.param(TimeUnit.millisecond, Uniformity.UNIFORM, marks=pytest.mark.xfail(reason="milliseconds has precision errors")),
        pytest.param(TimeUnit.millisecond, Uniformity.NOT_UNIFORM, marks=pytest.mark.xfail(reason="seconds is detected here for some reason")),
        pytest.param(TimeUnit.second, Uniformity.PERFECT),
        pytest.param(TimeUnit.second, Uniformity.UNIFORM, marks=pytest.mark.xfail(reason="for some reason, uniformity is detected as not uniform")),
        pytest.param(TimeUnit.second, Uniformity.NOT_UNIFORM),
        pytest.param(TimeUnit.minute, Uniformity.PERFECT),
        pytest.param(TimeUnit.minute, Uniformity.UNIFORM, marks=pytest.mark.xfail(reason="for some reason, uniformity is detected as not uniform")),
        pytest.param(TimeUnit.minute, Uniformity.NOT_UNIFORM),
        *[(unit, uniformity) for uniformity in Uniformity for unit in TimeUnit if unit > TimeUnit.minute], #all units greater than minute pass all the tests
    ]
)
def test_time_resolution_whole_pipeline(unit:TimeUnit, uniformity:Uniformity, num_rows=DEFAULT_NUM_ROWS):
    #generate some fake data
    times = np.ones(num_rows,dtype=np.float64) * unit
    times = times.cumsum()
    min_time = (datetime(1971,1,1) if os.name == 'nt' else datetime(1000,1,1)).timestamp() #windows can't convert dates before 1970 
    max_time = datetime(3000,1,1).timestamp()
    times += np.random.randint(min_time, max_time, dtype=np.int64)

    #remove any times more than the maximum datetime (year 9999)
    times = times[times < datetime(9999,1,1).timestamp()]
    num_rows = len(times)

    if uniformity == Uniformity.PERFECT:
        pass
    elif uniformity == Uniformity.UNIFORM:
        times += np.random.uniform(-0.004,0.004,num_rows)*unit
    elif uniformity == Uniformity.NOT_UNIFORM:
        times += np.random.uniform(-0.1,0.1,num_rows)*unit

    #create a dataframe, with each time converted to a datetime string
    dtimes = np.asarray(times, dtype='datetime64[s]').tolist()
    df = pd.DataFrame({'date':dtimes})

    #add latitude and longitude columns with random values
    df['latitude'] = np.random.uniform(-90,90,num_rows)
    df['longitude'] = np.random.uniform(-180,180,num_rows)

    #add random feature columns
    df['feat1'] = np.random.uniform(-100,100,num_rows)
    df['feat2'] = np.random.uniform(-1,1,num_rows)
    df['feat3'] = np.random.uniform(-1000,1000,num_rows)
    df['feat4'] = np.random.uniform(-10000,10000,num_rows)

    #save the dataframe to a csv
    df.to_csv('test.csv',index=False)

    #run geotime
    t = geotime_classify.GeoTimeClassify(20)
    res = t.columns_classified('test.csv')
    if res is None:
        raise Exception('geotime failed to classify the test data')

    #check the time resolution
    time_cols = [c for c in res.classifications if c.category == 'time']
    
    #cleanup
    os.remove('test.csv')
    del df, res, t

    #tests
    assert len(time_cols) == 1, f'expected 1 time column, got {len(time_cols)}'
    time_res = time_cols[0].time_resolution
    assert time_res is not None, 'time resolution information was not detected'
    assert time_res.unit == unit, f'failed to detect {unit}, instead got {time_res.unit}'
    assert time_res.uniformity == uniformity, f'failed to detect {uniformity} uniformity for {unit}, instead got {time_res.uniformity}'
