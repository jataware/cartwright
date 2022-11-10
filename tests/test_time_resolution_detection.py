from datetime import datetime
from cartwright.analysis import time_resolution
from cartwright.categorize import CartwrightClassify
from cartwright.schemas import TimeUnit, Uniformity
from .utils import dedup_tests
import os
import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("unit,uniformity", 
    dedup_tests([
        pytest.param(TimeUnit.millisecond, Uniformity.PERFECT, marks=pytest.mark.xfail(reason="milliseconds has precision errors")),
        pytest.param(TimeUnit.millisecond, Uniformity.UNIFORM, marks=pytest.mark.xfail(reason="milliseconds has precision errors")),
        pytest.param(TimeUnit.millisecond, Uniformity.NOT_UNIFORM),
        *[(unit, uniformity) for uniformity in Uniformity for unit in TimeUnit]
    ])
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
    res = time_resolution.detect_temporal_resolution(times)
    assert res.unit == unit, f'failed to detect {unit}, instead got {res.unit}'
    assert res.uniformity == uniformity, f'failed to detect {uniformity} uniformity for {unit}, instead got {res.uniformity}'


@pytest.mark.parametrize("unit,uniformity", 
    dedup_tests([
        pytest.param(TimeUnit.millisecond, Uniformity.PERFECT, marks=pytest.mark.xfail(reason="milliseconds has precision errors")),
        pytest.param(TimeUnit.millisecond, Uniformity.UNIFORM, marks=pytest.mark.xfail(reason="milliseconds has precision errors")),
        pytest.param(TimeUnit.millisecond, Uniformity.NOT_UNIFORM, marks=pytest.mark.xfail(reason="seconds is detected here for some reason")),
        pytest.param(TimeUnit.second, Uniformity.UNIFORM, marks=pytest.mark.xfail(reason="for some reason, uniformity is detected as not uniform")),
        pytest.param(TimeUnit.minute, Uniformity.UNIFORM, marks=pytest.mark.xfail(reason="for some reason, uniformity is detected as not uniform")),
        # pytest.param(TimeUnit.hour, Uniformity.PERFECT, marks=pytest.mark.xfail(reason="time resolution information was not detected")),  #depends on the random seed!
        # pytest.param(TimeUnit.month, Uniformity.PERFECT, marks=pytest.mark.xfail(reason="time resolution information was not detected")), #depends on the random seed!
        *[(unit, uniformity) for uniformity in Uniformity for unit in TimeUnit],
    ])
)
def test_time_resolution_whole_pipeline(unit:TimeUnit, uniformity:Uniformity, num_rows=DEFAULT_NUM_ROWS):
    #generate some fake data
    times = np.ones(num_rows,dtype=np.float64) * unit
    times = times.cumsum()
    times += np.random.randint(datetime(1000,1,1).timestamp(), datetime(3000,1,1).timestamp(), dtype=np.int64)

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
    df.to_csv('test.csv',index=False, date_format='%m/%d/%Y %H:%M:%S')

    #run geotime
    t = CartwrightClassify()
    res = t.columns_classified(path='test.csv')
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



if __name__ == '__main__':
    test_time_resolution_whole_pipeline(TimeUnit.week, Uniformity.PERFECT)