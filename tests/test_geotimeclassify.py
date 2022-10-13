from datetime import datetime
from geotime_classify import geotime_classify
import os
import numpy as np
import pandas as pd
import timeit
import time

import pdb

from geotime_classify.geotime_schema import TimeUnit, Uniformity

#disable logging for tests
import logging
logging.disable(logging.CRITICAL)

#seed the random number generator
np.random.seed(0)

def testoncsv():
    t= geotime_classify.GeoTimeClassify(20)
    assert t.columns_classified(os.getcwd()+'/geotime_classify/datasets/Test_1.csv')

def test_time_resolution_in_pipeline():
    t = geotime_classify.GeoTimeClassify(20)
    res = t.columns_classified(os.getcwd()+'/examples/example_1.csv')
    time_res = res.classifications[0].time_resolution
    assert time_res.uniformity == Uniformity.NOT_UNIFORM
    assert time_res.unit == TimeUnit.year


def time_resolution_algorithm(unit:TimeUnit, uniformity:Uniformity, num_rows=int(1e6)):

    #generate some fake data
    times = np.ones(num_rows,dtype=np.float64) * unit
    times = times.cumsum()
    times += np.random.randint(-377711962054, 379654882584) #10,000 BCE - 14,000 CE

    if uniformity == Uniformity.PERFECT:
        pass
    elif uniformity == Uniformity.UNIFORM:
        times += np.random.uniform(-0.0045,0.0045,num_rows)*unit
    elif uniformity == Uniformity.NOT_UNIFORM:
        times += np.random.uniform(-0.1,0.1,num_rows)*unit

    #run the test
    res = geotime_classify.time_resolution.detect_resolution(times)
    # print(res)
    assert res.unit == unit, f'failed to detect {unit}, instead got {res.unit}'
    assert res.uniformity == uniformity, f'failed to detect {uniformity} uniformity for {unit}, instead got {res.uniformity}'


def time_resolution_whole_pipeline(unit:TimeUnit, uniformity:Uniformity, num_rows=int(1e6)):
    #generate some fake data
    times = np.ones(num_rows,dtype=np.float64) * unit
    times = times.cumsum()
    times += np.random.randint(datetime(1000,1,1).timestamp(), datetime(3000,1,1).timestamp())

    #remove any times more than the maximum datetime (year 9999)
    times = times[times < datetime(9999,1,1).timestamp()]
    num_rows = len(times)

    if uniformity == Uniformity.PERFECT:
        pass
    elif uniformity == Uniformity.UNIFORM:
        times += np.random.uniform(-0.0045,0.0045,num_rows)*unit
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
    assert len(time_cols) == 1, f'expected 1 time column, got {len(time_cols)}'
    time_res = time_cols[0].time_resolution
    assert time_res is not None, 'time resolution information was not detected'


    assert time_res.unit == unit, f'failed to detect {unit}, instead got {time_res.unit}'
    assert time_res.uniformity == uniformity, f'failed to detect {uniformity} uniformity for {unit}, instead got {time_res.uniformity}'

    os.remove('test.csv')
    del df, res, t


def tests():
    yield testoncsv
    yield test_time_resolution_in_pipeline

    for uniformity in Uniformity:
        for unit in [*TimeUnit][1:]: #skip milliseconds b/c precision isn't high enough
            yield f'time_resolution_algorithm({unit},{uniformity})', lambda: time_resolution_algorithm(unit, uniformity)
    
    for uniformity in Uniformity:
        for unit in [*TimeUnit][1:]:
            yield f'time_resolution_whole_pipeline({unit},{uniformity})', lambda: time_resolution_whole_pipeline(unit, uniformity)


if __name__ == "__main__":

    for test in tests():
        if isinstance(test, tuple):
            name, test = test
        else:
            name = test.__name__
        try:
            t = timeit.Timer(test)
            print(f'PASSED {name}: {t.timeit(1):.04f} (seconds)')
        except Exception as e:
            print(f'FAILED {name}: {e}')
    time.sleep(1)

    print('Completed all tests')
    