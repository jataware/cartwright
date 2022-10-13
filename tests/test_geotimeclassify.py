from geotime_classify import geotime_classify
import os
import numpy as np
import pdb

from geotime_classify.geotime_schema import TimeUnit, Uniformity
def testoncsv():
    t= geotime_classify.GeoTimeClassify(20)
    assert t.columns_classified(os.getcwd()+'/geotime_classify/datasets/Test_1.csv')

def test_time_resolution_in_pipeline():
    t = geotime_classify.GeoTimeClassify(20)
    res = t.columns_classified(os.getcwd()+'/examples/example_1.csv')
    time_res = res.classifications[0].time_resolution
    assert time_res.uniformity == Uniformity.NOT_UNIFORM
    assert time_res.unit == TimeUnit.year


def test_time_resolution_prediction(unit:TimeUnit, uniformity:Uniformity, num_rows=int(2e6), density=1.0):
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
    print(res)
    assert res.unit == unit, f'failed to detect {unit}, instead got {res.unit}'
    assert res.uniformity == uniformity, f'failed to detect {uniformity} uniformity for {unit}, instead got {res.uniformity}'
        

if __name__ == "__main__":
    # testoncsv()

    test_time_resolution_in_pipeline()

    for uniformity in Uniformity:
        for unit in [*TimeUnit][1:]: #skip milliseconds b/c precision isn't high enough
            test_time_resolution_prediction(unit, uniformity)

    print('All tests passed')
    