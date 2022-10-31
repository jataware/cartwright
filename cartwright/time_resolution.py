import numpy as np
from typing import Optional


## when working locally use

from .schemas import TimeUnit, Uniformity, TimeResolution

# from cartwright.schemas import TimeUnit, Uniformity, TimeResolution
def detect_resolution(times:np.ndarray) -> Optional[TimeResolution]:
    """
    Detect the resolution of temporal data.
    
    @param times: a numpy array of unix times in [SECONDS] (may have duplicates)
    @return: (optional) TimeResolution(uniformity, unit_name, avg_density, avg_error) where 
        - uniformity is a Uniformity enum 
        - unit_name is a TimeUnits enum
        - avg_density is the median density of the data in (unit_name) units
        - avg_error is the mean error in (unit_name) units   
    """

    #get all the deltas between each (unique) time
    times = np.unique(times)
    if len(times) < 2:
        print('error: not enough timestamps to determine resolution')
        return None
    times.sort()
    deltas = times[1:] - times[:-1]

    #compute the average delta
    avg = np.median(deltas)

    #if all data within 1% of the mode, assume uniform
    uniformity_score = np.abs(deltas - avg)
    if np.all(uniformity_score < 1e-9 * avg):
        uniformity = Uniformity.PERFECT
    elif uniformity_score.max() < 0.01 * avg:
        uniformity = Uniformity.UNIFORM
    else:
        uniformity = Uniformity.NOT_UNIFORM

    #find the closest duration unit
    names = [*TimeUnit.__members__.keys()]
    durations = np.array([getattr(TimeUnit, name).value for name in names])
    unit_errors = np.abs(durations - avg)/durations
    closest = np.argmin(unit_errors)
    unit = getattr(TimeUnit, names[closest])
    errors = np.abs(1 - deltas / durations[closest]) #errors in terms of the closest unit

    #return the results
    return TimeResolution(uniformity, unit, avg/durations[closest], errors.mean())
