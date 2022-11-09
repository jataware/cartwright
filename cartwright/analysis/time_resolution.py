import numpy as np
from typing import Optional
from ..schemas import TimeUnit, Uniformity, Resolution
from .helpers import get_uniformity, match_unit


def detect_temporal_resolution(times:np.ndarray) -> Optional[Resolution]:
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

    #compute the average delta, and the average deviation
    avg = np.median(deltas)

    #compute the uniformity of the deltas
    uniformity = get_uniformity(deltas, avg)

    #find the closest matching unit for the average duration
    scale, unit = match_unit(TimeUnit, avg)
    error = np.abs((deltas - avg)).mean() / unit

    #return the results
    return Resolution(uniformity, unit, scale, error)
