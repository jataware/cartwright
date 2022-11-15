import numpy as np
from typing import Optional, List
from datetime import datetime, timezone
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



def convert_to_timestamps(times:List[str], fmt:str) -> np.ndarray:
    """
    Convert a list of strings to unix timestamps.

    @param times: a list of strings representing times
    @param format: a string representing the format of the times

    @return: a numpy array of unix timestamps in [SECONDS]

    @example:
    ```
    times = ['2019-01-01 00:00:00', '2019-01-01 00:00:01', '2019-01-01 00:00:02']
    fmt = '%Y-%m-%d %H:%M:%S'
    convert_to_timestamps(times, fmt)
    ```
    """
    times = [
        datetime.strptime(str(time_), fmt)
            .replace(tzinfo=timezone.utc)
            .timestamp()
        for time_ in times
    ]
    times = np.array(times)

    return times