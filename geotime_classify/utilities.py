import numpy as np
from typing import Optional
from geotime_classify.geotime_schema import TimeUnits, Uniformity, TimeResolution

def detect_resolution(times:np.ndarray) -> Optional[TimeResolution]:
    """
    Detect the resolution of temporal data.
    
    @param times: a numpy array of unix times in SECONDS (may have duplicates)
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
    # avg = stats.mode(deltas)[0][0]
    # avg = np.mean(deltas)
    avg = np.median(deltas)

    #if all data within 1% of the mode, assume uniform
    uniformity_score = np.abs(deltas - avg)
    if np.all(uniformity_score == 0.0):
        print('perfectly uniform')
        uniformity = Uniformity.PERFECT
    elif uniformity_score.max() < 0.01 * avg:
        print('uniform to within 1%')
        uniformity = Uniformity.UNIFORM
    else:
        print('not uniform')
        uniformity = Uniformity.NOT_UNIFORM

    #find the closest duration unit
    names = [*TimeUnits.__members__.keys()]
    durations = np.array([getattr(TimeUnits, name).value for name in names])
    unit_errors = np.abs(durations - avg)/durations
    closest = np.argmin(unit_errors)
    unit = getattr(TimeUnits, names[closest])
    errors = np.abs(1 - deltas / durations[closest]) #errors in terms of the closest unit

    #TODO: could do some sort of thresholding to determine if it matched any of the durations or was something else
    # print(f'`{names[closest]}` matches average delta with {unit_errors[closest]*100:.2f}% error')
    # print(f'all deltas ({names[closest]}s):')
    # print(f'{deltas/durations[closest]}')
    # print('=========================================')

    return TimeResolution(uniformity, unit, avg/durations[closest], errors.mean())
