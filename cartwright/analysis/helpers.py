import numpy as np
from enum import EnumMeta
from typing import Tuple, Type
from ..schemas import Unit, Uniformity


def get_uniformity(vals: np.ndarray, avg: float):
    uniformity_score = np.abs(vals - avg)
    avg_mag = np.abs(avg)
    if np.all(uniformity_score < 1e-9 * avg_mag):
        return Uniformity.PERFECT
    elif uniformity_score.max() < 0.01 * avg_mag:
        return Uniformity.UNIFORM
    else:
        return Uniformity.NOT_UNIFORM


def match_unit(cls:Type[Unit], avg:float) -> Tuple[float, Unit]:
    #find the closest matching unit
    names = [*cls.__members__.keys()]
    durations = np.array([getattr(cls, name).value for name in names], dtype=float)
    unit_errors = np.abs(durations - avg)/durations
    closest = np.argmin(unit_errors)
    unit = getattr(cls, names[closest])
    return avg/durations[closest], unit
