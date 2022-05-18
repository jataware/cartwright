#script for training models to catch anomalies in spreadsheets not handled well by the system

NoneType = type(None)
from typing import Callable
import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt

import pdb

# ideas/tasks
#   create a simple image-like auto-encoder over spreadsheet space
#   web scrape for datasets to test if they would work or not. this is how to get negative examples
#   tbd


def main():
    path = '~/Downloads/messy_data_1.csv'
    # path = '~/Downloads/cleaned_data_2.csv'
    sheet = csv_to_img(path)

    pdb.set_trace()
    1

#WIP
# if path.endswith('.xlsx'):
#     sheet = pd.read_excel(path, *args, **kwargs)

def try_convert(cell):
    try:
        return float(cell)
    except Exception:
        pass
    #TODO: try to convert to a date using geotime functions...
    return cell

def ndmap(func: Callable, arr: np.ndarray, shape, dtype) -> np.ndarray:
    """apply a function to each element of an N-Dimensional array array"""    
    return np.array([*map(func, arr.flatten())], dtype=dtype).reshape(shape)

def csv_to_img(path: str, *args, **kwargs) -> pd.DataFrame:
    """read in a csv and convert the spreadsheet to a feature image for ML analysis"""

    if not path.endswith('.csv'):
        raise ValueError('expected a csv file')

    datatype_list = [str, float, NoneType] #TODO: add more types, e.g. date
    datatype_map = {t: i for i, t in enumerate(datatype_list)}


    sheet = pd.read_csv(path, header=None, keep_default_na=False, *args, **kwargs)
    sheet = np.array(sheet, dtype=object)
    shape = sheet.shape
    dtype = object
    
    #convert all strings that are understood to correct types (just floats for now)
    sheet = ndmap(try_convert, sheet, shape=shape, dtype=dtype)

    #replace empty strings with None
    sheet[sheet == ''] = None
    

    #convert array of types to array of datatype indices
    sheet = ndmap(lambda x: datatype_map[type(x)], sheet, shape=shape, dtype=int)
    plt.imshow(sheet); plt.show()
    
    #convert the datatype indices to one-hot vectors, creating channels for each data type
    sheet = ndmap(lambda x: np.eye(len(datatype_list))[x], sheet, shape=(*shape, len(datatype_list)), dtype=np.float32)

    #commute the dimensions so that the first dimension is the channel, followed by height, then width
    sheet = np.transpose(sheet, (2, 0, 1))

    return sheet


if __name__ == '__main__':
    main()