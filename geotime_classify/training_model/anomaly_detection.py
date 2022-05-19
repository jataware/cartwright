#script for training models to catch anomalies in spreadsheets not handled well by the system

NoneType = type(None)
from datetime import datetime
import dateutil.parser as dateparser
from typing import Callable
import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
# import torchvision

import pdb

# ideas/tasks
#   create a simple image-like auto-encoder over spreadsheet space
#   web scrape for datasets to test if they would work or not. this is how to get negative examples
#   tbd


def main():
    path = '~/Downloads/messy_data_1.csv'
    # path = '~/Downloads/cleaned_data_1.csv'
    # path = '~/Downloads/messy_data_2.csv'
    # path = '~/Downloads/cleaned_data_2.csv'
    sheet = csv_to_img(path)
    # imshow(sheet)

    in_channels = sheet.shape[0]
    latent_channels = 8
    encoder = Encoder(in_channels, latent_channels)
    decoder = Decoder(latent_channels, in_channels)

    #attempt to encode and then decode the image
    x = sheet.unsqueeze(0)



    encoded = encoder(x)
    decoded = decoder(encoded)

    pdb.set_trace()




def toNone(s):
    """convert an empty string to None, or throw an exception"""
    if s == '':
        return None
    raise Exception('unexpected value')


datatype_converters = {
    NoneType: toNone,
    float: float,
    datetime: dateparser.parse,
    str: str, #fallback to string
}
datatype_list = [*datatype_converters.keys()]
datatype_map = {t: i for i, t in enumerate(datatype_list)}





def imshow(sheet):
    #if channel dimension is not 3, resize it to 3 (with interpolation)
    
    #move color channel last so that matplotlib understands it
    sheet = torch.transpose(sheet, (1, 2, 0))

    #resize color channel if more than 3
    if sheet.shape[0] != 3:
        sheet = F.interpolate(sheet, size=3, mode='linear', align_corners=False)

    plt.imshow(sheet); plt.show()

#WIP
# if path.endswith('.xlsx'):
#     sheet = pd.read_excel(path, *args, **kwargs)


def try_convert(cell):
    """Attempt to convert the string from a cell into a concrete type"""
    for t, converter in datatype_converters.items():
        try:
            return converter(cell)
        except:
            pass
    raise Exception(f'unable to convert cell [{cell}] to a valid data type')


def ndmap(func: Callable, arr: np.ndarray, shape, dtype) -> np.ndarray:
    """apply a function to each element of an N-Dimensional array array"""    
    return np.array([*map(func, arr.flatten())], dtype=dtype).reshape(shape)

def csv_to_img(path: str, *args, **kwargs) -> pd.DataFrame:
    """read in a csv and convert the spreadsheet to a feature image for ML analysis"""

    if not path.endswith('.csv'):
        raise ValueError('expected a csv file')

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
    # plt.imshow(sheet); plt.show()
    
    #convert the datatype indices to one-hot vectors, creating channels for each data type
    sheet = ndmap(lambda x: np.eye(len(datatype_list))[x], sheet, shape=(*shape, len(datatype_list)), dtype=np.float32)

    #commute the dimensions so that the first dimension is the channel, followed by height, then width
    sheet = np.transpose(sheet, (2, 0, 1))
    
    #convert the array to a tensor
    sheet = torch.tensor(sheet)

    #stretch to resize to a square 32x32 image
    sheet = sheet.unsqueeze(0)
    sheet = F.interpolate(sheet, size=(32, 32), mode='nearest')
    sheet = sheet.squeeze(0)

    return sheet


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super().__init__()

        #number of channels to operate most of the model on. final layer gets down to latent_channels
        working_channels = latent_channels * 2

        #embed the input to the specified number of channels
        self.embed = nn.Conv2d(in_channels=in_channels, out_channels=working_channels, kernel_size=3, padding=1)
        
        #5 convolutions blocks (BN, conv, relu, dropout) with stride of 2 to halve the image size down to 1x1
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(working_channels),
                nn.Conv2d(in_channels=working_channels, out_channels=working_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        for _ in range(5)])

        #embed to final number of channels
        self.final = nn.Linear(in_features=working_channels, out_features=latent_channels)

    def forward(self, x):
        x = self.embed(x)
        for conv in self.convs:
            x = conv(x)
        x = self.final(x.flatten(start_dim=1))
        return x
        

class Decoder(nn.Module):
    def __init__(self, latent_channels, out_channels):
        super().__init__()

        #number of channels to operate most of the model on. final layer gets down to latent_channels
        working_channels = latent_channels * 2

        #embed the input to the specified number of channels
        self.embed = nn.Linear(in_features=latent_channels, out_features=working_channels)

        #5 convolutions blocks (BN, conv, relu, dropout) with stride of 2 to halve the image size down to 1x1
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(working_channels) if _ != 0 else nn.Identity(),
                nn.ConvTranspose2d(in_channels=working_channels, out_channels=working_channels, kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        for _ in range(5)])

        #embed to final number of channels
        self.final = nn.ConvTranspose2d(in_channels=working_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.embed(x)
        x = x[..., None, None]
        for conv in self.convs:
            x = conv(x)
        x = self.final(x)
        return x


if __name__ == '__main__':
    main()