#script for training models to catch anomalies in spreadsheets not handled well by the system

import os
from tqdm import tqdm
from datetime import datetime
import dateutil.parser as dateparser
from typing import Callable
NoneType = type(None)

import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import IsolationForest
import pickle


import pdb

# ideas/tasks
#   create a simple image-like auto-encoder over spreadsheet space
#   web scrape for datasets to test if they would work or not. this is how to get negative examples
#   tbd


def main():

    #set up the dataset to train on
    print('Loading data...')
    data_paths = [
        '~/Downloads/cleaned_data_1.csv',
        '~/Downloads/cleaned_data_2.csv',
        '~/Downloads/raw_data.csv',
        '~/Downloads/edrmc_raw.csv',
        '~/Downloads/flood_area.csv',
        '~/Downloads/dhs_nutrition.csv',
        '~/Downloads/2_FAO_Locust_Swarms.csv',
        '~/Downloads/vdem_2000_2020.csv',
        '~/Downloads/DTM_IDP_Geocoded_fix.csv',
        '~/Downloads/maln_inference_c63039d5e4.csv',
        '~/Downloads/Monthly_Demand_Supply_water_WRI_Gridded.csv',
        '~/Downloads/NextGEN_PRCP_FCST_TercileProbablity_Jun-Sep_iniMay.csv',

        # '~/Downloads/messy_data_1.csv',
        # '~/Downloads/messy_data_2.csv',
    ]
    data = []
    for path in tqdm(data_paths):
        data.append(csv_to_img(path))

    data = torch.stack(data)
    data.to('cuda')

    #create a dataset from the data
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # # debug
    # for (img,) in dataset:
    #     imshow(img)


    # train an auto-encoder on the dataset
    in_channels = len(datatype_list)
    latent_channels = 8
    encoder = Encoder(in_channels, latent_channels).cuda()
    decoder = Decoder(latent_channels, in_channels).cuda()


    #check if a saved version of the auto-encoder exists, otherwise train a new one
    if os.path.exists('encoder.pt') and os.path.exists('decoder.pt'):
        print('loading saved auto-encoder model')
        encoder.load_state_dict(torch.load('encoder.pt'))
        decoder.load_state_dict(torch.load('decoder.pt'))
    else:
        #train the model
        optimizer = optim.Adam(nn.ModuleList([encoder, decoder]).parameters(), lr=5e-4)

        for epoch in range(100000):
            # print(f'------------------ epoch {epoch} ------------------') #don't print since dataset is too small right now
            #train the model
            for (data,) in loader:
                data = data.to('cuda')
                optimizer.zero_grad()
                output = decoder(encoder(data))
                loss = F.mse_loss(output, data)
                loss.backward()
                optimizer.step()
                print(f'{epoch}: {loss.item()}')

        # save trained model
        torch.save(encoder.state_dict(), 'encoder.pt')
        torch.save(decoder.state_dict(), 'decoder.pt')

    # set model to eval mode
    encoder.eval(); decoder.eval()

    #convert all examples in the dataset to latent vectors for the isolation forest
    latent_vectors = []
    for (data,) in loader:
        data = data.to('cuda')
        latent_vectors.append(encoder(data).detach().cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors, axis=0)

    #get latent vectors for the anomalous data
    anomalous_data = torch.stack([
        csv_to_img('~/Downloads/messy_data_1.csv'),
        csv_to_img('~/Downloads/messy_data_2.csv'),
    ]).to('cuda')
    anomalous_latent_vectors = encoder(anomalous_data).detach().cpu().numpy()


    #check if a saved version of the isolation forest exists, otherwise train a new one
    if os.path.exists('isolation_forest.pkl'):
        print('loading saved isolation forest model')
        isolation_forest = pickle.load(open('isolation_forest.pkl', 'rb'))
    else:
        #train the model
        isolation_forest = IsolationForest(contamination=0.01).fit(latent_vectors)
        pickle.dump(isolation_forest, open('isolation_forest.pkl', 'wb'))

    normal_predictions = isolation_forest.predict(latent_vectors)
    anomalous_predictions = isolation_forest.predict(anomalous_latent_vectors)
    print(f'normal: {normal_predictions}')
    print(f'anomalous: {anomalous_predictions}')



    #debug show the data and the reconstruction
    with torch.no_grad():
        # test the model
        for (data,) in loader:
            data = data.to('cuda')
            output = decoder(encoder(data))
            for img, img_hat in zip(data, output):
                imshow(torch.cat([img, img_hat], dim=2))





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
    #ensure on cpu and not attached to gradients
    sheet = sheet.detach().cpu()
    
    #move color channel last so that matplotlib understands it
    sheet = sheet.permute(1, 2, 0)

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

    sheet = pd.read_csv(path, header=None, keep_default_na=False, low_memory=False, *args, **kwargs)
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
        x = F.sigmoid(x)
        return x




if __name__ == '__main__':
    main()