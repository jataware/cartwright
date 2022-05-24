#script for training models to catch anomalies in spreadsheets not handled well by the system

from os import walk, makedirs
from os.path import join, relpath, exists, getsize, dirname, abspath, expanduser
from tqdm import tqdm
from datetime import datetime
import dateutil.parser as dateparser
from typing import Callable
NoneType = type(None)
from multiprocessing import Pool, cpu_count

import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset


import pdb

# ideas/tasks
#   create a simple image-like auto-encoder over spreadsheet space
#   web scrape for datasets to test if they would work or not. this is how to get negative examples
#   tbd


def main():

    #set up the dataset to train on
    dataset = SpreadsheetLoader(
        data_root='~/Downloads/datasets/spreadsheet_images', 
        raw_root='~/Downloads/datasets/spreadsheets'
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # # debug show images from the dataset
    # for img in dataset:
    #     imshow(img)

    # train an auto-encoder on the dataset
    in_channels = len(datatype_list)
    latent_channels = 32
    encoder = Encoder(in_channels, latent_channels).cuda()
    decoder = Decoder(latent_channels, in_channels).cuda()


    #check if a saved version of the auto-encoder exists, otherwise train a new one
    try:
        load_autoencoder(encoder, decoder)
    except FileNotFoundError:
        
        #train the model from scratch
        optimizer = optim.Adam(nn.ModuleList([encoder, decoder]).parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        for epoch in range(100000):
            total_loss = 0
            count = 0
            for data in loader:
                data = data.to('cuda')
                optimizer.zero_grad()
                output = decoder(encoder(data))
                loss = F.mse_loss(output, data)
                loss.backward()
                optimizer.step()
                count += 1
                total_loss += loss.item()

            print(f'epoch: {epoch}, loss: {total_loss / count}')
            scheduler.step()

            if (epoch+1) % 100 == 0:
                save_autoencoder(encoder, decoder)

        # save trained model after end of training
        save_autoencoder(encoder, decoder)

    # set model to eval mode
    encoder.eval(); decoder.eval()

    #convert all examples in the dataset to latent vectors for the isolation forest
    with torch.no_grad():
        normal_latent_vectors = []
        normal_scores = []
        for data in loader:
            data = data.to('cuda')
            latent = encoder(data)
            normal_latent_vectors.append(latent)
            reconstructions = decoder(latent)
            scores = ((reconstructions - data)**2).mean(dim=(-1,-2,-3))
            normal_scores.append(scores)

        normal_latent_vectors = torch.cat(normal_latent_vectors, dim=0)
        normal_scores = torch.cat(normal_scores, dim=0)
        
        #get latent vectors for the anomalous data
        print(f'scoring anomalous data...')
        anomalous_data = torch.stack([
            csv_to_img(x) for x in tqdm([
                '~/Downloads/messy_data_1.csv',
                '~/Downloads/messy_data_2.csv',
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
            ])
        ]).to('cuda')
        anomalous_latent_vectors = encoder(anomalous_data)
        anomalous_reconstructions = decoder(anomalous_latent_vectors)
        anomalous_scores = ((anomalous_reconstructions - anomalous_data)**2).mean(dim=(-1,-2,-3))

        #plot a histogram of the normal scores, and points of the anomalous scores on top
        plt.figure()
        plt.hist(normal_scores.cpu())
        plt.scatter(anomalous_scores.cpu(), np.ones(anomalous_scores.shape[0])*20, c='r', s=200, marker='v')
        plt.show()

        pdb.set_trace(); 1


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
    # with Pool(cpu_count()) as pool:
    #     return np.array([*pool.map(func, arr.flatten())], dtype=dtype).reshape(shape)
    
    return np.array([*map(func, arr.flatten())], dtype=dtype).reshape(shape)

def csv_to_img(path: str, *args, max_rows=100_000, **kwargs) -> torch.Tensor:
    """read in a csv and convert the spreadsheet to a feature image for ML analysis"""

    if not path.endswith('.csv'):
        raise ValueError('expected a csv file')

    try:

        sheet = pd.read_csv(path, header=None, keep_default_na=False, low_memory=False, *args, **kwargs)
        sheet = np.array(sheet, dtype=object)

        #if the sheet is too large, only read the first max_rows rows
        if sheet.shape[0] > max_rows:
            sheet = sheet[:max_rows]    

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

    except Exception as e:
        print(f'unable to read csv \'{path}\'')
        raise e


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
                nn.Conv2d(in_channels=working_channels, out_channels=working_channels, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=working_channels, out_channels=working_channels, kernel_size=3, padding=1),
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
                nn.Conv2d(in_channels=working_channels, out_channels=working_channels, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=working_channels, out_channels=working_channels, kernel_size=3, padding=1),
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
        x = torch.sigmoid(x)
        return x

def save_autoencoder(encoder: Encoder, decoder: Decoder, root_dir='models/autoencoder', verbose=True):
    #ensure root_dir exists
    if not exists(root_dir):
        makedirs(root_dir)
    
    if verbose: print(f'saving autoencoder to {root_dir}')
    torch.save(encoder.state_dict(), join(root_dir, 'encoder.pt'))
    torch.save(decoder.state_dict(), join(root_dir, 'decoder.pt'))

def load_autoencoder(encoder: Encoder, decoder: Decoder, root_dir='models/autoencoder', verbose=True):
    #ensure the model exists
    if exists(join(root_dir, 'encoder.pt')) and exists(join(root_dir, 'decoder.pt')):
        if verbose: print(f'loading saved auto-encoder model from {root_dir}')
        encoder.load_state_dict(torch.load(join(root_dir, 'encoder.pt')))
        decoder.load_state_dict(torch.load(join(root_dir, 'decoder.pt')))
    else:
        raise FileNotFoundError(f'No saved autoencoder model found in directory {root_dir}')


def save_processed_sheet_data(paths_tuple):
    raw_path, out_path = paths_tuple
    if not exists(out_path):
        size_mb = getsize(raw_path)/1e6
        # if size_mb > 1000:
        #     print(f'skipping file that is too big: {size_mb:.2f} MB, path: {raw_path}')
        #     return
        # print(f'path: {raw_path}, size: {size_mb:.2f} MB')
        img = csv_to_img(raw_path) #create image
        
        #ensure output path exists to write to
        dir = dirname(out_path)
        if not exists(dir):
            makedirs(dir)

        torch.save(img, out_path) #save image as png

    # else:
        # print(f'skipping {raw_path}')
                

class SpreadsheetLoader(Dataset):
    """Load a spreadsheet as a dataset"""
    def __init__(self, data_root: str, data_suffix='raw_data.pt', raw_suffix='raw_data.csv', raw_root=None):

        data_root = abspath(expanduser(data_root))

        #convert the raw spreadsheets into images if not done already
        if raw_root is not None:
            raw_root = abspath(expanduser(raw_root))
            raw_paths = []
            for root, dirs, files in walk(raw_root):
                for file in files:
                    if file.endswith(raw_suffix):
                        raw_paths.append(join(root, file))
            
            #construct the output paths
            out_paths = []
            for raw_path in raw_paths:
                #construct the output path based on the relative path in the raw root directory
                out_path = join(data_root, relpath(raw_path, raw_root))
                out_path = out_path.replace(raw_suffix, data_suffix)
                out_paths.append(out_path)

            
            # #parallel version
            with Pool(processes=cpu_count()) as pool:
                for _ in tqdm(pool.imap_unordered(save_processed_sheet_data, zip(raw_paths, out_paths), chunksize=1), total=len(raw_paths), desc='saving preprocessed data'):
                    pass

            #non-parallel version
            # for raw_path, out_path in tqdm(zip(raw_paths, out_paths), total=len(raw_paths), desc='saving preprocessed data'):
            #     save_processed_sheet_data((raw_path, out_path))


        #load the paths to the preprocessed data
        self.paths = []
        for root, dirs, files in walk(data_root):
            for file in files:
                if file.endswith(data_suffix):
                    self.paths.append(join(root, file))
        
        print('preloading data...')            
        self.sheets = [torch.load(path) for path in tqdm(self.paths, desc='loading data', unit='sheets', total=len(self.paths))]
            

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.sheets[index]



if __name__ == '__main__':
    main()