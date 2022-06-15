"""
script for training models to catch anomalies in spreadsheets not handled well by the system


How to use externally:
from anomaly_detection import AnomalyDetector

#instantiate the model
detector = AnomalyDetector()
detector.load_autoencoder()

#load a spreadsheet+convert to image
img = detector.csv_to_img('path/to/you/data.csv')

#to get raw scores/percentiles
score, percentile = detector.rate(img)

#or, to get 'low', 'medium', or 'high' classification (thresholds are optional)
result = detector.classify(img, low_threshold=0.33, high_threshold=0.66, entropy_threshold=0.15)

"""



from os import walk, makedirs
from os.path import join, relpath, exists, getsize, dirname, abspath, expanduser, split
import re
from tqdm import tqdm
from datetime import datetime
import dateutil.parser as dateparser
from typing import Callable, List, Union
NoneType = type(None)
from multiprocessing import Pool, cpu_count

import pandas as pd
from pandas import DataFrame
import numpy as np
from scipy.stats import percentileofscore
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
        data_root='~/Downloads/datasets/spreadsheet_images',    #preprocessed images from spreadsheets
        raw_root='~/Downloads/datasets/spreadsheets',           #raw spreadsheets
        raw_suffix='raw_data.csv',
        data_suffix='raw_data.pt'
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    #dataset of spreadsheets from google
    wild_dataset = SpreadsheetLoader(
        data_root='~/Downloads/datasets/wild_spreadsheet_images/Crop_production',    #preprocessed images from spreadsheets
        raw_root='~/Downloads/datasets/wild_spreadsheets/Crop_production',           #raw spreadsheets
        raw_suffix='.csv',
        data_suffix='.pt'
    )


    # train an auto-encoder on the dataset
    detector = AnomalyDetector().cuda()

    #check if a saved version of the auto-encoder exists, otherwise train a new one
    try:
        detector.load_autoencoder()
    except FileNotFoundError:
        
        #train the model from scratch
        optimizer = optim.Adam(detector.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        for epoch in range(100000):
            total_loss = 0
            count = 0
            for data in loader:
                data = data.to('cuda')
                optimizer.zero_grad()
                output = detector(data) #decoder(encoder(data))
                loss = detector.score(output, data)
                loss.backward()
                optimizer.step()
                count += 1
                total_loss += loss.item()

            print(f'epoch: {epoch}, loss: {total_loss / count}')
            scheduler.step()

            if (epoch+1) % 100 == 0:
                detector.save_autoencoder()

        # save trained model after end of training
        detector.save_autoencoder()

    # set model to eval mode
    detector.eval()

    #compute reconstruction scores for each image in the dataset
    with torch.no_grad():
        normal_scores = []
        for data in loader:
            data = data.to('cuda')
            reconstructions = detector(data)
            scores = detector.score(reconstructions, data, reduce_batch_dim=False)
            normal_scores.append(scores)

        #save the normal data scores for later to compute quartiles
        normal_scores = torch.cat(normal_scores, dim=0)
        torch.save(normal_scores, 'models/autoencoder/scores.pt')
        
        #get reconstruction scores for the anomalous data
        print(f'scoring anomalous data...')
        # anomalous_data = torch.stack([
        #     AnomalyDetector.csv_to_img(x) for x in tqdm([
        #         '~/Downloads/messy_data_1.csv',
        #         '~/Downloads/messy_data_2.csv',
        #         '~/Downloads/cleaned_data_1.csv',
        #         '~/Downloads/cleaned_data_2.csv',
        #         '~/Downloads/raw_data.csv',
        #         '~/Downloads/edrmc_raw.csv',
        #         '~/Downloads/flood_area.csv',
        #         '~/Downloads/dhs_nutrition.csv',
        #         '~/Downloads/2_FAO_Locust_Swarms.csv',
        #         '~/Downloads/vdem_2000_2020.csv',
        #         '~/Downloads/DTM_IDP_Geocoded_fix.csv',
        #         '~/Downloads/maln_inference_c63039d5e4.csv',
        #         '~/Downloads/Monthly_Demand_Supply_water_WRI_Gridded.csv',
        #         '~/Downloads/NextGEN_PRCP_FCST_TercileProbablity_Jun-Sep_iniMay.csv',
        #     ])
        # ]).to('cuda')
        anomalous_data = torch.stack([sheet for sheet in wild_dataset]).to('cuda')
        anomalous_reconstructions = detector(anomalous_data)
        # anomalous_scores = detector.score(anomalous_reconstructions, anomalous_data, reduce_batch_dim=False)
        anomalous_scores, anomalous_percentiles = detector.rate(anomalous_data)

        #plot a histogram of the normal scores, and points of the anomalous scores on top
        plt.figure()
        plt.hist(normal_scores.cpu())
        plt.scatter(anomalous_scores.cpu(), np.ones(anomalous_scores.shape[0])*0, c='r', s=200, marker='^')

        #annotate the scatter plot with the filenames of each point, rotated by 45 degrees
        for score, path in zip(anomalous_scores, wild_dataset.paths):
            plt.annotate(
                split(path)[1], 
                xy=(score, 10), 
                # xytext=(score, 20), 
                rotation=90,
                ha='center',
            )

        plt.show()

        pdb.set_trace()
        1



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


def imshow(sheet, show=True):
    sheet = sheet_tensor_to_img(sheet)
    plt.imshow(sheet)
    if show:
        plt.show()


def sheet_tensor_to_img(sheet):
    #ensure on cpu and not attached to gradients
    sheet = sheet.detach().cpu()
    
    #move color channel last so that matplotlib understands it
    sheet = sheet.permute(1, 2, 0)

    #resize color channel if more than 3
    if sheet.shape[0] != 3:
        sheet = F.interpolate(sheet, size=3, mode='linear', align_corners=False)

    return sheet

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


class AnomalyDetector(nn.Module):
    def __init__(self, latent_channels=32, root_dir='models/autoencoder'):
        super().__init__()
        in_channels = len(datatype_list)
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(latent_channels, in_channels)

        self.root_dir = root_dir
        self.encoder_path = join(root_dir, 'encoder.pt')
        self.decoder_path = join(root_dir, 'decoder.pt')
        self.scores_path = join(root_dir, 'scores.pt') #scores of all data that was trained. for generating percentile scores
       
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def load_autoencoder(self, verbose=True):
        #ensure the model exists
        if exists(self.encoder_path) and exists(self.decoder_path):
            if verbose: print(f'loading saved auto-encoder model from {self.root_dir}')
            self.encoder.load_state_dict(torch.load(self.encoder_path))
            self.decoder.load_state_dict(torch.load(self.decoder_path))
            self.scores = torch.load(self.scores_path)
            self.eval()
        else:
            raise FileNotFoundError(f'No saved autoencoder model found in directory {self.root_dir}')
    
    def save_autoencoder(self, verbose=True):
        #ensure root_dir exists
        if not exists(self.root_dir):
            makedirs(self.root_dir)
        
        if verbose: print(f'saving autoencoder to {self.root_dir}')
        torch.save(self.encoder.state_dict(), self.encoder_path)
        torch.save(self.decoder.state_dict(), self.decoder_path)

    @staticmethod
    def entropy(img, bins=4):
        """measure the 2D shannon entropy of an image"""

        #if torch, convert to numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        assert len(img.shape) == 2, 'image must be 2D'

        #take the 2-pixel partial finite differences of the image in x and y (padding to keep the original shape)
        dx = img[2:] - img[:-2]; dx = np.concatenate([np.zeros_like(dx[:1]), dx, np.zeros_like(dx[:1])], axis=0)
        dy = img[:,2:] - img[:,:-2]; dy = np.concatenate([np.zeros_like(dy[:,:1]), dy, np.zeros_like(dy[:,:1])], axis=1)

        #compute the 2D histogram, i.e. the joint pdf of dx and dy
        hist, xedges, yedges = np.histogram2d(dx.flatten(), dy.flatten(), bins=bins, range=[[-1,1], [-1,1]])
        hist /= hist.sum()

        entropy = -(hist * np.log2(hist + 1e-10)).sum()

        return entropy


    
    def score(self, data: torch.Tensor, reconstructions: torch.Tensor, reduce_batch_dim=True):
        reduction_dims = (*range(not reduce_batch_dim, len(data.shape)),)
        return ((reconstructions - data)**2).mean(dim=reduction_dims)

    def rate(self, data):
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        assert len(data.shape) == 4, 'data must be a tensor of shape ([batch,] channels, height, width)'

        #get the reconstruction of the data
        reconstructions = self.forward(data)

        #score the reconstruction
        scores = self.score(data, reconstructions, reduce_batch_dim=False)

        #get the percentile of the score relative to self.scores
        percentiles = torch.tensor([percentileofscore(self.scores.cpu(), score.item(), kind='weak') for score in scores]) / 100

        return scores, percentiles

    
    def classify(self, data, low_threshold=0.33, high_threshold=0.66, entropy_threshold=0.15) -> Union[str, List[str]]:
        """
        classify the data as low, medium, or high probability of being anomalous
        returns a list of "low", "medium", or "high" (or single string if only one image passed in)
        """
        LOW=0
        MEDIUM=1
        HIGH=2
        options = ('low', 'medium', 'high')

        #handle single image
        multiple = len(data.shape) > 3
        if not multiple:
            data = data.unsqueeze(0)

        #compute the score
        _, percentiles = self.rate(data)
        classes = [LOW if percentile < low_threshold else MEDIUM if percentile < high_threshold else HIGH for percentile in percentiles]

        #measure the entropy of the data
        entropies = [self.entropy(img[0], bins=2) for img in data]

        #adjust classifications down one level if the entropy is high
        classes = [c - 1 if c > LOW and e > entropy_threshold else c for (e, c) in zip(entropies, classes)]

        #convert to strings
        classes = [options[c] for c in classes]

        #strip extra dimension if it was just a single image passed in
        if not multiple:
            classes = classes[0]

        return classes
    
    #TODO
    #@staticmethod
    #def xlsx_to_img(xlsx_path):
    # if path.endswith('.xlsx'):
    #     sheet = pd.read_excel(path, *args, **kwargs)


    @staticmethod
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

        

def save_processed_sheet_data(paths_tuple):
    """Simple loop for dumping the data from a csv file into a preprocessed torch tensor"""
    raw_path, out_path = paths_tuple
    if not exists(out_path):
        img = AnomalyDetector.csv_to_img(raw_path) #create image
        
        #ensure output path exists to write to
        dir = dirname(out_path)
        if not exists(dir):
            makedirs(dir)

        torch.save(img, out_path) #save image as png
       

class SpreadsheetLoader(Dataset):
    """Load a spreadsheet as a dataset"""
    def __init__(self, data_root: str, data_suffix='.pt', raw_suffix='.csv', raw_root=None):

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