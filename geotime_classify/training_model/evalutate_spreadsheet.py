from mss.linux import MSS as mss
sct = mss()
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import os
import subprocess
from time import sleep
import signal
import shutil

from anomaly_detection import AnomalyDetector, sheet_tensor_to_img, imshow

import pdb


# def main2():
    

#     detector = AnomalyDetector()
#     detector.load_autoencoder()
#     detector.cuda()

#     if not True:    
#         csv_path = p('/home/david/Downloads/datasets/spreadsheet_web_scraping_results/percentiles/90-100/57/data.csv')
#         img = detector.csv_to_img(csv_path).cuda()
#         img = img[0] #collect the empty channel of the data

#     else:
#         img_path = p('~/Downloads/Lenna.png')
#         # img_path = p('~/Downloads/2aysZ.png')
#         # img_path = p('~/Downloads/hPjHn.png')
#         img = plt.imread(img_path)
#         img = torch.from_numpy(img).float()
#         img = img.cuda()

#         #convert to grayscale
#         img = img.mean(dim=2)
#         img = (img-img.min())/(img.max() - img.min())

#     for b in [2, 4, 8, 16, 32, 64, 128, 256]:
#         print(f'{b} bins lenna: ', detector.entropy(img, bins=b))
#     pdb.set_trace()

#     # plt.imshow(img.cpu().numpy(), cmap='gray')
#     # plt.show()


#     #take the 2-pixel partial finite differences of the image in x and y (padding to keep the original shape)
#     dx = img[2:] - img[:-2]; dx = torch.cat([torch.zeros_like(dx[:1]), dx, torch.zeros_like(dx[:1])], dim=0)
#     dy = img[:,2:] - img[:,:-2]; dy = torch.cat([torch.zeros_like(dy[:,:1]), dy, torch.zeros_like(dy[:,:1])], dim=1)

#     #reshape dx/dy into 1D vectors
#     dx_vec = dx.view(-1).cpu().numpy()
#     dy_vec = dy.view(-1).cpu().numpy()

#     #compute the 2D histogram of the derivatives using numpy
#     bins = 5
#     hist, xedges, yedges = np.histogram2d(dx_vec, dy_vec, bins=bins, range=[[-1,1], [-1,1]])
#     hist /= hist.sum()
    
#     #compute the entropy per bin in the histogram
#     entropy = torch.tensor(hist * np.log(hist + 1e-10)).cuda()

#     #map each dx/dy pair to a bin in the histogram
#     # ndx = (dx - dx.min()) / (dx.max() - dx.min())
#     # ndy = (dy - dy.min()) / (dy.max() - dy.min())
#     # x = (ndx * bins).clamp(0, bins-1).long()
#     # y = (ndy * bins).clamp(0, bins-1).long()
#     # x = np.digitize(dx, )
    
#     #generate an image of the per-pixel entropy
#     entropy_img = entropy[x, y]
#     entropy_img = (entropy_img - entropy_img.min()) / (entropy_img.max() - entropy_img.min())

#     plt.imshow(torch.cat([img, entropy_img], dim=1).cpu(), cmap='gray')
#     plt.show()

#     #plot the original image, but for each pixel, plot the entropy of the 2D histogram
#     #plot the histogram
#     # plt.figure()
#     # plt.scatter(dx.cpu().numpy(), dy.cpu().numpy())
#     # plt.figure()
#     # plt.imshow(np.log(hist+1e-6), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='magma')
#     # plt.show()
    
#     #convert dx/dy to integer -255 <= x <= 255 range
#     # dx = (dx * 255).round()
#     # dy = (dy * 255).round()
#     pdb.set_trace()

#     #generate a 256x256 image representing a 2D histogram of the gradient values



#     #take the magnitude of the finite differences
#     pdb.set_trace()
#     mag = (dx**2 + dy**2).sqrt()

#     #plot the magnitude
#     plt.imshow(mag.cpu().numpy(), cmap='gray')
#     plt.show()
    


#     pdb.set_trace()

#     #take the 2-pixel finite difference of the test image
#     img_diff = img[:,:,:2] - img[:,:,1:-1]
    
    
#     #######TESTING/DEBUG#######
#     detector = AnomalyDetector()
#     detector.load_autoencoder()
#     detector.cuda()
#     csv_path = p('/home/david/Downloads/datasets/spreadsheet_web_scraping_results/percentiles/90-100/1/data.csv')
#     img = detector.csv_to_img(csv_path).cuda()
#     entropy = detector.empty_entropy(img)
#     pdb.set_trace()
#     1
#     ###########################

def main():

    #ML model for scoring each spreadsheet
    detector = AnomalyDetector()
    detector.load_autoencoder()
    detector.cuda()

    entropies = []

    #path to raw data + output location
    data_path = p('~/Downloads/datasets/spreadsheet_web_scraping/')
    save_path = p('~/Downloads/datasets/spreadsheet_web_scraping_results/')
    paths = list(walk_root(data_path, lambda x: x.endswith('.csv')))
    print(f'Evaluating {len(paths)} spreadsheets in {data_path}')
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        #get the filename for the spreadsheet
        filename = os.path.basename(path)

        #create a folder at save_path/i
        save_dir = os.path.join(save_path, str(i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # #save the screenshot
        # sct_path = os.path.join(save_dir, 'screenshot.png')
        # if not os.path.exists(sct_path):
        #     img = screenshot_spreadsheet(path)
        #     plt.imsave(sct_path, img)

        #copy the csv file
        csv_path = os.path.join(save_dir, 'data.csv')
        if not os.path.exists(csv_path):
            shutil.copy(path, csv_path)

        #score the spreadsheet
        img_path = os.path.join(save_dir, 'image.png')
        rec_path = os.path.join(save_dir, 'reconstruction.png')
        hist_path = os.path.join(save_dir, 'histogram.png')
        result_path = os.path.join(save_dir, 'results.txt')
        if not os.path.exists(img_path) or not os.path.exists(rec_path) or not os.path.exists(hist_path) or not os.path.exists(result_path):
            try:
                with torch.no_grad():
                    img = detector.csv_to_img(csv_path).cuda()
                    reconstruction = detector(img[None])[0]
                    score, percentile = detector.rate(img)
                    score, percentile = score.item(), percentile.item()
                    entropy = detector.entropy(img[0], bins=2)
                    
                #save the image and the reconstruction
                plt.figure()
                imshow(img, show=False)
                plt.savefig(img_path)
                plt.close()
                plt.figure()
                imshow(reconstruction, show=False)
                plt.savefig(rec_path)
                plt.close()

                #save a histogram of the score relative to the training data
                plt.figure()
                plt.hist(detector.scores.cpu())
                plt.scatter([score], [0], c='r', s=200, marker='^')
                plt.annotate(filename, xy=(score, 10), rotation=90, ha='center')
                plt.savefig(hist_path)
                plt.close()

                #save a text file with the filename, score, and percentile
                with open(result_path, 'w') as f:
                    f.write(f'filename: {filename}\nscore: {score}\npercentile: {percentile}\nentropy: {entropy}')

                #save the entropy
                entropies.append(entropy)

            except Exception as e:
                print(f'{filename} failed: {e}')
                with open(result_path, 'w') as f:
                    f.write(f'filename: {filename}\nfailed: {e}')



    #finally, sort all the spreadsheets into folders of percentile ranges
    #TODO->make it just put them in the final folder in the first place
    sort_by_percentile(save_path)

    #plot a histogram of the entropies
    plt.figure()
    plt.hist(entropies)
    plt.show()

    pdb.set_trace()


def histogram_of_scores(save_path):
    """plot a histogram of the recorded scores"""

    all_scores = []
    for path in walk_root(save_path, lambda x: x.endswith('results.txt')):
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                score = float(lines[1].split()[1])
                all_scores.append(score)
        except Exception as e:
            print(f'{path} failed: {e}')
            continue
    plt.hist(all_scores)
    plt.show()

def sort_by_percentile(save_path):
    """sort all the spreadsheets into folders of percentile ranges"""
    #for each folder in the save_path
    for folder in os.listdir(save_path):
        folder_path = os.path.join(save_path, folder)

        #get the percentile from {folder}/results.txt
        with open(os.path.join(folder_path, 'results.txt'), 'r') as f:
            for line in f:
                if line.startswith('percentile'):
                    percentile = float(line.split(' ')[1])
                    break
            else: 
                # raise Exception('percentile not found')
                print(f'percentile not found for {folder}')
                continue
        
        #move the folder to the percentile folder ranges are 0-10%, 10-20%, ..., 90-100%
        #check if the folder already exists
        percentile_lower = int(abs(percentile - 1e-6) * 10) * 10
        percentile_upper = percentile_lower + 10
        percentile_folder = os.path.join(save_path, 'percentiles', f'{percentile_lower}-{percentile_upper}')
        # pdb.set_trace()
        if not os.path.exists(percentile_folder):
            os.makedirs(percentile_folder)

        #move the folder
        shutil.move(folder_path, percentile_folder)




def p(path):
    """fully qualify the path"""
    return os.path.abspath(os.path.expanduser(path))

def screenshot():
    """take a screenshot"""
    img = sct.grab(sct.monitors[0])
    return bgr2rgb(np.array(img))

def bgr2rgb(img):
    """convert the bgr format to rgb"""
    return img[:,:,[2,1,0]] #also removes alpha channel if present

def walk_root(root_dir, filter: lambda x: True):
    """generator for walking through all files in a folder, with optional filter"""
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if filter(file):
                yield os.path.join(root, file)


def open_csv_in_libreoffice(path, wait_time=2):
    subprocess.Popen(['libreoffice', '--norestore', '--calc', path, '--infilter="CSV:44,34,0,1"'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
    sleep(wait_time)
    

def close_libreoffice():
    #determine the process id (not that process.pid is not correct)
    pids = subprocess.Popen(['pgrep', '-f', 'libreoffice'], stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')[:-1]
    pids = [int(pid) for pid in pids]

    #close libreoffice
    try:
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
    except:
        pass


def screenshot_spreadsheet(path):
    #open the spreadsheet in libreoffice    
    pids = open_csv_in_libreoffice(path)

    #take a screenshot
    img = screenshot()

    #crop to (114,224) by (2488,1382)
    img = img[225:1382, 114:2488]
    
    #close libreoffice
    close_libreoffice()

    return img

if __name__ == '__main__':
    # main2()
    main()
    sct.close()