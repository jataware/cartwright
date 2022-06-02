import requests
from bs4 import BeautifulSoup as bs
import os

import pdb

def make_url(query):
    """
        e.g. given 'agriculture data', 
        return https://www.google.com/search?as_q=agriculture+data&as_epq=.csv
    """
    query = '+'.join(query.split())
    return f'https://www.google.com/search?as_q={query}&as_epq=.csv'

def run_search():

    #ensure the output directory exists
    out_dir = 'data/csv/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    search_url = make_url('agriculture data')

    try:
        while True:
            #get the html
            page = requests.get(search_url)
            soup = bs(page.content, 'html.parser')

            anchors = soup.find_all('a', href=True)

            #collect the google search URLs (they start with /url?q= )
            urls = []
            for anchor in anchors:
                href = anchor['href']
                if href.startswith('/url?q=') and 'google.com' not in href:
                    url = href[7:].split('&')[0]
                    urls.append(url)


            #for each url, go into the link and download all links that are either .csv or .xls
            for url in urls:
                download_page(url, out_dir)


            #go to next page, and repeat the process
            google_path_base = 'https://www.google.com'
            next_urls = [anchor for anchor in anchors if '>' in anchor.get_text()]
            next_url = next_urls[-1]['href'] #should be length 1, but get the last one just in case
            search_url = google_path_base + next_url

    except Exception as e:
        print(e)
        return None


def download_page(url, out_dir):
    """find all tabular data on the page, and download"""

    print(f'[Downloading data from {url}]')

    #for now just append the url to the file urls.txt
    with open('data/csv/urls.txt', 'a') as f:
        f.write(f'{url}\n')

    # #make a folder for the data at out_dir/url
    # out_dir = generate_folder_name(url, out_dir)
    
    # #save a text file with the full url
    # with open(os.path.join(out_dir, 'url.txt'), 'w') as f:
    #     f.write(url)
    
    # #download any data on the page
    # page = requests.get(url)
    # soup = bs(page.content, 'html.parser')
    # anchors = soup.find_all('a', href=True)

    # for anchor in anchors:
    #     href = anchor['href']
    #     # pdb.set_trace()
    #     if href.endswith('.csv') or href.endswith('.xls'):
    #         print(href)
    #         #download the file
    #         # file_name = href.split('/')[-1]
    #         # file_path = out_dir + file_name
    #         # if not os.path.exists(file_path):
    #         #     with open(file_path, 'wb') as f:
    #         #         f.write(requests.get(href).content)
    #         #     print('downloaded: ' + file_name)

    # pdb.set_trace()

        
def generate_folder_name(url, out_dir):
    """
        given a url to some website, generate a folder name
    """
    
    clean_url = url                 \
        .replace('https://', '')    \
        .replace('http://', '')     \
        .replace('www.', '')

    illegal_chars = set('/\\#%&{}<>*? $~\'":@+`|=[]')
    clean_url = ''.join(c if c not in illegal_chars else '_' for c in clean_url)
    
    out_dir = os.path.join(out_dir, clean_url)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir

    


if __name__ == '__main__':
    run_search()