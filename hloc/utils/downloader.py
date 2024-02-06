import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

def download_file_with_progress(url, filename):
    # Stream the download to monitor its progress
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def download_files_from_directory(url, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for link in soup.find_all('a'):
        file_url = urljoin(url, link.get('href'))
        
        if not file_url.endswith('/'):  # Assuming it's a file if the URL doesn't end with '/'
            filename = os.path.join(target_folder, file_url.split('/')[-1])
            print(f'Downloading {filename}')
            download_file_with_progress(file_url, filename)

