import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import zipfile

def is_valid_file_or_dir(href):
    # Basic check to exclude non-file and non-directory patterns
    return href and ('/' in href or '.' in href)

def get_local_path_for_href(base_local_path, href):
    # Normalize and construct a safe local path for the href
    # This removes any '../' and ensures the path stays within the target directory
    normalized_href = os.path.normpath(href).strip('/')
    return os.path.join(base_local_path, normalized_href)



def download_file_with_progress(url, filename):
    response = requests.head(url)
    total_size_in_bytes = int(response.headers.get('content-length', 0))

    if os.path.exists(filename):
        existing_file_size = os.path.getsize(filename)
        if existing_file_size == total_size_in_bytes:
            print(f"File already exists and is complete: {filename}")
            return

    print(f"Downloading: {filename}")
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size_in_bytes,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_files_from_directory(url, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Filter out the parent directory link explicitly
    for link in soup.find_all('a'):
        href = link.get('href')
        if not href or 'Parent Directory' in link.text or '?' in href:
            continue  # Skip the parent directory link, sorting links, and any other non-relevant links

        file_url = urljoin(url, href)
        local_path = os.path.join(target_folder, href.strip('/'))

        # Check if it's likely a directory (ends with a slash)
        if href.endswith('/'):
            next_directory = os.path.join(target_folder, href.strip('/'))
            if not os.path.exists(next_directory):
                os.makedirs(next_directory)
            download_files_from_directory(file_url, next_directory)
        else:
            download_file_with_progress(file_url, local_path)



def unpack_zip(zip_path, extract_to):
    """
    Unpacks a zip file to the specified directory, preserving its file structure.

    :param zip_path: The path to the zip file.
    :param extract_to: The directory where the zip contents will be extracted.
    """
    # Ensure the target directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the contents into the directory
        zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")

# # Example usage
# url = 'https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/'
# target_folder = '/data/aachen/'
# download_files_from_directory(url, target_folder)