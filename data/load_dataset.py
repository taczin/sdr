import csv
import sys
import os
from data.data_utils import get_gt_seeds_titles, raw_data_link

def read_all_articles(raw_data_path):
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2147483647 else 2147483647)
    with open(raw_data_path, encoding='utf8', newline="") as f:
        reader = csv.reader(f)
        all_articles = list(reader)
    return all_articles[1:]

def download_raw(dataset_name):
    raw_data_path = f"data/datasets/{dataset_name}/raw_data"
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    if not os.path.exists(raw_data_path):
        #os.system(f"wget -O {raw_data_path} {raw_data_link(dataset_name)}") # this is for linux and mac
        os.system(f"curl.exe -o {raw_data_path} {raw_data_link(dataset_name)}") # this is for windows
    return raw_data_path

if __name__ == "__main__":
    dataset_name = 'video_games' #or 'wines'
    raw_data_path = download_raw(dataset_name)
    data = read_all_articles(raw_data_path)
    print('Finished loading')
