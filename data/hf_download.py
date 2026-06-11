import requests
from tqdm import tqdm

def download_file(url, output_file):
    print(f"Downloading from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, "wb") as file, tqdm(
            desc=output_file,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024*1024):
                size = file.write(chunk)
                bar.update(size)
        print(f"Success!")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

# Try raw git objects
urls = [
    "https://raw.githubusercontent.com/Wang-Shuo/Amazon-Reviews-2014/master/meta_Beauty.json.gz",
    "https://raw.githubusercontent.com/hyperscience/amazon-product-data/master/meta_Beauty.json.gz",
    "https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz" # Test tuna
]

from pathlib import Path

output_file = str(Path(__file__).resolve().parent / "meta_Beauty.json.gz")

for url in urls:
    if download_file(url, output_file):
        break
