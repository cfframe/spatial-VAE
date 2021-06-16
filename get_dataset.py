# get_dataset.py
"""
Description: This script downloads and unzips (if a zip) files from the given source.

Example usage:

python get_dataset.py --help
python get_dataset.py -d data -rd n
python get_dataset.py -d data -rd y -rt y
python get_dataset.py -d data -rd n -ruc n -s http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated_translated.tar.gz
python get_dataset.py -d data -rd n -ruc n -s http://bergerlab-downloads.csail.mit.edu/spatial-vae/galaxy_zoo.tar.gz

Datasets for spatial-VAE
- [Rotated MNIST](http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated.tar.gz)
- [Rotated & Translated MNIST](http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated_translated.tar.gz)
- [5HDB simulated EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/5HDB.tar.gz)
- [CODH/ACS EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/codhacs.tar.gz)
- [Antibody EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/antibody.tar.gz)
- [Galaxy zoo](http://bergerlab-downloads.csail.mit.edu/spatial-vae/galaxy_zoo.tar.gz)
"""

import argparse
from src.download_helper import DownloadHelper

DOWNLOAD_URL = 'http://bergerlab-downloads.csail.mit.edu/spatial-vae/galaxy_zoo.tar.gz'


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Download the target training dataset')
    parser.add_argument('--data_dir', '-d', type=str, help="Path to the root target data director")
    parser.add_argument('--replace_download', '-rd', type=str, default='n',
                        help="Flag to overwrite existing download file")
    parser.add_argument('--replace_unzip_content', '-ruc', type=str, default='n',
                        help="Flag to replace existing unzip folder content")
    parser.add_argument('--src_url', '-s', type=str, default=DOWNLOAD_URL,
                        help="Source URL for download")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    DownloadHelper.download_dataset(**args.__dict__)
