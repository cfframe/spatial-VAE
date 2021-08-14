# get_dataset.py
"""
Description: This script downloads and unzips (if a zip) files from the given source.

Example usage:

python get_dataset.py --help
python get_dataset.py -d data -rd
python get_dataset.py -d data -rd y -rt y
python get_dataset.py -d data -rd -ruc -s http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated_translated.tar.gz
python get_dataset.py -d data -rd -ruc -s http://bergerlab-downloads.csail.mit.edu/spatial-vae/galaxy_zoo.tar.gz

Datasets for spatial-VAE
- [Rotated MNIST](http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated.tar.gz)
- [Rotated & Translated MNIST](http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated_translated.tar.gz)
- [5HDB simulated EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/5HDB.tar.gz)
- [CODH/ACS EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/codhacs.tar.gz)
- [Antibody EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/antibody.tar.gz)
- [Galaxy zoo](http://bergerlab-downloads.csail.mit.edu/spatial-vae/galaxy_zoo.tar.gz)

Datasets for ISIC:

python get_dataset.py -d data -wd isic2018 -i -s https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
python get_dataset.py -d data -wd isic2018 -i -s https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip
python get_dataset.py -d data -wd isic2018 -i -s https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip
"""

import argparse
from src.download_helper import DownloadHelper
from src.file_tools import FileTools

DOWNLOAD_URL = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_LesionGroupings.csv'


def parse_args():
    parser = argparse.ArgumentParser(description='Download the target training dataset')
    parser.add_argument('--data_dir', '-d', type=str, help="Path to the root target data director")
    parser.add_argument('--replace_download', '-rd', action='store_true',
                        help="Flag to overwrite existing download file")
    parser.add_argument('--replace_unzip_content', '-ruc', action='store_true',
                        help="Flag to replace existing unzip folder content")
    parser.add_argument('--src_url', '-s', type=str, default=DOWNLOAD_URL,
                        help="Source URL for download")
    parser.add_argument('--is_isic', '-i', action='store_true',
                        help='Indicate download is an ISIC dataset following ISIC conventions')
    parser.add_argument('--working_dir', '-wd', type=str, default='',
                        help='Target directory for extraction etc (optional)')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    extraction_dir, working_dir = DownloadHelper.download_dataset(
        data_dir=args.data_dir,
        replace_download=args.replace_download, replace_unzip_content=args.replace_unzip_content,
        src_url=args.src_url, is_isic=args.is_isic, working_dir=args.working_dir)

    result = FileTools.create_numpy_archive_from_images_dir(
        src_dir=extraction_dir, target_path=extraction_dir, new_shape=(64, 64), suffix='.jpg')

    print(result)


if __name__ == "__main__":
    main()
