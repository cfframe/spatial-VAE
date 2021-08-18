# numpy_array_from_images.py

from src.file_tools import FileTools
import argparse

"""Description: generate a numpy archive of images given a source directory and image extension suffix. 
Optionally resize.

Example usage:
* WARNING * This requires over 30GB RAM for ISIC2018 Challenge Task 3 training images
* USE AT YOUR OWN RISK *
python numpy_array_from_images.py -s C:/GitHub/cfframe/spatial-VAE/data/isic2018/training_input -t C:/GitHub/cfframe/spatial-VAE/data/isic2018/train2 

"""


def parse_args():
    parser = argparse.ArgumentParser(description='Download the target training dataset')
    parser.add_argument('--source_dir', '-s', type=str, help="Source directory with images")
    parser.add_argument('--target_path', '-t', type=str, help="Base path (no extension) for array to be saved")
    parser.add_argument('--new_shape', '-sh', type=tuple, default=0,
                        help="Tuple of shape, in form (rows, columns); (optional, default 0)")
    parser.add_argument('--suffix', '-e', type=str, help="Extension suffix including period/full-stop (default '.jpg')")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if input('WARNING May require a lot of memory - e.g. ISIC2018 Challenge Task 3 Training data needs over 30GB. \nContinue (y/n and Enter)?').lower() == 'n':
        quit()
    for k, v in vars(args).items():
        print(f'{k}={v or ""}')
    src_dir = args.source_dir
    target_path = args.target_path
    new_shape = args.new_shape
    suffix = args.suffix

    result = FileTools.create_numpy_archive_from_images_dir(
        src_dir=src_dir, target_path=target_path, new_shape=new_shape, suffix=suffix)

    print(result)


if __name__ == '__main__':
    main()
