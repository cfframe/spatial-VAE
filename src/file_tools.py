# file_tools.py

import datetime
import numpy as np
import pandas
from pathlib import Path
from PIL import Image
import os
import shutil
from skimage.transform import resize
import sys


class FileTools:
    """Utilities for managing data from and to files"""

    @staticmethod
    def chunks_generator(input_list: list, chunk_size: int) -> list:
        """Yield chunks of supplied data by given size

        :param input_list: the list from which chunks are to be yielded
        :param chunk_size: number of items in each chunk
        :returns: list of length chunk_size
        """
        remainder = len(input_list) - (int(len(input_list)/chunk_size) * chunk_size)

        for i in range(0, int(len(input_list) / chunk_size)):
            yield input_list[i * chunk_size: (i + 1) * chunk_size]
        if remainder > 0:
            i = int(len(input_list)/chunk_size)
            yield input_list[i * chunk_size: i * chunk_size + remainder]

    @staticmethod
    def create_dirs_from_file_header(file_path: str, separator: str, target_root: str) -> list():
        """Generate folder names from the first line of a file

        Assumes the first column is a list of files/items and the remaining column headers require associated folders

        Keyword arguments:
        :param file_path: full path to file
        :param separator: string separator for folder names
        :param target_root: root where new folders to be created
        :returns list of folder names
        """
        with open(file_path, 'r') as infile:
            header_line = (infile.readline()).strip()

        headers = header_line.split(separator)[1:]

        for header in headers:
            os.mkdir(os.path.join(target_root, header))

        return headers

    @staticmethod
    def copy_files_to_class_dirs(info_file_path: str, separator: str, src_root: str, target_root: str,
                                 extension: str = ''):
        """Copy files from source dir to class dirs

        Keyword arguments:
        :param info_file_path: full path to file with class data of source files; assume this structure:
            line 1: headers
            column 1: file names
        :param separator: string separator for class names
        :param src_root: root for source files
        :param target_root: root for class dirs
        :param extension: if extension given, then suffix to file names
        :returns list of folder names
        """

        df = pandas.read_csv(info_file_path, index_col=0)

        FileTools.create_dirs_from_file_header(info_file_path, separator, target_root)

        for col in df.columns:
            target_dir = os.path.join(target_root, col)
            count = 0
            for filename in df[df[col] == 1].index:
                src_file = os.path.join(src_root, '.'.join([filename, extension]))
                target_file = os.path.join(target_dir, '.'.join([filename, extension]))

                shutil.copyfile(src_file, target_file)
                count += 1
            print(f'{count} files copied to {target_dir}')

        return df

    @staticmethod
    def ensure_empty_directory(dir_path: str) -> str:
        """If path does not exist, create it. If it does exist, empty it.

        Keyword arguments:
        :param dir_path: root directory path
        :returns: descriptor of result
        """
        result = 'Invalid'

        try:
            if not dir_path:
                raise ValueError('No value supplied for directory path.')

            if Path(dir_path).exists():
                if len(os.listdir(dir_path)) > 0:
                    result = 'Directory exists, not empty, deleting content'
                    # Clear sub-dirs first then tackle files
                    for root, dirs, files in os.walk(dir_path, topdown=False):
                        for directory in dirs:
                            to_remove = os.path.join(root, directory)
                            shutil.rmtree(to_remove)
                    for root, dirs, files in os.walk(dir_path, topdown=False):
                        for file in files:
                            to_remove = os.path.join(dir_path, file)
                            os.remove(to_remove)
                else:
                    result = 'Directory exists'
            else:
                result = 'Creating directory'
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        except ValueError as err:
            raise err

        except Exception as err:
            error_message = \
                "Unexpected error in FileTools.ensure_empty_directory\n"\
                + str(err.args)
            raise Exception(error_message)

        print('{}: {}'.format(result, dir_path))
        return result

    @staticmethod
    def lines_list_from_file(file_path: str) -> list:
        """Retrieve lines of text from file, return list

        Keyword arguments:
        :param file_path: full path to file
        :returns: list of text lines
        """

        with open(file_path, 'r') as infile:
            line_list = infile.readlines()

        # Strip newline chars
        line_list = [line.strip() for line in line_list]

        return line_list

    @staticmethod
    def make_datetime_named_archive(base_name: str, format: str, dir_path_to_archive: str):
        """Make archive, name prefixed with current datetime (yyyymmdd_HHMM_).
        For more detail of each parameter, see definition of shutil.make_archive.

        Example usage:

        shutil.make_archive('/home/code/target_file_name', 'zip', '/home/code/', 'base_directory')

        Keyword arguments:

        :param base_name: str, the full path of the file to create, including the base name, minus any format-specific
        extension; datetime will be prefixed to the base name
        :param format: str, the archive format
        :param dir_path_to_archive: str, the path to the directory that is to be archived
        :returns: name of file
        """
        print('Archiving files...')
        file_name = datetime.datetime.now().strftime('%y%m%d_%H%M_') + Path(base_name).name
        dir_path = Path(base_name).parent
        base_name = os.path.join(dir_path, file_name)

        root_dir = Path(dir_path_to_archive).parent
        base_dir = Path(dir_path_to_archive).name
        # print('\nmake_archive params etc')
        # print('base_name: {}'.format(base_name))
        # print('root_dir: {}'.format(root_dir))
        # print('base_dir: {}'.format(base_dir))

        result = shutil.make_archive(base_name, format, root_dir, base_dir)

        end_file_name = base_name + '.' + format

        print('Images saved at {}'.format(end_file_name))

        return result

    @staticmethod
    def save_command_args_to_file(args: dict, save_path: str):
        """Save arguments and their values to file. Expects args of type dict, so use vars(args) as input.

        Keyword arguments:

        :param args: dict, full arguments list
        :param save_path: str, path to file
        """
        parts = ['python']
        lines = []
        parts.append(os.path.basename(sys.argv[0]))
        for item in sys.argv[1:]:
            parts.append(item)

        command_line = ' '.join(parts) + '\n'

        for k, v in args.items():
            lines.append('{}={}'.format(k, v or ''))

        lines.insert(0, command_line)
        content = '\n'.join(lines)

        with open(save_path, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
            print('Command arguments saved to {}.'.format(save_path))

    @staticmethod
    def create_numpy_archive_from_images_dir(src_dir: str, target_path: str,
                                             new_shape: tuple = 0,
                                             suffix: str = '.jpg'):
        """Create a numpy array archive of images sourced from a single directory.

        If new_shape is not provided, and images are of different dimensions, then this will generate
        an exception.

        Keyword arguments:

        :param src_dir: path to source directory
        :param target_path: path to final final, excluding extension
        :param new_shape: optional, end shape of resized image arrays
        :param suffix: suffix of images to be processed, including preceding full-stop (default '.jpg')
        """
        # Catch items where None passed in
        if new_shape is None:
            new_shape = 0
        if suffix is None:
            suffix = '.jpg'

        if src_dir == '':
            result = f'No source directory supplied for images, so no npy file created.'
        elif not Path(src_dir).is_dir():
            result = f'"{src_dir}" is not a directory, so no npy file created.'
        else:
            image_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
                           if os.path.isfile(os.path.join(src_dir, f))
                           and Path(os.path.join(src_dir, f)).suffix == suffix]

            if len(image_files) == 0:
                result = f'No {suffix} files at {src_dir} so no npy file created.'
            else:
                processed_images = []

                try:
                    for img in [np.array(Image.open(image_path)) for image_path in image_files]:
                        processed_images.append(
                            np.asarray(
                                np.asarray(img, dtype='int') if new_shape == 0 else
                                resize(img, new_shape, preserve_range=True, anti_aliasing=False),
                                dtype='int'
                            )
                        )
                except Exception as err:
                    error_message = \
                        "Unexpected error in FileTools.create_numpy_archive_from_images_dir\n"\
                        + str(err.args)
                    raise Exception(error_message)

                final_path = target_path + '.npy'
                np.save(final_path, processed_images)

                result = f'Npy file saved at {final_path}'

        return result
