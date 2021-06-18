# file_tools.py

import datetime
from pathlib import Path
import os
import shutil


class FileTools:
    """Utilities for managing data from and to files"""

    @staticmethod
    def ensure_empty_directory(dir_path: str) -> str:
        """If path does not exist, create it. If it does exist, empty it.

        Keyword arguments:
        :param dir_path: root directory path
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
    def save_command_args_to_file(script: str, args: dict, save_path: str):
        """Save arguments and their values to file. Expects args of type dict, so use vars(args) as input.

        Keyword arguments:

        :param script: python script being run
        :param args: dict, full arguments list
        :param save_path: str, path to file
        """
        if not script.endswith('.py'):
            script += '.py'

        command = 'python {}'.format(script)
        parts = [command]
        lines = []

        for k, v in args.items():
            parts.append('--{} {}'.format(k, v))
            lines.append('{}={}'.format(k, v))

        command_line = ' '.join(parts)
        lines.insert(0, command_line)
        content = '\n'.join(lines)

        with open(save_path, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
            print('Output saved to {}.'.format(save_path))

