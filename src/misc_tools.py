# misc_tools.py

import datetime
import numpy as np
import os
import shutil

from src.file_tools import FileTools
from pathlib import Path
from torchvision.utils import save_image


class MiscTools:
    """Miscellaneous utilities for this repo"""

    @staticmethod
    def export_batch_as_image(data, output, image_dims, to_permute_for_channels=True):
        # Re-cast data view to original image dimensions
        images = data.view(data.size()[0], *image_dims, -1)
        if to_permute_for_channels:
            images = images.permute(0, 3, 1, 2)
        # Assume square of images, so no. of rows is square root of number of images
        rows = int(data.size()[0] ** 0.5)
        save_image(images.cpu(), output, nrow=rows, padding=3, pad_value=0.5)

    @staticmethod
    def sample_images(iterator, image_dims=None, name='sample', prefix='', output_dir='outputs'):
        for y, in iterator:
            MiscTools.export_batch_as_image(data=y, output='{}/images/{}_{}.png'.format(output_dir, prefix, name),
                                            image_dims=image_dims, to_permute_for_channels=True)
            return


    @staticmethod
    def prep_pre_load_images(args, dataset_type):
        # Final warning
        if not args.delete_outputs_at_start \
                and input(
            'WARNING This will clear the outputs directory if it exists. Continue (y/n and Enter)?').lower() == 'n':
            quit()

        start_time = datetime.datetime.now()
        print(f"Start : {start_time.strftime('%y%m%d_%H%M%S')}")

        output_dir = 'outputs_{}'.format(dataset_type)

        trained_dir = os.path.join(output_dir, 'trained')
        images_dir = os.path.join(output_dir, 'images')
        FileTools.ensure_empty_sub_directory(trained_dir)
        FileTools.ensure_empty_sub_directory(images_dir)

        # Save list of arguments with values
        FileTools.save_command_args_to_file(script=os.path.basename(__file__), args=vars(args),
                                            save_path=os.path.join(output_dir, 'command.txt'))

        num_epochs = args.num_epochs
        num_train_images = args.num_train_images
        val_split = args.val_split

        digits = int(np.log10(num_epochs)) + 1

        return start_time, output_dir, trained_dir, images_dir, num_epochs, num_train_images, val_split, digits

    @staticmethod
    def save_results(output_dir, train_results, val_results):
        train_results_path = os.path.join(output_dir, 'train.txt')
        val_results_path = os.path.join(output_dir, 'val.txt')

        with open(train_results_path, 'w') as train_file:
            print('\n'.join(train_results), file=train_file)

        with open(val_results_path, 'w') as val_file:
            print('\n'.join(val_results), file=val_file)

