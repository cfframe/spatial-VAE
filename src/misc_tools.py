# misc_tools.py

import datetime
import numpy as np
import os
import shutil
import torch

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
    def prep_pre_load_images(script, dataset_type, args):
        # Final warning
        if input('WARNING Will clear the outputs directory if it exists. Continue (y/n and Enter)?').lower() == 'n':
            quit()

        start_time = datetime.datetime.now()
        print(f"Start : {start_time.strftime('%y%m%d_%H%M%S')}")

        output_dir = 'outputs_{}'.format(dataset_type)

        trained_dir = os.path.join(output_dir, 'trained')
        images_dir = os.path.join(output_dir, 'images')
        FileTools.ensure_empty_directory(output_dir)
        FileTools.ensure_empty_directory(trained_dir)
        FileTools.ensure_empty_directory(images_dir)

        # Save list of arguments with values
        FileTools.save_command_args_to_file(script=script, args=vars(args),
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
            print('\n'.join(list(map(str, train_results))), file=train_file)

        with open(val_results_path, 'w') as val_file:
            print('\n'.join(list(map(str, val_results))), file=val_file)

    @staticmethod
    def save_trained_models(path_prefix, epoch, digits, save_interval, trained_dir, p_net, q_net, use_cuda):

        if path_prefix is not None and (epoch+1) % save_interval == 0:
            epoch_str = str(epoch + 1).zfill(digits)

            path = os.path.join(trained_dir, path_prefix + '_generator_epoch{}.sav'.format(epoch_str))
            p_net.eval().cpu()
            torch.save(p_net, path)

            path = os.path.join(trained_dir, path_prefix + '_inference_epoch{}.sav'.format(epoch_str))
            q_net.eval().cpu()
            torch.save(q_net, path)

            # Revert to cuda
            if use_cuda:
                p_net.cuda()
                q_net.cuda()

    @staticmethod
    def save_model_specs_to_file(outputs_dir, models):
        path = os.path.join(outputs_dir, 'models.txt')

        with open(path, 'w') as file:
            for model in models:
                print(model, file=file)

