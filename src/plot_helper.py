# plot_helper.py

import matplotlib.pyplot as plt
import os


class PlotHelper:
    """Utilities for plotting graphs"""

    @staticmethod
    def basic_train_val_plot_and_save(title, train_data, validation_data, legend_location, output_dir):
        """Plot pairs of datasets

        :param title: str -- figure title, used as base for saved file name
        :param train_data: dataset -- training results
        :param validation_data: dataset -- validation results
        :param legend_location: str -- location of legend on the Figure e.g. 'upper right'
        :param output_dir: str -- target directory for saving plot
        """

        plt.plot(train_data, color='b', label='Training')
        plt.plot(validation_data, color='g', label='Validation')
        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc=legend_location)
        plt.grid()

        target_path = os.path.join(output_dir, title.replace(' ', '_')+'.svg')
        # If use plt.show() before saving, then saved figure is blank. Works ok other way round.
        plt.savefig(target_path)

        return plt

    @staticmethod
    def basic_run_plot(elbo_train, elbo_val, kl_train, kl_val, bce_loss_train, bce_loss_val, output_dir):

        PlotHelper.basic_train_val_plot_and_save(
            # style='seaborn',
            title='ELBO',
            train_data=elbo_train,
            validation_data=elbo_val,
            legend_location='lower right',
            output_dir=output_dir)

        PlotHelper.basic_train_val_plot_and_save(
            # style='seaborn',
            title='KL Divergence',
            train_data=kl_train,
            validation_data=kl_val,
            legend_location='upper right',
            output_dir=output_dir)

        PlotHelper.basic_train_val_plot_and_save(
            # style='seaborn',
            title='BCE Loss',
            train_data=bce_loss_train,
            validation_data=bce_loss_val,
            legend_location='upper right',
            output_dir=output_dir)
