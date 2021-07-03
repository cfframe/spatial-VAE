# plot_helper.py

import matplotlib.pyplot as plt
import numpy as np
import os

from src.result_columns import ResultColumns


class PlotHelper:
    """Utilities for plotting graphs"""

    @staticmethod
    def basic_train_val_plot_and_save(title, y_label, train_data, validation_data, legend_location, output_dir):
        """Plot pairs of datasets

        :param title: str -- figure title, used as base for saved file name
        :param y_label: str -- y-axis label
        :param train_data: dataset -- training results
        :param validation_data: dataset -- validation results
        :param legend_location: str -- location of legend on the Figure e.g. 'upper right'
        :param output_dir: str -- target directory for saving plot
        """
        plt.plot(train_data, color='b', label='Training')
        plt.plot(validation_data, color='g', label='Validation')
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc=legend_location)
        plt.grid()

        target_path = os.path.join(output_dir, title.replace(' ', '_')+'.svg')
        # If use plt.show() before saving, then saved figure is blank. Works ok other way round.
        plt.savefig(target_path)

        # plt.show()

    @staticmethod
    def basic_run_plot(train_results, val_results, output_dir):
        train_arr = np.asarray(train_results)
        val_arr = np.asarray(val_results)

        plt.figure()
        PlotHelper.basic_train_val_plot_and_save(
            # style='seaborn',
            title='ELBO',
            y_label='ELBO',
            train_data=train_arr[:, ResultColumns.ELBO],
            validation_data=val_arr[:, ResultColumns.ELBO],
            legend_location='lower right',
            output_dir=output_dir)

        plt.figure()
        PlotHelper.basic_train_val_plot_and_save(
            # style='seaborn',
            title='KL Divergence',
            y_label='KL Divergence',
            train_data=train_arr[:, ResultColumns.KL],
            validation_data=val_arr[:, ResultColumns.KL],
            legend_location='lower right',
            output_dir=output_dir)

        plt.figure()
        PlotHelper.basic_train_val_plot_and_save(
            # style='seaborn',
            title='BCE Loss',
            y_label='BCE Loss',
            train_data=train_arr[:, ResultColumns.BCE],
            validation_data=val_arr[:, ResultColumns.BCE],
            legend_location='upper right',
            output_dir=output_dir)
