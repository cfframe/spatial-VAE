from __future__ import print_function, division

import argparse
import datetime
import logging
import numpy as np
import os
import sys

import spatial_vae.models as models
import spatial_vae.mrc as mrc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from pathlib import Path
from PIL import Image
from src.file_tools import FileTools
from src.logging_levels import LoggingLevels
from src.misc_tools import MiscTools
from src.plot_helper import PlotHelper
from src.result_columns import ResultColumns
from torch.autograd import Variable


def eval_minibatch(x, y, p_net, q_net, rotate=True, translate=True, dx_scale=0.1, theta_prior=np.pi,
                   augment_rotation=False, z_scale=1, use_cuda=False,
                   display_activation='sigmoid'):
    batch_size = y.size(0)
    channels = y.size(2)
    x = x.expand(batch_size, x.size(0), x.size(1))

    # Assumes square image of side n
    n = int(np.sqrt(y.size(1)))

    # augment training by randomly rotating images by offset
    offset = np.zeros(batch_size)
    y_rot = y
    if rotate and augment_rotation:
        # in order to encourage robustness of the inference network
        # randomly rotate the observed image before doing inference
        y_rot = y.clone()
        offset = np.random.uniform(0, 2 * np.pi, size=batch_size)
        if rotate < 1:
            r = np.random.binomial(1, p=rotate, size=batch_size)
            offset *= r
        for i in range(batch_size):
            # PIL doesn't support RGB images with values of type float - it assumes they are uint8 in range [0, 255]
            # ...So convert to that ...
            im = Image.fromarray((y[i].view(n, n, channels).cpu().numpy() * 255).astype(np.uint8))
            im = im.rotate(360 * offset[i] / 2 / np.pi, resample=Image.BICUBIC)
            # ... and reverse that conversion after doing the rotation
            im = torch.from_numpy(np.array(im, copy=False).astype(float) / 255).to(y.device)
            y_rot[i] = im.view(-1, channels)

    if use_cuda:
        y = y.cuda()
        y_rot = y_rot.cuda()

    # first do inference on the latent variables
    z_mu, z_logstd = q_net(y_rot.view(batch_size, -1))
    z_std = torch.exp(z_logstd)
    z_dim = z_mu.size(1)

    # draw samples from variational posterior to calculate
    # E[p(x|z)]
    # 1. r is a random sample from unit variance zero mean Gaus. dist.
    # We want a sample from z_mu and z_std - the above is the 'reparametrisation trick'
    r = Variable(x.data.new(batch_size, z_dim).normal_())
    z = z_std * r + z_mu

    kl_div = 0
    if rotate:
        # z[0] is the rotation
        theta_mu = z_mu[:, 0]
        theta_std = z_std[:, 0]
        theta_logstd = z_logstd[:, 0]
        theta = z[:, 0]
        z = z[:, 1:]
        z_mu = z_mu[:, 1:]
        z_std = z_std[:, 1:]
        z_logstd = z_logstd[:, 1:]

        if np.any(offset > 0):
            # invert the random rotation to reconstruct original with rotation offset
            offset = torch.from_numpy(offset).float().to(z.device)
            theta = theta + offset

        # calculate rotation matrix
        rot = Variable(theta.data.new(batch_size, 2, 2).zero_())
        rot[:, 0, 0] = torch.cos(theta)
        rot[:, 0, 1] = torch.sin(theta)
        rot[:, 1, 0] = -torch.sin(theta)
        rot[:, 1, 1] = torch.cos(theta)
        x = torch.bmm(x, rot)  # rotate coordinates by theta

        # use modified KL for rotation with no penalty on mean
        sigma = theta_prior
        kl_div = -theta_logstd + np.log(sigma) + theta_std ** 2 / 2 / sigma ** 2 - 0.5

    if translate:
        # z[0, 1] are the translations
        dx_mu = z_mu[:, :2]
        dx_std = z_std[:, :2]
        dx_logstd = z_logstd[:, :2]
        dx = z[:, :2] * dx_scale  # scale dx by standard deviation
        dx = dx.unsqueeze(1)
        z = z[:, 2:]

        x = x + dx  # translate coordinates

    z = z * z_scale

    # reconstruct
    y_hat = p_net(x.contiguous(), z)
    y_hat = y_hat.view(batch_size, -1, channels)

    size = y.size(1) * channels
    log_p_x_g_z = -F.binary_cross_entropy_with_logits(y_hat, y) * size

    # unit normal prior over z and translation
    z_kl = -z_logstd + 0.5 * z_std ** 2 + 0.5 * z_mu ** 2 - 0.5
    kl_div = kl_div + torch.sum(z_kl, 1)
    kl_div = kl_div.mean()

    elbo = log_p_x_g_z - kl_div

    y_hat = activate_y_for_display(y_hat, display_activation, channels)

    return elbo, log_p_x_g_z, kl_div, y_hat


def minibatch_for_display(x, y, q_net, p_net, rotate=True, translate=True, z_scale=1, use_cuda=False,
                          display_activation='sigmoid'):
    batch_size = y.size(0)
    channels = y.size(2)
    x = x.expand(batch_size, x.size(0), x.size(1))

    if use_cuda:
        y = y.cuda()

    # first do inference on the latent variables
    z_mu, z_logstd = q_net(y.view(batch_size, -1))
    z_std = torch.exp(z_logstd)
    z_dim = z_mu.size(1)

    # draw samples from variational posterior to calculate
    # E[p(x|z)]
    r = Variable(x.data.new(batch_size, z_dim).normal_())
    z = z_std * r + z_mu

    if rotate:
        # z[0] is the rotation so clear that
        z = z[:, 1:]

    if translate:
        # z[0, 1] are the translations, so clear these
        z = z[:, 2:]

    z = z * z_scale

    # reconstruct
    y_hat = p_net(x.contiguous(), z)
    y_hat = y_hat.view(batch_size, -1, channels)

    y_hat = activate_y_for_display(y_hat, display_activation, channels)

    return y_hat


def random_minibatch_generator(x, y, p_net, z_dim, z_scale=1, use_cuda=False,
                               display_activation='sigmoid'):
    batch_size = y.size(0)
    channels = y.size(2)
    x = x.expand(batch_size, x.size(0), x.size(1))

    if use_cuda:
        y = y.cuda()

    # draw samples from normal
    z = Variable(x.data.new(batch_size, z_dim).normal_())

    z = z * z_scale

    # reconstruct
    y_hat = p_net(x.contiguous(), z)
    y_hat = y_hat.view(batch_size, -1, channels)

    y_hat = activate_y_for_display(y_hat, display_activation, channels)

    return y_hat


def activate_y_for_display(y, display_activation, channels):

    if display_activation == 'sigmoid':
        y = torch.sigmoid(y)
    else:
        y = F.softmax(y, dim=channels - 1)

    return y


def train_epoch(iterator, x_coord, p_net, q_net, optim, rotate=True, translate=True,
                dx_scale=0.1, theta_prior=np.pi, augment_rotation=False, z_scale=1,
                epoch=1, num_epochs=1, train_images_len=1, use_cuda=False):
    p_net.train()
    q_net.train()

    count_accum = 0
    bce_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0

    for y, in iterator:
        batch_size = y.size(0)
        x = Variable(x_coord)
        y = Variable(y)

        elbo, log_p_x_g_z, kl_div, __ = eval_minibatch(x, y, p_net, q_net, rotate=rotate, translate=translate,
                                                      dx_scale=dx_scale, theta_prior=theta_prior,
                                                      augment_rotation=augment_rotation, z_scale=z_scale,
                                                      use_cuda=use_cuda)

        loss = -elbo
        loss.backward()
        optim.step()
        optim.zero_grad()

        elbo = elbo.item()
        bce_loss = -log_p_x_g_z.item()
        kl_loss = kl_div.item()

        count_accum += batch_size
        delta = batch_size * (bce_loss - bce_loss_accum)
        bce_loss_accum += delta / count_accum

        delta = batch_size * (elbo - elbo_accum)
        elbo_accum += delta / count_accum

        delta = batch_size * (kl_loss - kl_loss_accum)
        kl_loss_accum += delta / count_accum

        template = '# [{}/{}] training {:.1%}, ELBO={:.5f}, Error={:.5f}, KL={:.5f}'
        line = template.format(epoch + 1, num_epochs, count_accum / train_images_len, elbo_accum, bce_loss_accum,
                               kl_loss_accum)
        print(line, end='\r', file=sys.stderr)

    print(' ' * 80, end='\r', file=sys.stderr)
    return elbo_accum, bce_loss_accum, kl_loss_accum


def eval_model(iterator, x_coord, p_net, q_net, z_dim, rotate=True, translate=True,
               dx_scale=0.1, theta_prior=np.pi, z_scale=1, use_cuda=False,
               to_save_image_samples=False,
               image_dims=None, epoch='0',
               output_dir='outputs',
               save_label='',
               display_activation='sigmoid'):
    p_net.eval()
    q_net.eval()

    count_accum = 0
    bce_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0
    iteration_count = -1

    for y, in iterator:
        iteration_count += 1
        batch_size = y.size(0)
        x = Variable(x_coord)
        y = Variable(y)

        elbo, log_p_x_g_z, kl_div, y_hat = eval_minibatch(x, y, p_net, q_net, rotate=rotate, translate=translate,
                                                          dx_scale=dx_scale, theta_prior=theta_prior,
                                                          z_scale=z_scale, use_cuda=use_cuda,
                                                          display_activation=display_activation)

        elbo = elbo.item()
        gen_loss = -log_p_x_g_z.item()
        kl_loss = kl_div.item()

        count_accum += batch_size
        delta = batch_size * (gen_loss - bce_loss_accum)
        bce_loss_accum += delta / count_accum

        delta = batch_size * (elbo - elbo_accum)
        elbo_accum += delta / count_accum

        delta = batch_size * (kl_loss - kl_loss_accum)
        kl_loss_accum += delta / count_accum

        # Reconstruct and save images in first batch of each epoch, as a sample
        if iteration_count == 0 and to_save_image_samples and image_dims:
            y_display = minibatch_for_display(x, y, q_net, p_net, rotate=rotate, translate=translate,
                                              z_scale=z_scale, use_cuda=use_cuda,
                                              display_activation=display_activation)

            y_random = random_minibatch_generator(x, y, p_net, z_dim,
                                                  z_scale=z_scale, use_cuda=use_cuda,
                                                  display_activation=display_activation)

            MiscTools.export_batch_as_image(data=y_display,
                                            output='{}/images/{}_dis_{}.png'.format(output_dir, epoch, save_label),
                                            image_dims=image_dims, to_permute_for_channels=True)

            MiscTools.export_batch_as_image(data=y_hat,
                                            output='{}/images/{}_{}.png'.format(output_dir, epoch, save_label),
                                            image_dims=image_dims, to_permute_for_channels=True)

            MiscTools.export_batch_as_image(data=y_random,
                                            output='{}/images/{}_rnd_{}.png'.format(output_dir, epoch, save_label),
                                            image_dims=image_dims, to_permute_for_channels=True)

    return elbo_accum, bce_loss_accum, kl_loss_accum


def load_images(path):
    if path.endswith('mrc') or path.endswith('mrcs'):
        with open(path, 'rb') as f:
            content = f.read()
        images, __, __ = mrc.parse(content)
    elif path.endswith('npy'):
        images = np.load(path)
    return images


def galaxy_arguments():
    parser = argparse.ArgumentParser('Train spatial-VAE on galaxy datasets')

    parser.add_argument('train_path', help='path to training data')
    parser.add_argument('test_path', help='path to testing data')

    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--p-hidden-dim', type=int, default=500, help='dimension of hidden layers (default: 500)')
    parser.add_argument('--p-num-layers', type=int, default=2, help='number of hidden layers (default: 2)')
    parser.add_argument('--q-hidden-dim', type=int, default=5000, help='dimension of hidden layers (default: 5000)')
    parser.add_argument('--q-num-layers', type=int, default=2, help='number of hidden layers (default: 2)')
    parser.add_argument('-a', '--activation', choices=['tanh', 'relu'], default='tanh',
                        help='activation function (default: tanh)')
    parser.add_argument('--vanilla', action='store_true',
                        help='use the standard MLP generator architecture, decoding each pixel with an independent function. disables structured rotation and translation inference')
    parser.add_argument('--no-rotate', action='store_true', help='do not perform rotation inference')
    parser.add_argument('--no-translate', action='store_true', help='do not perform translation inference')

    parser.add_argument('--dx-scale', type=float, default=0.1,
                        help='standard deviation of translation latent variables (default: 0.1)')
    parser.add_argument('--theta-prior', type=float, default=np.pi,
                        help='standard deviation on rotation prior (default: pi)')

    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')

    parser.add_argument('--augment-rotation', action='store_true',
                        help='use data augmentation by randomly rotating images before inference')
    parser.add_argument('--z-delay', type=int, default=0,
                        help='delay using unstructured latent variables for this many training epochs (default: 0)')

    parser.add_argument('--save-prefix', help='path prefix to save models (optional)')
    parser.add_argument('--save-interval', default=10, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs (default: 100)')

    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--num-train-images', type=int, default=0, help='number of training images (default: 0 = all)')
    parser.add_argument('--val-split', type=int, default=50,
                        help='% split of training images for validation instead of training (default: 50)')
    parser.add_argument('--make-mono', action='store_true',
                        help='convert rbg images to monochrome')
    parser.add_argument('--logging-level', type=str, default='INFO',
                        help='logging level (default: INFO')
    parser.add_argument('-da', '--display-activation', choices=['sigmoid', 'softmax'], default='sigmoid',
                        help='activation used for image display purposes')
    parser.add_argument('--invert_colours', action='store_true',
                        help='convert images to negatives')

    return parser.parse_args()


def main():
    args = galaxy_arguments()
    dataset_type = 'galaxy'

    start_time, output_dir, trained_dir, images_dir, num_epochs, num_train_images, val_split, digits \
        = MiscTools.prep_pre_load_images(dataset_type, args)

    logging_level = LoggingLevels.logging_level(args.logging_level)

    logging.basicConfig(filename=f'{output_dir}/galaxy.log', format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%Y%m%d %H:%M:%S', filemode='w',
                        level=logging_level)
    logger = logging.getLogger()
    LoggingLevels.print_and_log_info(logger, 'Started')

    # load the images
    print('# loading data...', file=sys.stderr)
    images_train = np.load(args.train_path)
    images_val = np.load(args.test_path)

    channels = 3
    if args.make_mono:
        # Convert to grayscale. Not for human perception, so just take a simple mean across the channels
        # Shape is (image-count, rows, columns, channels)
        images_train = np.mean(images_train, axis=3)
        channels = 1

    np.random.shuffle(images_train)
    # num_train_images enables a quick litmus test with fewer images
    if num_train_images > 0:
        images_train = images_train[:num_train_images]
        images_val = images_val[:num_train_images]

    # COMMENT OUT below - revert back to using test set for validation
    # num_val_images = int(val_split * len(images_train) / 100)
    # images_val = images_train[:num_val_images]
    # images_train = images_train[num_val_images:]

    image_rows, image_cols = images_train.shape[1:3]
    image_dims = [image_rows, image_cols]

    images_train = torch.from_numpy(images_train).float() / 255
    images_val = torch.from_numpy(images_val).float() / 255
    if args.invert_colours:
        images_train = 1 - images_train
        images_val = 1 - images_val

    y_train = images_train.view(-1, image_rows * image_cols, channels)
    y_val = images_val.view(-1, image_rows * image_cols, channels)

    # # x coordinate array
    xgrid = np.linspace(-1, 1, image_cols)
    ygrid = np.linspace(1, -1, image_rows)
    x0, x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    # # set the device
    device = args.device
    use_cuda = (device != -1) and torch.cuda.is_available()
    if device >= 0:
        torch.cuda.set_device(device)
        LoggingLevels.print_and_log_info(logger, f'# using CUDA device: {device}')
    else:
        LoggingLevels.print_and_log_info(logger, f'# using CPU')

    augment_rotation = args.augment_rotation
    if use_cuda:
        y_train = y_train.cuda()
        y_val = y_val.cuda()
        x_coord = x_coord.cuda()

    data_train = torch.utils.data.TensorDataset(y_train)
    data_val = torch.utils.data.TensorDataset(y_val)

    z_dim = args.z_dim
    print('# training with z-dim:', z_dim, file=sys.stderr)

    num_layers = args.p_num_layers
    hidden_dim = args.p_hidden_dim

    # default activation
    activation = nn.LeakyReLU
    if args.activation == 'tanh':
        activation = nn.Tanh
    elif args.activation == 'relu':
        activation = nn.LeakyReLU

    # display activation
    display_activation = args.display_activation

    # Build models
    if args.vanilla:
        print('# using the vanilla MLP generator architecture', file=sys.stderr)
        n_out = channels * image_rows * image_cols
        p_net = models.VanillaGenerator(n_out, z_dim, hidden_dim, num_layers=num_layers,
                                        activation=activation)
        inf_dim = z_dim
        rotate = False
        translate = False
    else:
        n_out = channels
        print('# using the spatial generator architecture', file=sys.stderr)
        rotate = not args.no_rotate
        translate = not args.no_translate
        inf_dim = z_dim
        if rotate:
            print('# spatial-VAE with rotation inference', file=sys.stderr)
            inf_dim += 1
        if translate:
            print('# spatial-VAE with translation inference', file=sys.stderr)
            inf_dim += 2
        p_net = models.SpatialGenerator(z_dim, hidden_dim, n_out=n_out, num_layers=num_layers,
                                        activation=activation)

    num_layers = args.q_num_layers
    hidden_dim = args.q_hidden_dim
    q_net = models.InferenceNetwork(channels * image_rows * image_cols, inf_dim, hidden_dim, num_layers=num_layers, activation=activation)

    if use_cuda:
        p_net.cuda()
        q_net.cuda()

    MiscTools.save_model_specs_to_file(output_dir, [p_net, q_net])

    dx_scale = args.dx_scale
    theta_prior = args.theta_prior

    print('# using priors: theta={}, dx={}'.format(theta_prior, dx_scale), file=sys.stderr)

    train_images_len = len(images_train)
    params = list(p_net.parameters()) + list(q_net.parameters())

    lr = args.learning_rate
    optim = torch.optim.Adam(params, lr=lr)
    minibatch_size = args.minibatch_size

    train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size, shuffle=True)
    val_iterator = torch.utils.data.DataLoader(data_val, batch_size=minibatch_size, shuffle=False)

    path_prefix = args.save_prefix
    save_label = MiscTools.save_label(args)
    save_interval = args.save_interval

    z_delay = args.z_delay

    MiscTools.sample_images(iterator=val_iterator, image_dims=image_dims, output_dir=output_dir,
                            save_label=save_label)

    # Initialise results bins
    output = sys.stdout
    header_parts = '\t'.join(['Epoch', 'ELBO', 'BCE loss', 'KL'])
    print(header_parts, file=output)

    train_results = []
    val_results = []
    train_results_for_file = [header_parts]
    val_results_for_file = [header_parts]

    for epoch in range(num_epochs):
        z_scale = 1
        epoch_str = str(epoch + 1).zfill(digits)

        if epoch < z_delay:
            z_scale = 0

        elbo_accum, bce_loss_accum, kl_loss_accum = train_epoch(train_iterator, x_coord, p_net, q_net,
                                                                optim, rotate=rotate, translate=translate,
                                                                dx_scale=dx_scale, theta_prior=theta_prior,
                                                                augment_rotation=augment_rotation,
                                                                z_scale=z_scale,
                                                                epoch=epoch, num_epochs=num_epochs,
                                                                train_images_len=train_images_len,
                                                                use_cuda=use_cuda)

        train_loss = [epoch, elbo_accum, bce_loss_accum, kl_loss_accum]
        train_results.append(train_loss)
        line = '\t'.join(list(map(str, train_loss)))
        train_results_for_file.append(line)
        print(line, file=output)
        output.flush()

        # evaluate on the validation set
        to_save_image_samples = ((epoch + 1) % save_interval == 0)
        elbo_accum, bce_loss_accum, kl_loss_accum = eval_model(val_iterator, x_coord, p_net, q_net,
                                                               z_dim=z_dim,
                                                               rotate=rotate, translate=translate,
                                                               dx_scale=dx_scale, theta_prior=theta_prior,
                                                               z_scale=z_scale,
                                                               use_cuda=use_cuda,
                                                               to_save_image_samples=to_save_image_samples,
                                                               image_dims=image_dims,
                                                               epoch=epoch_str, output_dir=output_dir,
                                                               save_label=save_label,
                                                               display_activation=display_activation
                                                               )

        val_loss = [epoch, elbo_accum, bce_loss_accum, kl_loss_accum]
        val_results.append(val_loss)
        line = '\t'.join(list(map(str, val_loss)))
        val_results_for_file.append(line)
        print(line, file=output)
        output.flush()

    # save the models
    # Previously run within epochs cycle (using save_interval) but these can get large. May revert back to that
    # at some stage. Swapped epochs for num_epochs-1 and save_interval for 1.
    MiscTools.save_trained_models(path_prefix, num_epochs - 1, digits, 1, trained_dir, p_net, q_net, use_cuda)

    PlotHelper.basic_run_plot(train_results, val_results, output_dir=os.path.join(output_dir, 'images'))
    MiscTools.save_results(output_dir=output_dir,
                           train_results=train_results_for_file,
                           val_results=val_results_for_file)

    end_time = datetime.datetime.now()
    print(f"End : {end_time.strftime('%y%m%d_%H%M%S')}")

    elapsed_time = (end_time - start_time)
    LoggingLevels.print_and_log_info(logger, f'Elapsed time: {elapsed_time}')
    LoggingLevels.print_and_log_info(logger, f'Finished')

    for hndlr in logger.handlers:
        hndlr.flush()
        hndlr.close()

    # Create archive of output directory - INCLUDING log
    FileTools.make_datetime_named_archive(output_dir, 'zip', output_dir)


if __name__ == '__main__':
    main()
