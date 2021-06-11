from __future__ import print_function, division

import numpy as np
import os
import pandas as pd
import sys

from PIL import Image
from src.file_tools import FileTools
import spatial_vae.models as models
import spatial_vae.mrc as mrc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torchvision
from torchvision.utils import save_image

from tqdm import tqdm


def eval_minibatch(x, y, p_net, q_net, rotate=True, translate=True, dx_scale=0.1, theta_prior=np.pi,
                   augment_rotation=False, z_scale=1, use_cuda=False):

    b = y.size(0)
    x = x.expand(b, x.size(0), x.size(1))
    n = int(np.sqrt(y.size(1)))

    # augment training by randomly rotating images by offset
    offset = np.zeros(b)
    y_rot = y
    if rotate and augment_rotation:
        # in order to encourage robustness of the inference network
        # randomly rotate the observed image before doing inference
        y_rot = y.clone()
        offset = np.random.uniform(0, 2*np.pi, size=b)
        if rotate < 1:
            r = np.random.binomial(1, p=rotate, size=b)
            offset *= r
        for i in range(b):
            im = Image.fromarray(y[i].view(n, n, 3).cpu().numpy())
            im = im.rotate(360*offset[i]/2/np.pi, resample=Image.BICUBIC)
            im = torch.from_numpy(np.array(im, copy=False)).to(y.device)
            y_rot[i] = im.view(-1, 3)

    if use_cuda:
        y = y.cuda()
        y_rot = y_rot.cuda()

    # first do inference on the latent variables
    z_mu, z_logstd = q_net(y_rot.view(b, -1))
    z_std = torch.exp(z_logstd)
    z_dim = z_mu.size(1)

    # draw samples from variational posterior to calculate
    # E[p(x|z)]
    r = Variable(x.data.new(b, z_dim).normal_())
    z = z_std*r + z_mu
    
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
        rot = Variable(theta.data.new(b, 2, 2).zero_())
        rot[:, 0, 0] = torch.cos(theta)
        rot[:, 0, 1] = torch.sin(theta)
        rot[:, 1, 0] = -torch.sin(theta)
        rot[:, 1, 1] = torch.cos(theta)
        x = torch.bmm(x, rot) # rotate coordinates by theta

        # use modified KL for rotation with no penalty on mean
        sigma = theta_prior
        kl_div = -theta_logstd + np.log(sigma) + theta_std**2/2/sigma**2 - 0.5

    if translate:
        # z[0, 1] are the translations
        dx_mu = z_mu[:, :2]
        dx_std = z_std[:, :2]
        dx_logstd = z_logstd[:, :2]
        dx = z[:, :2]*dx_scale  # scale dx by standard deviation
        dx = dx.unsqueeze(1)
        z = z[:, 2:]

        x = x + dx  # translate coordinates

    z = z*z_scale

    # reconstruct
    y_hat = p_net(x.contiguous(), z)
    y_hat = y_hat.view(b, -1, 3)

    size = y.size(1)*3
    log_p_x_g_z = -F.binary_cross_entropy_with_logits(y_hat, y) * size

    # unit normal prior over z and translation
    z_kl = -z_logstd + 0.5*z_std**2 + 0.5*z_mu**2 - 0.5
    kl_div = kl_div + torch.sum(z_kl, 1)
    kl_div = kl_div.mean()
    
    elbo = log_p_x_g_z - kl_div

    return elbo, log_p_x_g_z, kl_div, y_hat


def train_epoch(iterator, x_coord, p_net, q_net, optim, rotate=True, translate=True,
                dx_scale=0.1, theta_prior=np.pi, augment_rotation=False, z_scale=1,
                epoch=1, num_epochs=1, N=1, use_cuda=False):
    p_net.train()
    q_net.train()

    count_accum = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0

    for y, in iterator:
        batch_size = y.size(0)
        x = Variable(x_coord)
        y = Variable(y)

        elbo, log_p_x_g_z, kl_div, _ = eval_minibatch(x, y, p_net, q_net, rotate=rotate, translate=translate,
                                                   dx_scale=dx_scale, theta_prior=theta_prior,
                                                   augment_rotation=augment_rotation, z_scale=z_scale,
                                                   use_cuda=use_cuda)

        loss = -elbo
        loss.backward()
        optim.step()
        optim.zero_grad()

        elbo = elbo.item()
        gen_loss = -log_p_x_g_z.item()
        kl_loss = kl_div.item()

        count_accum += batch_size
        delta = batch_size*(gen_loss - gen_loss_accum)
        gen_loss_accum += delta/count_accum

        delta = batch_size*(elbo - elbo_accum)
        elbo_accum += delta/count_accum

        delta = batch_size*(kl_loss - kl_loss_accum)
        kl_loss_accum += delta/count_accum

        template = '# [{}/{}] training {:.1%}, ELBO={:.5f}, Error={:.5f}, KL={:.5f}'
        line = template.format(epoch+1, num_epochs, count_accum/N, elbo_accum, gen_loss_accum, kl_loss_accum)
        print(line, end='\r', file=sys.stderr)

    print(' '*80, end='\r', file=sys.stderr)
    return elbo_accum, gen_loss_accum, kl_loss_accum


def eval_model(iterator, x_coord, p_net, q_net, rotate=True, translate=True,
               dx_scale=0.1, theta_prior=np.pi, z_scale=1, use_cuda=False,
               to_save_image_samples=False,
               image_dims=None, epoch='0'):

    p_net.eval()
    q_net.eval()

    count_accum = 0
    gen_loss_accum = 0
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
                                                          z_scale=z_scale, use_cuda=use_cuda)

        elbo = elbo.item()
        gen_loss = -log_p_x_g_z.item()
        kl_loss = kl_div.item()

        count_accum += batch_size
        delta = batch_size*(gen_loss - gen_loss_accum)
        gen_loss_accum += delta/count_accum

        delta = batch_size*(elbo - elbo_accum)
        elbo_accum += delta/count_accum

        delta = batch_size*(kl_loss - kl_loss_accum)
        kl_loss_accum += delta/count_accum

        # Reconstruct and save images in first batch of each epoch, as a sample
        if iteration_count == 0 and to_save_image_samples and image_dims:
            export_batch_as_image(data=y_hat, output='outputs/images/{}_output.png'.format(epoch),
                                  image_dims=image_dims)

    return elbo_accum, gen_loss_accum, kl_loss_accum


def load_images(path):
    if path.endswith('mrc') or path.endswith('mrcs'):
        with open(path, 'rb') as f:
            content = f.read()
        images, _, _ = mrc.parse(content)
    elif path.endswith('npy'):
        images = np.load(path)
    return images


def export_batch_as_image(data, output, image_dims):
    # Re-cast data view
    images = data.view(data.size()[0], *image_dims, -1)
    images = images.permute(0, 3, 1, 2)
    # need torch.cat?
    # Assume square of images, so no. of rows is square root of number of images
    rows = int(data.size()[0] ** 0.5)
    save_image(images.cpu(), output, nrow=rows)


def sample_images(iterator, image_dims=None, name='sample', prefix=''):
    for y, in iterator:
        export_batch_as_image(data=y, output='outputs/images/{}_{}.png'.format(prefix, name), image_dims=image_dims)
        return


def main():
    import argparse

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

    parser.add_argument('--augment-rotation', action='store_true', help='use data augmentation by randomly rotating images before inference')
    parser.add_argument('--z-delay', type=int, default=0, help='delay using unstructured latent variables for this many training epochs (default: 0)')

    parser.add_argument('--save-prefix', help='path prefix to save models (optional)')
    parser.add_argument('--save-interval', default=10, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs (default: 100)')

    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--num-train-images', type=int, default=0, help='number of training images (default: 0 = all)')
    parser.add_argument('--val-split', type=int, default=50, help='% split of training images for validation instead of training (default: 50)')

    args = parser.parse_args()

    output_dir = 'outputs'
    trained_dir = os.path.join(output_dir, 'trained')
    images_dir = os.path.join(output_dir, 'images')
    FileTools.ensure_empty_sub_directory(trained_dir)
    FileTools.ensure_empty_sub_directory(images_dir)

    num_epochs = args.num_epochs
    num_train_images = args.num_train_images
    val_split = args.val_split

    digits = int(np.log10(num_epochs)) + 1

    # load the images
    print('# loading data...', file=sys.stderr)
    images_train = np.load(args.train_path)
    # images_test = np.load(args.test_path)

    # cff Add-in to enable a quick litmus test
    np.random.shuffle(images_train)
    if num_train_images > 0:
        images_train = images_train[:num_train_images]

    num_val_images = int(val_split * len(images_train) / 100)
    images_val = images_train[:num_val_images]
    images_train = images_train[num_val_images:]

    n, m = images_train.shape[1:3]
    image_dims = [n, m]

    # x coordinate array
    xgrid = np.linspace(-1, 1, m)
    ygrid = np.linspace(1, -1, n)
    x0, x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    images_train = torch.from_numpy(images_train).float()/255
    images_val = torch.from_numpy(images_val).float()/255
    y_train = images_train.view(-1, n*m, 3)
    y_val = images_val.view(-1, n*m, 3)

    # # set the device
    device = args.device
    use_cuda = (device != -1) and torch.cuda.is_available()
    if device >= 0:
        torch.cuda.set_device(device)
        print('# using CUDA device:', device, file=sys.stderr)

    augment_rotation = args.augment_rotation
    if use_cuda:
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

    if args.vanilla:
        print('# using the vanilla MLP generator architecture', file=sys.stderr)
        n_out = 3*n*m
        p_net = models.VanillaGenerator(n_out, z_dim, hidden_dim, num_layers=num_layers, activation=activation)
        inf_dim = z_dim
        rotate = False
        translate = False
    else:
        n_out = 3 
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
        p_net = models.SpatialGenerator(z_dim, hidden_dim, n_out=n_out, num_layers=num_layers, activation=activation)

    num_layers = args.q_num_layers
    hidden_dim = args.q_hidden_dim
    q_net = models.InferenceNetwork(3*n*m, inf_dim, hidden_dim, num_layers=num_layers, activation=activation)

    if use_cuda:
        p_net.cuda()
        q_net.cuda()

    dx_scale = args.dx_scale
    theta_prior = args.theta_prior

    print('# using priors: theta={}, dx={}'.format(theta_prior, dx_scale), file=sys.stderr)

    N = len(data_train)
    params = list(p_net.parameters()) + list(q_net.parameters())

    lr = args.learning_rate
    optim = torch.optim.Adam(params, lr=lr)
    minibatch_size = args.minibatch_size

    train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size, shuffle=True)
    val_iterator = torch.utils.data.DataLoader(data_val, batch_size=minibatch_size, shuffle=False)

    output = sys.stdout
    print('\t'.join(['Epoch', 'Split', 'ELBO', 'Error', 'KL']), file=output)

    path_prefix = args.save_prefix
    save_interval = args.save_interval

    z_delay = args.z_delay
    for epoch in range(num_epochs):
        z_scale = 1
        epoch_str = str(epoch + 1).zfill(digits)

        if epoch < z_delay:
            z_scale = 0

        elbo_accum, gen_loss_accum, kl_loss_accum = train_epoch(train_iterator, x_coord, p_net, q_net,
                                                                optim, rotate=rotate, translate=translate,
                                                                dx_scale=dx_scale, theta_prior=theta_prior,
                                                                augment_rotation=augment_rotation,
                                                                z_scale=z_scale,
                                                                epoch=epoch, num_epochs=num_epochs, N=N,
                                                                use_cuda=use_cuda)

        line = '\t'.join([str(epoch+1), 'train', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        print(line, file=output)
        output.flush()

        # sample from / evaluate on the validation set
        sample_images(iterator=val_iterator, image_dims=image_dims)

        to_save_image_samples = ((epoch+1) % save_interval == 0)
        elbo_accum, gen_loss_accum, kl_loss_accum = eval_model(val_iterator, x_coord, p_net,
                                                               q_net, rotate=rotate, translate=translate,
                                                               dx_scale=dx_scale, theta_prior=theta_prior,
                                                               z_scale=z_scale,
                                                               use_cuda=use_cuda,
                                                               to_save_image_samples=to_save_image_samples,
                                                               image_dims=image_dims,
                                                               epoch=epoch_str
                                                               )
        line = '\t'.join([str(epoch+1), 'validation', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        print(line, file=output)
        output.flush()

        # save the models
        if path_prefix is not None and (epoch+1) % save_interval == 0:

            path = os.path.join(trained_dir, path_prefix + '_generator_epoch{}.sav'.format(epoch_str))
            p_net.eval().cpu()
            torch.save(p_net, path)

            path = os.path.join(trained_dir, path_prefix + '_inference_epoch{}.sav'.format(epoch_str))
            q_net.eval().cpu()
            torch.save(q_net, path)

            if use_cuda:
                p_net.cuda()
                q_net.cuda()


if __name__ == '__main__':
    main()

