from __future__ import print_function, division

import argparse
import datetime
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torchvision

import spatial_vae.models as models
from src.misc_tools import MiscTools


def eval_minibatch(x, y, p_net, q_net, rotate=True, translate=True, dx_scale=0.1, theta_prior=np.pi, use_cuda=False):
    batch_size = y.size(0)
    x = x.expand(batch_size, x.size(0), x.size(1))

    # first do inference on the latent variables
    if use_cuda:
        y = y.cuda()

    z_mu, z_logstd = q_net(y)
    z_std = torch.exp(z_logstd)
    z_dim = z_mu.size(1)

    # draw samples from variational posterior to calculate
    # E[p(x|z)]
    r = Variable(x.data.new(batch_size, z_dim).normal_())
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

        # calculate rotation matrix
        rot = Variable(theta.data.new(batch_size, 2, 2).zero_())
        rot[:, 0, 0] = torch.cos(theta)
        rot[:, 0, 1] = torch.sin(theta)
        rot[:, 1, 0] = -torch.sin(theta)
        rot[:, 1, 1] = torch.cos(theta)
        x = torch.bmm(x, rot)  # rotate coordinates by theta

        # calculate the KL divergence term
        sigma = theta_prior
        kl_div = -theta_logstd + np.log(sigma) + (theta_std**2 + theta_mu**2)/2/sigma**2 - 0.5

    if translate:
        # z[0, 1] are the translations
        dx_mu = z_mu[:, :2]
        dx_std = z_std[:, :2]
        dx_logstd = z_logstd[:, :2]
        dx = z[:, :2]*dx_scale  # scale dx by standard deviation
        dx = dx.unsqueeze(1)
        z = z[:, 2:]

        x = x + dx  # translate coordinates

    # reconstruct
    y_hat = p_net(x.contiguous(), z)
    y_hat = y_hat.view(batch_size, -1)

    size = y.size(1)
    log_p_x_g_z = -F.binary_cross_entropy_with_logits(y_hat, y)*size

    # unit normal prior over z and translation
    z_kl = -z_logstd + 0.5*z_std**2 + 0.5*z_mu**2 - 0.5
    kl_div = kl_div + torch.sum(z_kl, 1)
    kl_div = kl_div.mean()
    
    elbo = log_p_x_g_z - kl_div

    return elbo, log_p_x_g_z, kl_div, y_hat


def train_epoch(iterator, x_coord, p_net, q_net, optim, rotate=True, translate=True,
                dx_scale=0.1, theta_prior=np.pi,
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
               dx_scale=0.1, theta_prior=np.pi, use_cuda=False,
               to_save_image_samples=False,
               image_dims=None, epoch='0',
               output_dir='outputs'):

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
                                                          use_cuda=use_cuda)

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
            MiscTools.export_batch_as_image(data=y_hat,
                                            output='{}/images/{}_output.png'.format(output_dir, epoch),
                                            image_dims=image_dims, to_permute_for_channels=True)

    return elbo_accum, gen_loss_accum, kl_loss_accum


def mnist_arguments():
    parser = argparse.ArgumentParser('Train spatial-VAE on MNIST datasets')

    parser.add_argument('--dataset', choices=['mnist', 'mnist-rotated', 'mnist-rotated-translated'],
                        default='mnist-rotated-translated',
                        help='which MNIST datset to train/validate on (default: mnist-rotated-translated)')

    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--hidden-dim', type=int, default=500, help='dimension of hidden layers (default: 512)')
    parser.add_argument('--num-layers', type=int, default=2, help='number of hidden layers (default: 1)')
    parser.add_argument('-a', '--activation', choices=['tanh', 'relu'], default='tanh',
                        help='activation function (default: tanh)')

    parser.add_argument('--vanilla', action='store_true',
                        help='use the standard MLP generator architecture, decoding each pixel with an independent function. disables structured rotation and translation inference')
    parser.add_argument('--no-rotate', action='store_true', help='do not perform rotation inference')
    parser.add_argument('--no-translate', action='store_true', help='do not perform translation inference')

    parser.add_argument('--dx-scale', type=float, default=0.1,
                        help='standard deviation of translation latent variables (default: 0.1)')
    parser.add_argument('--theta-prior', type=float, default=(np.pi / 4)**2,
                        help='standard deviation on rotation prior (default: square of pi/4)')

    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')

    parser.add_argument('--save-prefix', help='path prefix to save models (optional)')
    parser.add_argument('--save-interval', default=10, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs (default: 100)')

    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--num-train-images', type=int, default=0, help='number of training images (default: 0 = all)')
    parser.add_argument('--val-split', type=int, default=50, help='% split of training images for validation instead of training (default: 50)')
    parser.add_argument('--delete_outputs_at_start', action='store_true', help='delete Outputs directory content at start')

    return parser.parse_args()


def main():

    args = mnist_arguments()
    dataset_type = 'mnist'

    start_time, output_dir, trained_dir, images_dir, num_epochs, num_train_images, val_split, digits\
        = MiscTools.prep_pre_load_images(args, dataset_type)

    # load the images
    if args.dataset == 'mnist':
        print('# training on MNIST', file=sys.stderr)
        images_train = torchvision.datasets.MNIST('data/mnist/', train=True, download=True)
        images_test = torchvision.datasets.MNIST('data/mnist/', train=False, download=True)

        array = np.zeros((len(images_train), 28, 28), dtype=np.uint8)
        for i in range(len(images_train)):
            array[i] = np.array(images_train[i][0], copy=False)
        images_train = array

        array = np.zeros((len(images_test), 28, 28), dtype=np.uint8)
        for i in range(len(images_test)):
            array[i] = np.array(images_test[i][0], copy=False)
        images_test = array

    elif args.dataset == 'mnist-rotated':
        print('# training on rotated MNIST', file=sys.stderr)
        images_train = np.load('data/mnist_rotated/images_train.npy')
        images_test = np.load('data/mnist_rotated/images_test.npy')

    else:
        print('# training on rotated and translated MNIST', file=sys.stderr)
        images_train = np.load('data/mnist_rotated_translated/images_train.npy')
        images_test = np.load('data/mnist_rotated_translated/images_test.npy')

    n = m = 28
    image_dims = [n, m]

    images_train = torch.from_numpy(images_train).float()/255
    images_test = torch.from_numpy(images_test).float()/255
    y_train = images_train.view(-1, n*m)
    y_test = images_test.view(-1, n*m)

    # x coordinate array
    xgrid = np.linspace(-1, 1, m)
    ygrid = np.linspace(1, -1, n)
    x0, x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    # # set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)
        print('# using CUDA device:', d, file=sys.stderr)

    if use_cuda:
        y_train = y_train.cuda()
        y_test = y_test.cuda()
        x_coord = x_coord.cuda()

    data_train = torch.utils.data.TensorDataset(y_train)
    data_test = torch.utils.data.TensorDataset(y_test)

    z_dim = args.z_dim
    print('# training with z-dim:', z_dim, file=sys.stderr)

    num_layers = args.num_layers
    hidden_dim = args.hidden_dim

    # default activation
    activation = nn.LeakyReLU
    if args.activation == 'tanh':
        activation = nn.Tanh
    elif args.activation == 'relu':
        activation = nn.LeakyReLU

    # Build models
    if args.vanilla:
        print('# using the vanilla MLP generator architecture', file=sys.stderr)
        n_out = n * m
        p_net = models.VanillaGenerator(n_out, z_dim, hidden_dim, num_layers=num_layers, activation=activation)
        inf_dim = z_dim
        rotate = False
        translate = False
    else:
        n_out = 1
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

    q_net = models.InferenceNetwork(n*m, inf_dim, hidden_dim, num_layers=num_layers, activation=activation)

    if use_cuda:
        p_net.cuda()
        q_net.cuda()

    dx_scale = args.dx_scale
    theta_prior = args.theta_prior

    print('# using priors: theta={}, dx={}'.format(theta_prior, dx_scale), file=sys.stderr)

    N = len(images_train)
    params = list(p_net.parameters()) + list(q_net.parameters())

    lr = args.learning_rate
    optim = torch.optim.Adam(params, lr=lr)
    minibatch_size = args.minibatch_size

    train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size, shuffle=True)
    val_iterator = torch.utils.data.DataLoader(data_test, batch_size=minibatch_size)

    path_prefix = args.save_prefix
    save_interval = args.save_interval

    MiscTools.sample_images(iterator=val_iterator, image_dims=image_dims, output_dir=output_dir)

    # Initialise results bins
    output = sys.stdout
    header_parts = '\t'.join(['Epoch', 'Date', 'Time', 'Split', 'ELBO', 'General loss', 'KL'])
    print(header_parts, file=output)

    train_results = []
    val_results = []

    train_results.append(header_parts)
    val_results.append(header_parts)

    for epoch in range(num_epochs):

        epoch_str = str(epoch + 1).zfill(digits)

        elbo_accum, gen_loss_accum, kl_loss_accum = train_epoch(train_iterator, x_coord, p_net, q_net,
                                                                optim, rotate=rotate, translate=translate,
                                                                dx_scale=dx_scale, theta_prior=theta_prior,
                                                                epoch=epoch, num_epochs=num_epochs, N=N,
                                                                use_cuda=use_cuda)

        line = '\t'.join([str(epoch+1), datetime.datetime.now().strftime('%y%m%d'),
                          datetime.datetime.now().strftime('%H%M%S'),
                          'train',
                          str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        train_results.append(line)
        print(line, file=output)
        output.flush()

        # evaluate on the test set
        to_save_image_samples = ((epoch+1) % save_interval == 0)
        elbo_accum, gen_loss_accum, kl_loss_accum = eval_model(val_iterator, x_coord, p_net,
                                                               q_net, rotate=rotate, translate=translate,
                                                               dx_scale=dx_scale, theta_prior=theta_prior,
                                                               use_cuda=use_cuda,
                                                               to_save_image_samples=to_save_image_samples,
                                                               image_dims=image_dims,
                                                               epoch=epoch_str, output_dir=output_dir
                                                               )
        line = '\t'.join([str(epoch + 1), datetime.datetime.now().strftime('%y%m%d'),
                          datetime.datetime.now().strftime('%H%M%S'),
                          'validation',
                          str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        val_results.append(line)
        print(line, file=output)
        output.flush()

        # save the models
        if path_prefix is not None and (epoch+1) % save_interval == 0:

            path = path_prefix + '_generator_epoch{}.sav'.format(epoch_str)
            p_net.eval().cpu()
            torch.save(p_net, path)

            path = path_prefix + '_inference_epoch{}.sav'.format(epoch_str)
            q_net.eval().cpu()
            torch.save(q_net, path)

            # Revert to cuda
            if use_cuda:
                p_net.cuda()
                q_net.cuda()

    MiscTools.save_results(output_dir=output_dir, train_results=train_results, val_results=val_results)

    end_time = datetime.datetime.now()
    print(f"End : {end_time.strftime('%y%m%d_%H%M%S')}")

    elapsed_time = (end_time - start_time)
    print(f"Elapsed time: {elapsed_time}")


if __name__ == '__main__':
    main()

