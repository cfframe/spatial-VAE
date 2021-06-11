# spatial-VAE

Source code and datasets for [Explicitly disentangling image content from translation and rotation with spatial-VAE](https://arxiv.org/abs/1909.11663) to appear at NeurIPS 2019.


Learned hinge motion of 5HDB (1-d latent variable) <br />
![5HDB_gif](gifs/5HDB_spatial.gif)

Learned arm motion of CODH/ACS (2-d latent variable) <br />
![codhacs_gif](gifs/codhacs_spatial.gif)

Learned antibody conformations (2-d latent variable) <br />
![antibody_gif](gifs/antibody_spatial.gif)

## Bibtex

For Bibtex relating to the original paper, see [https://github.com/tbepler/spatial-VAE](https://github.com/tbepler/spatial-VAE) 

## Setup

Dependencies:
- python 3
- pytorch >= 0.4
- torchvision
- numpy
- pillow
- [topaz](https://github.com/tbepler/topaz) (for loading MRC files)

## Datasets

Datasets as tarballs are available from the links below.

- [Rotated MNIST](http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated.tar.gz)
- [Rotated & Translated MNIST](http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated_translated.tar.gz)
- [5HDB simulated EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/5HDB.tar.gz)
- [CODH/ACS EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/codhacs.tar.gz)
- [Antibody EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/antibody.tar.gz)
- [Galaxy zoo](http://bergerlab-downloads.csail.mit.edu/spatial-vae/galaxy_zoo.tar.gz)

U### Assumptions
Data has been downloaded and extracted to spatial-VAE/data/XXX, where XXX is the top level folder of the 
main data set e.g. spatial-VAE/data/galaxy_zoo, which contains galaxy_zoo_test.py and galaxy_zoo_train.py. 

## Usage

The scripts, "train\_mnist.py", "train\_particles.py", and "train\_galaxy.py", train spatial-VAE models on the MNIST, single particle EM, and galaxy zoo data.

For example, to train a spatial-VAE model on the CODH/ACS dataset

```
python train_particles.py data/codhacs/processed_train.npy data/codhacs/processed_test.npy --num-epochs=1000 --augment-rotation
```

Some script options include:  
--z-dim: dimension of the unstructured latent variable (default: 2)  
--p-hidden-dim and --p-num-layers: the number of layers and number of units per layer in the spatial generator network  
--q-hidden-dim and --q-num-layers: the number of layers and number of units per layer in the approximate inference network  
--dx-prior, --theta-prior: standard deviation (in fraction of image size) of the translation prior and standard deviation of the rotation prior  
--no-rotate, --no-translate: flags to disable rotation and translation inference  
--normalize: normalize the images before training (subtract mean, divide by standard deviation)  
--ctf-train, --ctf-test: path to tables containing CTF parameters for the train and test images, used to perform CTF correction if provided  
--fit-noise: also output the standard deviation of each pixel from the spatial generator network, sometimes called a colored noise model  
--save-prefix: save model parameters every few epochs to this path prefix  

See --help for complete arguments list.

### Specific to galaxy.py
Validation uses a portion of the training data. Control of how much is via these:  
--num-train-images: number of training images (default: 0 = all)  
--val-split: % split of training images for validation instead of training (default: 50)  

Example for use on a personal computer:  
```
cd spatial-VAE
python train_galaxy.py data/galaxy_zoo/galaxy_zoo_train.npy data/galaxy_zoo/galaxy_zoo_test.npy --num-epochs=4  --minibatch-size=64 --num-train-images=4000 --val-split=50 --save-prefix=galaxy --save-interval=2 -z=20
```
Adjust these for initial testing on an underpowered machine:  
 * mini-batch-size
 * num-train-images (the default of 0 uses all of them)
 * num-epochs
 
## Known issues
File data/galaxy_zoo/galaxy_zoo_test.npy is currently redundant, code to be updated to either remove references to it 
completely or to run a test after all training complete.
 
Trained models and output images are saved to the *outputs/trained* and *outputs/images* directories respectively.

## License

This source code is provided under the [MIT License](https://github.com/tbepler/spatial-VAE/blob/master/LICENSE).

