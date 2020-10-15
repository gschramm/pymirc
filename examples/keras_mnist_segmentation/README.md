# Minimal keras example for 2D image segmentation using a unet

## Installation

### Installation of anaconda (miniconda)

Download and install Miniconda from <https://docs.conda.io/en/latest/miniconda.html>.

Please use the ***Python 3.x*** installer and confirm that the installer
should run ```conda init``` at the end of the installtion process.

### Create a dedicated conda virtual environment

We recommend to create a separate virtual conda environment to avoid problems
with dependencies. Here we assume that the environment is called "tf".

```conda create -n tf tensorflow numpy scipy matplotlib ipython numba scikit_image```


### Installation of pymirc package

Activate the conda environment:
```conda activate tf```

Install pymirc via pip from the github repository in the environment:
```pip install git+https://github.com/gschramm/pymirc```


## Run the mnist segmentation toy problem

```conda activate tf```
```python mnist_seg_with_generator.py --epochs 5 --batch_size 50 --nfeat 8 --loss_fct dice```

The first time you run the script, the mnist data will be downloaded
and preprocessed.
