# INSTALL conda env with dependencies needed

This should install a conda env "tf" with tensorflow >= 2.0
which comes with keras integrated.

(CPU) conda create -n tf tensorflow scipy matplotlib ipython numba scikit-image
(GPU) conda create -n tf tensorflow-gpu scipy matplotlib ipython numba scikit-image

# run the mnist segmentation toy problem

conda activate tf
python mnist_seg_with_generator.py --epochs 10 --batch_size 50 --nfeat 8 --loss_fct dice

# requirements
- python>=3.6
- numpy>=1.15.0
- scipy>=1.1.0
- numba>=0.39.0
- matplotlib>=2.2.2
- tensorflow>=2.0
