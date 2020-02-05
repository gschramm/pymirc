# INSTALL conda env with dependencies needed

This should install a conda env "tf" with tensorflow >= 2.0
which comes with keras integrated.

(CPU) conda create -n tf tensorflow scipy matplotlib ipython
(GPU) conda create -n tf tensorflow-gpu scipy matplotlib ipython  

# run the mnist segmentation toy problem 

conda activate tf
python mnist_seg_with_generator.py --epochs 10 --batch_size 50 --nfeat 8 --loss_fct dice 
