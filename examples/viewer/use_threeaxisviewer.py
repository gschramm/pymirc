import sys, os
import numpy as np
import pylab as py

pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import pymirc.viewer as pymv

#---------------------------------------------------------------------------------------------

np.random.seed(42)
# generate 2 random 4D data sets
# the viewer expects the axis to be (time, patient left right, patient anterior posterior, patient feet head)
# in LPS orientation (dicom standard orientation)
# waring: the default nifti orientation is RAS
vols = [np.random.rand(3,100,100,50),np.random.rand(3,100,100,50)]
voxsize = [1.,1.,2.]
imshow_kwargs = [{'cmap':py.cm.Greys_r,'vmin':0,'vmax':1},{'cmap':py.cm.jet,'vmin':0,'vmax':1.2}]

vi = pymv.ThreeAxisViewer(vols, voxsize = voxsize, imshow_kwargs = imshow_kwargs)

if not pymirc_path in sys.path: sys.path.append(pymirc_path)
