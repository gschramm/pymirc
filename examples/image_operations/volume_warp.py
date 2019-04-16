import sys, os
import numpy as np
import pylab as py

pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import pymirc.viewer           as pymv
import pymirc.image_operations as pymi

# seed random genrator for random deformation field
np.random.seed(2)

# setup demo volume
n0 = 120
n1 = 110
n2 = 100

x0, x1, x2 = np.meshgrid(np.arange(n0),np.arange(n1),np.arange(n2))

vol = np.pad(0.5*((((-1)**(x0//6)) * ((-1)**(x1//6)) * ((-1)**(x2//6))) + 1), 5, mode = 'constant')

# generate random deformation field
d0, d1, d2 = pymi.random_deformation_field(vol.shape, shift = 2.5)

# apply warping
warped_vol  = pymi.backward_3d_warp(vol, d0, d1, d2)

vi = pymv.ThreeAxisViewer([vol, warped_vol, np.sqrt(d0**2 + d1**2 + d2**2)])
