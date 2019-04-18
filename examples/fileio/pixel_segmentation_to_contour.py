import sys, os
pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import pymirc.image_operations as pymi

import numpy as np
import pylab as py

from scipy.ndimage import gaussian_filter

#--------------------------------------------------------------------
# minimal example on how to convert a pixelized segmentation into
# a set of contours (useful for writing RTstructs)

np.random.seed(42)
bin_img = (gaussian_filter(np.random.rand(50,50),0.7) > 0.5).astype(int)

contours =  pymi.binary_2d_image_to_contours(bin_img, include_holes= True)

# Display the image and plot all contours found
fig, ax = py.subplots(1,1, figsize = (8,8))

cols = ['r','b','g','y','c','m']

ax.imshow(bin_img, interpolation = 'nearest', cmap= py.cm.gray)
for n, contour in enumerate(contours):
  ax.plot(contour[:, 1], contour[:, 0], color = cols[n % len(cols)])

fig.tight_layout()
fig.show()
