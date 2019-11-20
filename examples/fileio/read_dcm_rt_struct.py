import sys, os

#-------------------------------------------------------------------------------------
pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)
import pymirc.fileio as pymf
import pymirc.viewer as pymv

import numpy as np
import pylab as py

# check of data is there
data_dir = os.path.join('..','..','data','nema_petct')

if not os.path.exists(data_dir):
  url = 'https://kuleuven.box.com/s/wub9pk0yvt8kjqyj7p0bz11boca4334x'
  print('please first download example PET/CT data from:')
  print(url)
  print('and unzip into: ', data_dir)
  sys.exit()


# read CT vol that was used to generate ROIs in rtstruct file
ct_dcm = pymf.DicomVolume('../../data/nema_petct/CT/*.dcm')
ct_vol = ct_dcm.get_data()

aff   = ct_dcm.affine
shape = ct_vol.shape 

#-------------------------------------------------------------------------------------
# read the rt struct data
rtstruct_file = '../../data/nema_petct/rois/nonconvex_rtstruct.dcm'

# read the ROI contours (in world coordinates)
contour_data = pymf.read_rtstruct_contour_data(rtstruct_file)

# convert contour data to index arrays (voxel space)
roi_inds = pymf.convert_contour_data_to_roi_indices(contour_data, aff, shape)

#---------------------------------------------------------------------------
# create a label array
roi_vol = np.zeros(shape)

for i in range(len(roi_inds)):
  roi_vol[roi_inds[i]] = int(contour_data[i]['ROINumber'])

#---------------------------------------------------------------------------
# print some ROI statistics

print('ROI name.....:', [x['ROIName']     for x in contour_data])
print('ROI number...:', [x['ROINumber']   for x in contour_data])
print('ROI mean.....:', [ct_vol[x].mean() for x in roi_inds])
print('ROI max......:', [ct_vol[x].max()  for x in roi_inds])
print('ROI min......:', [ct_vol[x].min()  for x in roi_inds])
print('ROI # voxel..:', [len(ct_vol[x])   for x in roi_inds])

#---------------------------------------------------------------------------
# view the results
imshow_kwargs   = {'cmap':py.cm.Greys_r,'vmin':-500,'vmax':500}
oimshow_kwargs =  {'cmap':py.cm.nipy_spectral, 'alpha':0.3, 'vmax': 1.2*roi_vol.max()}
vi = pymv.ThreeAxisViewer([ct_vol,ct_vol], ovols = [None,roi_vol], voxsize = ct_dcm.voxsize, 
                          imshow_kwargs = imshow_kwargs, oimshow_kwargs = oimshow_kwargs)

print('\nPress "a" to hide/show overlay')
