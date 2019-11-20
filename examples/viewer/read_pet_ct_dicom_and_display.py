import sys, os
import numpy as np
import pylab as py

pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import pymirc.viewer           as pymv
import pymirc.fileio           as pymf
import pymirc.image_operations as pymi

#---------------------------------------------------------------------------------------------

data_dir = os.path.join('..','..','data','nema_petct')

if not os.path.exists(data_dir):
  url = 'https://kuleuven.box.com/s/wub9pk0yvt8kjqyj7p0bz11boca4334x'
  print('please first download example PET/CT data from:')
  print(url)
  print('and unzip into: ', data_dir)
  sys.exit()

# read PET/CT nema phantom dicom data sets from
pet_dcm = pymf.DicomVolume(os.path.join(data_dir,'PT','*.dcm'))
pet_vol = pet_dcm.get_data()

ct_dcm  = pymf.DicomVolume(os.path.join(data_dir,'CT','*.dcm'))
ct_vol  = ct_dcm.get_data()

# the PET and CT images are on different voxel grids
# to view them in parallel, we interpolate the PET volume to the CT grid
pet_vol_ct_grid = pymi.aff_transform(pet_vol, np.linalg.inv(pet_dcm.affine) @ ct_dcm.affine, 
                                     output_shape = ct_vol.shape)

imshow_kwargs = [{'cmap':py.cm.Greys},
                 {'cmap':py.cm.Greys_r,'vmin':-500,'vmax':500},
                 {'cmap':py.cm.Greys_r,'vmin':-500,'vmax':500}]

oimshow_kwargs = {'cmap':py.cm.hot, 'alpha':0.3}

print('\nPress "a" to hide/show overlay')

vi = pymv.ThreeAxisViewer([pet_vol_ct_grid,ct_vol,ct_vol], 
                          ovols = [None, None, pet_vol_ct_grid],
                          voxsize = ct_dcm.voxsize, imshow_kwargs = imshow_kwargs,
                          oimshow_kwargs = oimshow_kwargs)
