import sys, os
import numpy as np
import pylab as py

from scipy.optimize import minimize

pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import pymirc.viewer           as pymv
import pymirc.fileio           as pymf
import pymirc.image_operations as pymi
import pymirc.registration     as pymr

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
ct_vol[ct_vol < -1024] = -1024

# the PET and CT images are on different voxel grids
# to view them in parallel, we interpolate the PET volume to the CT grid
ct_vol_pet_grid = pymi.aff_transform(ct_vol, np.linalg.inv(ct_dcm.affine) @ pet_dcm.affine, pet_vol.shape, 
                                     cval = -1024)

# create a random affine tansformation
params      = np.array([5.5,-7.5,4.5,0.2,-0.3,0.15])
origin      = np.array(pet_vol.shape)/2
aff         = pymi.kul_aff(params, origin = origin)
pet_shifted = pymi.aff_transform(pet_vol, aff, pet_vol.shape) 

# rigidly align the shifted PET to CT by minimizing neg mutual information
res = minimize(pymr.regis_cost_func, np.zeros(6), args = (ct_vol_pet_grid, pet_shifted, True), method = 'Powell', 
               options = {'ftol':1e-2, 'xtol':1e-2, 'disp':True, 'maxiter':10, 'maxfev':500})

regis_aff = pymi.kul_aff(res.x, origin = origin)

pet_coreg = pymi.aff_transform(pet_shifted, regis_aff, pet_vol.shape) 

imshow_kwargs = [{'cmap':py.cm.Greys_r,'vmin':-1024,'vmax':5e2},
                 {'cmap':py.cm.Greys,  'vmin':0,    'vmax':7e3},
                 {'cmap':py.cm.Greys,  'vmin':0,    'vmax':7e3}]

vi = pymv.ThreeAxisViewer([ct_vol_pet_grid,pet_shifted, pet_coreg], 
                          imshow_kwargs = imshow_kwargs, voxsize = pet_dcm.voxsize)