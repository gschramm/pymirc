import sys, os
import numpy as np
import pylab as py

from scipy.optimize import minimize

pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import pymirc.viewer           as pymv
import pymirc.fileio           as pymf
import pymirc.image_operations as pymi
import pymirc.metrics.cost_functions as pymr

#---------------------------------------------------------------------------------------------

data_dir = os.path.join('..','..','data','nema_petct')

if not os.path.exists(data_dir):
  url = 'https://kuleuven.box.com/s/wub9pk0yvt8kjqyj7p0bz11boca4334x'
  print('please first download example PET/CT data from:')
  print(url)
  print('and unzip into: ', data_dir)
  sys.exit()

# read PET/CT nema phantom dicom data sets
# the image data in those two data sets are aligned but
# on different grids
pet_dcm = pymf.DicomVolume(os.path.join(data_dir,'PT','*.dcm'))
ct_dcm  = pymf.DicomVolume(os.path.join(data_dir,'CT','*.dcm'))

pet_vol = pet_dcm.get_data()
ct_vol  = ct_dcm.get_data()
ct_vol[ct_vol < -1024] = -1024

# we artificially rotate and shift the CT for the regisration of the PET
rp         = np.array([10,-5,-20,0.1,0.2,-0.1])
R          = pymi.kul_aff(rp, origin = np.array(ct_vol.shape)/2)
ct_vol_rot = pymi.aff_transform(ct_vol, R, ct_vol.shape, cval = ct_vol.min()) 

# define the affine transformation that maps the PET on the CT grid
# we need this to avoid the additional interpolation from the PET to the CT grid
pre_affine = np.linalg.inv(pet_dcm.affine) @ ct_dcm.affine

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# rigidly align the shifted PET to CT by minimizing neg mutual information
# downsample the arrays by a factor for a fast pre-registration
reg_params = np.zeros(6)

# (1) initial registration with downsampled arrays
# define the down sampling factor
dsf    = 3
ds_aff = np.diag([dsf,dsf,dsf,1.])

ct_vol_rot_ds = pymi.aff_transform(ct_vol_rot, ds_aff, np.ceil(np.array(ct_vol.shape)/dsf).astype(int))

res = minimize(pymr.regis_cost_func, reg_params, 
               args = (ct_vol_rot_ds, pet_vol, True, True, pymr.neg_mutual_information, pre_affine @ ds_aff), 
               method = 'Powell', 
               options = {'ftol':1e-2, 'xtol':1e-2, 'disp':True, 'maxiter':20, 'maxfev':5000})

reg_params = res.x.copy()
# we have to scale the translations by the down sample factor since they are in voxels
reg_params[:3] *= dsf

# (2) registration with full arrays
res = minimize(pymr.regis_cost_func, reg_params, 
               args = (ct_vol_rot, pet_vol, True, True, pymr.neg_mutual_information, pre_affine), 
               method = 'Powell', 
               options = {'ftol':1e-2, 'xtol':1e-2, 'disp':True, 'maxiter':20, 'maxfev':5000})
reg_params = res.x.copy()

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# define the final affine transformation that maps from the PET grid to the rotated CT grid
af  = pre_affine @ pymi.kul_aff(reg_params, origin = np.array(ct_vol_rot.shape)/2)
pet_coreg = pymi.aff_transform(pet_vol, af, ct_vol_rot.shape, cval = pet_vol.min())

imshow_kwargs = [{'cmap':py.cm.Greys_r, 'vmin': -200, 'vmax' : 200},
                 {'cmap':py.cm.Greys_r, 'vmin': -200, 'vmax' : 200},
                 {'cmap':py.cm.Greys, 'vmin':0, 'vmax':np.percentile(pet_coreg,99.9)}]

vi = pymv.ThreeAxisViewer([ct_vol_rot, ct_vol_rot, pet_coreg], ovols = [None, pet_coreg, None],
                          imshow_kwargs = imshow_kwargs, 
                           voxsize = ct_dcm.voxsize, width = 6)
