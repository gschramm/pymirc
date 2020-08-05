import os
import numpy as np
import matplotlib.pyplot as py

import pymirc.viewer           as pymv
import pymirc.fileio           as pymf
import pymirc.image_operations as pymi

data_dir = os.path.join('..','..','data','nema_petct')

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

pet_coreg, coreg_aff, coreg_params = pymi.rigid_registration(pet_vol, ct_vol_rot, 
                                                             pet_dcm.affine, ct_dcm.affine)

imshow_kwargs = [{'cmap':py.cm.Greys_r, 'vmin': -200, 'vmax' : 200},
                 {'cmap':py.cm.Greys_r, 'vmin': -200, 'vmax' : 200},
                 {'cmap':py.cm.Greys, 'vmin':0, 'vmax':np.percentile(pet_coreg,99.9)}]

vi = pymv.ThreeAxisViewer([ct_vol_rot, ct_vol_rot, pet_coreg], ovols = [None, pet_coreg, None],
                          imshow_kwargs = imshow_kwargs, 
                          voxsize = ct_dcm.voxsize, width = 6)
