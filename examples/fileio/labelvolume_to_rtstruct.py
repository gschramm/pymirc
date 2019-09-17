import sys, os
pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import numpy           as np
import nibabel         as nib
import pymirc.fileio   as pymf

from glob import glob

nii = nib.load('../../data/TestRTstruct/Multimask.nii.gz')
nii = nib.as_closest_canonical(nii)
roi_vol_ras = nii.get_data()[:,:,:,0,0,0]
roi_vol     = np.flip(roi_vol_ras,(0,1))

aff_ras = nii.affine.copy()
aff     = aff_ras.copy()

aff[0,-1] = (-1 * aff_ras @ np.array([roi_vol.shape[0]-1,0,0,1]))[0]
aff[1,-1] = (-1 * aff_ras @ np.array([0,roi_vol.shape[1]-1,0,1]))[1]

ct_files    = glob('../../data/TestRTstruct/CT/*.dcm')
ct_dcm      = pymf.DicomVolume(ct_files)
ct          = ct_dcm.get_data()
refdcm_file = ct_files

#------------------------------------------------------------
pymf.labelvol_to_rtstruct(roi_vol, aff, refdcm_file, '../../data/TestRTstruct/t_rtstruct.dcm',
                          tags_to_add = {'SpecificCharacterSet':'ISO_IR 192'})
