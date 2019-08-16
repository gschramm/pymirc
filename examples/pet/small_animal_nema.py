# script to perform NEMA 2008 analysis on small animal IQ phantom

import pylab as py

import pymirc.fileio           as pmf
import pymirc.viewer           as pmv
import pymirc.image_operations as pmi

from nema_pet import align_nema_2008_small_animal_iq_phantom, fit_nema_2008_cylinder_profiles
from nema_pet import nema_2008_small_animal_pet_rois, nema_2008_small_animal_iq_phantom_report

#input_file   = 'mydcm.dcm'
input_file   = '/uz/data/Admin/ngeworkingresearch/georg/molecubes/data/20190807140219_PET_OSEM_0_frame1_iter30.dcm'
align_volume = True

#--------------------------------------------------------------------
dcm     = pmf.DicomVolume([input_file])
vol     = dcm.get_data() 
voxsize = dcm.voxsize 

if align_volume:
  vol_orig = vol.copy()
  vol      = align_nema_2008_small_animal_iq_phantom(vol, voxsize)

fit_nema_2008_cylinder_profiles(vol, voxsize, Rrod_init  = [2.5,2,1.5,1,0.5],
                                fwhm_init = 1.5, S_init = 151, fix_S = True,
                                fix_R = False, fix_fwhm  = False)

roi_vol = nema_2008_small_animal_pet_rois(vol, voxsize)

nema_2008_small_animal_iq_phantom_report(vol, roi_vol)

# show the volume and the ROI volume
vi = pmv.ThreeAxisViewer([vol,roi_vol], width=5, imshow_kwargs = [{'cmap':py.cm.Greys},{'cmap':py.cm.nipy_spectral}])
