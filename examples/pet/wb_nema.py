import numpy as np
import pylab as py
import glob
import os

from argparse import ArgumentParser
from scipy.io import readsav

import pymirc.fileio as pmf
import pymirc.viewer as pv

from nema_pet import fit_WB_NEMA_sphere_profiles

dcm     = pmf.DicomVolume('my_dcm_dir/*.dcm')
vol     = dcm.get_data() 
voxsize = dcm.voxsize 

#res = fit_WB_NEMA_sphere_profiles(vol, voxsize, sm_fwhm = 4.5)
#res = fit_WB_NEMA_sphere_profiles(vol, voxsize, sm_fwhm = 4.5, Rfix = [18.5, 14., 11., 8.5, 6.5, 5.])
res = fit_WB_NEMA_sphere_profiles(vol, voxsize, sm_fwhm = 4.5, Rfix = [18.5, 14., 11., 8.5, 6.5, 5.], Sfix = 16914)

# show the n-th sphere sub volume
#pv.ThreeAxisViewer(res['subvols'][5], voxsize = voxsize, imshow_kwargs = {'vmin': 0, 'vmax': res['Ss'][0]})

# show the whole volume
#pv.ThreeAxisViewer(res['vol'], voxsize = voxsize, imshow_kwargs = {'vmin': 0, 'vmax': res['Ss'][0]})
