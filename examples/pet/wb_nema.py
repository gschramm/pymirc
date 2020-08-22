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

fitres, sphere_results = fit_WB_NEMA_sphere_profiles(vol, voxsize, sameSignal = True, sm_fwhm = 4.5)
                                                     #Rfix = [18.5, 14.0, 11.0, 8.5, 6.5, 5.])

print(sphere_results)

fig1 = show_WB_NEMA_profiles(fitres)
fig2 = show_WB_NEMA_recoveries(sphere_results, sphere_results['signal'].values[0])
