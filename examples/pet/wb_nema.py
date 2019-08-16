import numpy as np
import pylab as py
import glob
import os

from argparse import ArgumentParser
from scipy.io import readsav

import pymirc.fileio as pmf

from nema_pet import fit_WB_NEMA_sphere_profiles

dcm     = pmf.DicomVolume('mydcm_dir*.dcm')
vol     = dcm.get_data() 
voxsize = dcm.voxsize 

res = fit_WB_NEMA_sphere_profiles(vol, voxsize, Sfix = 16800, sm_fwhm = 4.5)
