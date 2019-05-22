import sys, os

#-------------------------------------------------------------------------------------
pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)
import pymirc.fileio as pymf
import pymirc.viewer as pymv

import numpy as np
import pylab as py

import nibabel as nib

from argparse import ArgumentParser

#-------------------------------------------------------------------------------------
# parse the command line
parser = ArgumentParser()
parser.add_argument('dcm_vol_dir',    help = 'dicom folder')
parser.add_argument('rtstruct_file',  help = 'dicom rt struct file')
parser.add_argument('--output_fname', help = 'output file name', default = 'labelarray.nii')
parser.add_argument('--dcm_pattern',  help = 'dicom pattern for files in dicom directory', default = '*')

args          = parser.parse_args()
dcm_vol_dir   = args.dcm_vol_dir
rtstruct_file = args.rtstruct_file
output_fname  = args.output_fname
dcm_pattern   = args.dcm_pattern

#-------------------------------------------------------------------------------------
# read CT vol that was used to generate ROIs in rtstruct file
ct_dcm = pymf.DicomVolume(os.path.join(dcm_vol_dir,dcm_pattern))
ct_vol = ct_dcm.get_data()

aff   = ct_dcm.affine
shape = ct_vol.shape 

#-------------------------------------------------------------------------------------
# read the rt struct data

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
# save the label array as nifti

# nifti is uses RAS so we have to convert the affine and the volume
output_aff_ras       = aff.copy()
output_aff_ras[0,-1] = (-1 * aff @ np.array([shape[0]-1,0,0,1]))[0]
output_aff_ras[1,-1] = (-1 * aff @ np.array([0,shape[1]-1,0,1]))[1]

nib.save(nib.Nifti1Image(np.flip(np.flip(roi_vol,0),1), output_aff_ras), output_fname)
