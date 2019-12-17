import sys, os

#-------------------------------------------------------------------------------------
pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)
import pymirc.fileio as pymf
import pymirc.viewer as pymv

import numpy as np
import pylab as py

import nibabel as nib

data_dir = '/users/nexuz/gschra2/tmp/data_sets/ibsi_1_ct_radiomics_phantom'

# read CT vol that was used to generate ROIs in rtstruct file
ct_dcm = pymf.DicomVolume(os.path.join(data_dir,'dicom','image','*.dcm'))
ct_vol = ct_dcm.get_data()

aff   = ct_dcm.affine
shape = ct_vol.shape 

#-------------------------------------------------------------------------------------
# read the rt struct data
rtstruct_file = os.path.join(data_dir,'dicom','mask','DCM_RS_00060.dcm')

# read the ROI contours (in world coordinates)
contour_data = pymf.read_rtstruct_contour_data(rtstruct_file)

# convert contour data to index arrays (voxel space)
# in this example we have to ignore the orientation of the saved 2D contours (polygons)
roi_inds = pymf.convert_contour_data_to_roi_indices(contour_data, aff, shape, use_contour_orientation = False)

#---------------------------------------------------------------------------
# create a label array
roi_vol = np.zeros(shape)

for i in range(len(roi_inds)):
  roi_vol[roi_inds[i]] = int(contour_data[i]['ROINumber'])

#---------------------------------------------------------------------------
# load the reference mask from nifti
nii = nib.load(os.path.join(data_dir,'nifti','mask','mask.nii.gz'))
# bring nifti vol to RAS
nii = nib.as_closest_canonical(nii)
nii_mask = nii.get_data()
# flip nifti vol to LPS
nii_mask = np.flip(nii_mask, (0,1))

print('sum of abs. diff between ref. mask and read mask: ', np.abs(nii_mask - roi_vol).sum())
print('')

#---------------------------------------------------------------------------
# print some ROI statistics

print('ROI name.....:', [x['ROIName']     for x in contour_data])
print('ROI number...:', [x['ROINumber']   for x in contour_data])
print('ROI mean.....:', [ct_vol[x].mean() for x in roi_inds])
print('ROI max......:', [ct_vol[x].max()  for x in roi_inds])
print('ROI min......:', [ct_vol[x].min()  for x in roi_inds])
print('ROI # voxel..:', [len(ct_vol[x])   for x in roi_inds])

#---------------------------------------------------------------------------
# view the results
imshow_kwargs   = {'cmap':py.cm.Greys_r,'vmin':-500,'vmax':500}
oimshow_kwargs =  {'cmap':py.cm.nipy_spectral, 'alpha':0.3, 'vmax': 1.2*roi_vol.max()}
vi = pymv.ThreeAxisViewer([ct_vol,ct_vol], ovols = [None,roi_vol], voxsize = ct_dcm.voxsize, 
                          imshow_kwargs = imshow_kwargs, oimshow_kwargs = oimshow_kwargs)

print('\nPress "a" to hide/show overlay')
