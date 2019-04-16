import sys, os
pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import numpy   as np
import nibabel as nib
import pydicom

import pymirc.fileio as pymf

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('nifti_file',   help = 'nifti file to be converted into dicom')
parser.add_argument('ref_dcm_file', help = 'the reference dicom file')
parser.add_argument('output_dir',   help = 'output directory')

parser.add_argument('--dcm_tag_file',       help = 'txt file with dcm tags to copy', 
                    default = 'dcm_tags_to_copy.txt')
parser.add_argument('--series_desc_prefix', help = 'prefix for dcm series description', 
                    default = '(KUL Segmentation)')
parser.add_argument('--output_modality', help = 'dicom modality of output', 
                    default = 'CT')

args       = parser.parse_args()
output_dir = args.output_dir

#-----------------------------------------------------------------------------------------

if os.path.isdir(output_dir):
  raise FileExistsError('output directory ' + output_dir + ' already exists')

# load the nifti and the affine and convert to LPS orientation
nii        = nib.load(args.nifti_file)
nii        = nib.as_closest_canonical(nii)
vol_ras    = nii.get_data()

affine_ras = nii.affine
vol    = np.flip(np.flip(vol_ras, 0), 1)
affine = affine_ras.copy()
affine[0,-1] = (-1 * nii.affine @ np.array([vol.shape[0]-1,0,0,1]))[0]
affine[1,-1] = (-1 * nii.affine @ np.array([0,vol.shape[1]-1,0,1]))[1]

# load the list of dicom tags to copy from the reference header from an input text file
with open(args.dcm_tag_file,'r') as f:
  tags_to_copy = [x.strip() for x in f.read().splitlines()]

# read the reference dicom volume
ref_dcm = pydicom.read_file(args.ref_dcm_file)

# create the dictionary of tags and values that are copied from the reference dicom header
dcm_header_kwargs = {}
for tag in tags_to_copy:
  if tag in ref_dcm:
    dcm_header_kwargs[tag] = ref_dcm.data_element(tag).value

# write the dicoms
pymf.write_3d_static_dicom(vol, output_dir, 
                           affine              = affine,
                           SeriesDescription   = args.series_desc_prefix + ' ' + ref_dcm.SeriesDescription,
                           modality            = args.output_modality,
                           **dcm_header_kwargs)
