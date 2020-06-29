import pymirc.fileio as pf
import pymirc.image_operations as pi
import numpy as np
import os

from argparse import ArgumentParser

#--------------------------------------------------------------------------------
# parse the command line

parser = ArgumentParser('Interpolate an RTDose dicom')
parser.add_argument('input_file', help = 'input 3D dicom file')
parser.add_argument('--target_voxsize', help = 'target voxel size', 
                    nargs = '+', default = [1.,1.,1.], type = float)
parser.add_argument('--output_dir', default = None, help = 'output directory')
parser.add_argument('--dcm_tag_file', default = 'rtdose_tags_to_copy.txt', help = 'file with tags to copy')

args = parser.parse_args()

input_file     = args.input_file
target_voxsize = np.array(args.target_voxsize)

if target_voxsize.shape[0] == 1:
  target_voxsize = np.full(3, target_voxsize[0])

output_dir = args.output_dir

if output_dir is None:
  output_dir = os.path.splitext(input_file)[0] + '_interpolated'

# check if output dir already exists
while os.path.exists(output_dir):
  output_dir += '_1'

# load the list of dicom tags to copy from the reference header from an input text file
with open(args.dcm_tag_file,'r') as f:
  tags_to_copy = [x.strip() for x in f.read().splitlines()]

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

dcm = pf.DicomVolume(input_file)
vol = dcm.get_data()
aff = dcm.affine

# interpolate the volume to the target voxelsize using trilinear interpolation
vol_interp = pi.zoom3d(vol, dcm.voxsize / target_voxsize)

# generate the new affine of the interpolated array
aff_interp = aff.copy()
aff_interp[:,0] *= (target_voxsize[0] / dcm.voxsize[0])
aff_interp[:,1] *= (target_voxsize[1] / dcm.voxsize[1])
aff_interp[:,2] *= (target_voxsize[2] / dcm.voxsize[2])

aff_interp[:-1,3] = aff[:-1,-1] - 0.5*dcm.voxsize + 0.5*target_voxsize

# create the dictionary of tags and values that are copied from the reference dicom header
dcm_header_kwargs = {}
for tag in tags_to_copy:
  if tag in dcm.firstdcmheader:
    dcm_header_kwargs[tag] = dcm.firstdcmheader.data_element(tag).value

# adjust the GridFrameOffsetVector
dcm_header_kwargs['GridFrameOffsetVector'] = (np.arange(vol_interp.shape[2])*target_voxsize[2]).tolist()

pf.write_3d_static_dicom(vol_interp, output_dir, affine = aff_interp, modality = 'RTDOSE', **dcm_header_kwargs)
print(f'wrote {output_dir}')
