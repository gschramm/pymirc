# small example on how to read nrrd files and segmentation as e.g.
# produced by 3D slicer

# to run it you need to install pynrrd

import argparse
import nrrd
import numpy as np
import matplotlib.pyplot as py
import pymirc.image_operations as pi
import pymirc.viewer as pv

#----------------------------------------------------------------------------

def get_lps_affine_from_hdr(hdr):
  aff = np.eye(4)
  aff[:3,:3] = hdr['space directions']
  aff[:3,-1] = hdr['space origin']

  # convert the affine to LPS
  if hdr['space'] == 'right-anterior-superior':
    aff = np.diag([-1,-1,1,1]) @ aff
  elif hdr['space'] == 'left-anterior-superior':
    aff = np.diag([1,-1,1,1]) @ aff
  elif hdr['space'] == 'right-posterior-superior':
    aff = np.diag([-1,1,1,1]) @ aff

  return aff

#----------------------------------------------------------------------------

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument('ct_file',  help = 'CT nrrd file')
parser.add_argument('seg_file', help = 'segmentation nrrd file')
args = parser.parse_args()

ct, ct_hdr   = nrrd.read(args.ct_file)
seg, seg_hdr = nrrd.read(args.seg_file)

ct_aff  = get_lps_affine_from_hdr(ct_hdr)
seg_aff = get_lps_affine_from_hdr(seg_hdr) 

ct_voxsize = np.sqrt((ct_aff**2).sum(0))[:-1]

# the segmentation array usually a crop of the original array
seg_offset = np.round((seg_aff[:-1,-1] - ct_aff[:-1,-1]) / np.diag(ct_aff)[:3]).astype(int)
seg2 = np.zeros(ct.shape, dtype = np.int8)
seg2[seg_offset[0]:(seg_offset[0]+seg.shape[0]), seg_offset[1]:(seg_offset[1]+seg.shape[1]),
     seg_offset[2]:(seg_offset[2]+seg.shape[2])] = seg
seg2_aff = ct_aff.copy()

# reorient the images to standard LPS orientation
ct, ct_aff = pi.reorient_image_and_affine(ct, ct_aff)
seg2, seg2_aff = pi.reorient_image_and_affine(seg2, seg2_aff)

imshow_kwargs = [{'cmap':py.cm.Greys_r, 'vmin':-300, 'vmax':300}, {}]

pv.ThreeAxisViewer([ct,seg2], voxsize = ct_voxsize, imshow_kwargs = imshow_kwargs)
