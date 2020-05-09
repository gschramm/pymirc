# small example on how to read nrrd files and segmentation as e.g.
# produced by 3D slicer

# to run it you need to install pynrrd

import nrrd
import numpy as np
import pymirc.image_operations as pi
import pymirc.viewer as pv

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

ct, ct_hdr   = nrrd.read('CTChest.nrrd')
seg, seg_hdr = nrrd.read('Segmentation preview.seg.nrrd')  

ct_aff  = get_lps_affine_from_hdr(ct_hdr)
seg_aff = get_lps_affine_from_hdr(seg_hdr) 

ct_voxsize = np.sqrt((ct_aff**2).sum(0))[:-1]

seg2 = pi.aff_transform(seg, np.linalg.inv(seg_aff) @ ct_aff, ct.shape, trilin = False)

pv.ThreeAxisViewer([ct,seg2], voxsize=ct_voxsize)
