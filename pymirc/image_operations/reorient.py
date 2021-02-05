import nibabel as nib
from nibabel.orientations import inv_ornt_aff
import numpy as np

def reorient_image_and_affine(img, aff):
  """ Reorient an image and and affine such that the affine is approx. diagonal
      and such that the elements on the main diagonal are positiv
  
  Parameters
  ----------
  img : 3D numpy array
    containing the image

  aff : 2D numpy array
    (4,4) affine transformation matrix from image to anatomical coordinate system in
    homogeneous coordinates

  Returns
  -------
  a tuple with the reoriented image and the accordingly transformed affine

  Note
  ----
  The reorientation uses nibabel's io_orientation() and apply_orientation()
  """
  ornt  = nib.io_orientation(aff)
  img_t = nib.apply_orientation(img, ornt)
  aff_t = aff.dot(inv_ornt_aff(ornt, img.shape))

  return img_t, aff_t

def flip_image_and_affine(img, aff, flip_dirs):
  """ Flip an image along given axis and transform the affine accordingly

  Parameters
  ----------
  img : numpy array (2D, 3D)
    containing the image

  aff : 2D numpy array
    (4,4) affine transformation matrix from image to anatomical coordinate system in
    homogeneous coordinates

  flip_dirs : int or tuple of ints
    containing the axis where the flip should  be applied

  Returns
  -------
  a tuple with the flipped image and the accordingly transformed affine
  """
  if not isinstance(flip_dirs, tuple):
    flip_dirs = (flip_dirs,)

  # flip the image
  img = np.flip(img, flip_dirs)

  for flip_dir in flip_dirs:
    # tmp is the diagonal of the flip affine (all ones, but -1 in the flip direction)
    tmp            = np.ones(img.ndim + 1)
    tmp[flip_dir]  = -1

    # tmp2 is the (i,j,k,1) vector for the 0,0,0 voxel
    tmp2           = np.zeros(img.ndim + 1)
    tmp2[-1]       = 1

    # tmp2 is the (i,j,k,1) vector for the last voxel along the flip direction
    tmp3           = tmp2.copy()
    tmp3[flip_dir] = img.shape[flip_dir] - 1

    # this is the affine that does the flipping 
    # the flipping is actual a shift to the center, followed by an inversion
    # followed by in the inverse shift
    flip_aff = np.diag(tmp)
    flip_aff[flip_dir,-1] = ((aff @ tmp2) + (aff @ tmp3))[flip_dir]

    # transform the affine
    aff = flip_aff @ aff

  return img, aff
