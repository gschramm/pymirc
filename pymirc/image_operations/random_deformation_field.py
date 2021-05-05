import numpy as np

from numba         import njit, prange
from scipy.ndimage import gaussian_filter

from .zoom3d       import zoom3d

def random_deformation_field(shape, shift = 2., n = 30, npad = 5, gaussian_std = 6):
  """Generate a smooth random 3d deformation vector field

  Parameters
  ----------
  shape : 3 element tuple
    shape of the deformation fields

  shift : float, 3 element tuple, optional
    standard devitiation of the displacements in the deformation fields (default 2)

  n : int, 3 element tuple, optional
    dimension of low resolution grid on which random deformations are sampled (default 30)

  npad: int, 3 element tuple, optional
    number of voxels used for 0 padding of the low resolution grid (default 5)

  gaussian_std: float, 3 element tuple, optional
    standard deviation of gaussian kernel used to smooth the low resolution deformations (default 6)

  Returns
  -------
  tuple of 3 3d numpy arrays
    containing the displacements in the 3 dimensions
  """
  if not isinstance(shift, tuple):
    shift = (shift,) * 3

  if not isinstance(n, tuple):
    n = (n,) * 3

  if not isinstance(npad, tuple):
    npad = (npad,) * 3

  if not isinstance(gaussian_std, tuple):
    gaussian_std = (gaussian_std,) * 3

  d0 = gaussian_filter(np.pad(np.random.randn(n[0],n[1],n[2]), npad[0], mode = 'constant'), gaussian_std[0])
  d1 = gaussian_filter(np.pad(np.random.randn(n[0],n[1],n[2]), npad[1], mode = 'constant'), gaussian_std[1])
  d2 = gaussian_filter(np.pad(np.random.randn(n[0],n[1],n[2]), npad[2], mode = 'constant'), gaussian_std[2])

  d0 = zoom3d(d0, np.array(shape) / np.array(d0.shape))
  d1 = zoom3d(d1, np.array(shape) / np.array(d1.shape))
  d2 = zoom3d(d2, np.array(shape) / np.array(d2.shape))

  d0 *= shift[0]/d0.std()
  d1 *= shift[1]/d1.std()
  d2 *= shift[2]/d2.std()

  d0 = d0[:min(d0.shape[0], shape[0]), :min(d0.shape[1], shape[1]), :min(d0.shape[2], shape[2])]
  d1 = d1[:min(d1.shape[0], shape[0]), :min(d1.shape[1], shape[1]), :min(d1.shape[2], shape[2])]
  d2 = d2[:min(d2.shape[0], shape[0]), :min(d2.shape[1], shape[1]), :min(d2.shape[2], shape[2])]

  return d0, d1, d2
