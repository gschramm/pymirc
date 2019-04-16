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

  shift : float, optional
    standard devitiation of the displacements in the deformation fields (default 2)

  n : int, optional
    dimension of low resolution grid on which random deformations are sampled (default 30)

  npad: int, optional 
    number of voxels used for 0 padding of the low resolution grid (default 5)

  gaussian_std: float, optional
    standard deviation of gaussian kernel used to smooth the low resolution deformations (default 6)

  Returns
  -------
  tuple of 3 3d numpy arrays
    containing the displacements in the 3 dimensions
  """
  d0 = gaussian_filter(np.pad(np.random.randn(n,n,n), npad, mode = 'constant'), gaussian_std)
  d1 = gaussian_filter(np.pad(np.random.randn(n,n,n), npad, mode = 'constant'), gaussian_std)
  d2 = gaussian_filter(np.pad(np.random.randn(n,n,n), npad, mode = 'constant'), gaussian_std)

  d0 = zoom3d(d0, np.array(shape) / np.array(d0.shape))
  d1 = zoom3d(d1, np.array(shape) / np.array(d1.shape))
  d2 = zoom3d(d2, np.array(shape) / np.array(d2.shape))

  d0 *= shift/d0.std()
  d1 *= shift/d1.std()
  d2 *= shift/d2.std()

  return d0, d1, d2

