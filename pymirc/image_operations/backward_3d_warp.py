import numpy as np
import math

from numba import njit, prange

#-------------------------------------------------------------------------------
@njit(parallel = True)
def backward_3d_warp(volume, d0, d1, d2, trilin = True, cval = 0.):
  """ Backwarp warp of 3D volume in parallel using numba's njit

  Parameters
  ----------
  volume : 3d numpy array
    containing the image (volume)

  d0, d1, d2 : 3d numpy array   
    containing the 3 components of the deformation field 

  trilin : bool, optional
    whether to use trilinear interpolation (default True)

  cval : float, optional
    value used for filling outside the FOV (default 0)

  Returns
  -------
  3d numpy array : 
    warped volume

  Note
  ----
  The value of the backward warped array is determined as:
  warped_volume[i,j,k] = volume[i - d0[i,j,k], k - d1[i,j,k], j - d2[i,j,k]]
  (using trinlinear interpolation by default)
  """
  # the dimensions of the output volume
  n0, n1, n2 = volume.shape

  output_volume = np.zeros((n0, n1, n2))
  if cval != 0: output_volume += cval

  for i in prange(n0):
    for j in range(n1):
      for k in range(n2):
        tmp_x = i - d0[i,j,k] 
        tmp_y = j - d1[i,j,k]
        tmp_z = k - d2[i,j,k]

        if trilin:
          # trilinear interpolatio mode
          # https://en.wikipedia.org/wiki/Trilinear_interpolation
          x0 = math.floor(tmp_x)  
          x1 = math.ceil(tmp_x)  
          y0 = math.floor(tmp_y)  
          y1 = math.ceil(tmp_y)  
          z0 = math.floor(tmp_z)  
          z1 = math.ceil(tmp_z)  

          if (x0 >= 0) and (x1 < n0) and (y0 >= 0) and (y1 < n1) and (z0 >= 0) and (z1 < n2):
            xd = (tmp_x - x0)
            yd = (tmp_y - y0)
            zd = (tmp_z - z0)

            c00 = volume[x0,y0,z0]*(1 - xd) + volume[x1,y0,z0]*xd
            c01 = volume[x0,y0,z1]*(1 - xd) + volume[x1,y0,z1]*xd
            c10 = volume[x0,y1,z0]*(1 - xd) + volume[x1,y1,z0]*xd
            c11 = volume[x0,y1,z1]*(1 - xd) + volume[x1,y1,z1]*xd

            c0 = c00*(1 - yd) + c10*yd
            c1 = c01*(1 - yd) + c11*yd
 
            output_volume[i,j,k] = c0*(1 - zd) + c1*zd
        else:
          # no interpolation mode
          x = round(tmp_x)
          y = round(tmp_y)
          z = round(tmp_z)

          if ((x >= 0) and (x < n0) and (y >= 0) and (y < n1) and (z >= 0) and (z < n2)):
            output_volume[i,j,k] = volume[x,y,z]

  return output_volume
