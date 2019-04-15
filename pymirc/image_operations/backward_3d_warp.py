import numpy as np
import math

from numba import njit, prange

#-------------------------------------------------------------------------------
@njit(parallel = True)
def backward_3d_warp(volume, d0, d1, d2, trilin = True, cval = 0., os0 = 1, os1 = 1, os2 = 1):
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

  os0, os1, os2 : int
    oversampling factors in the three directions (defaut 1 -> no oversampling)
    those factors are needed when going from big to small voxels

  Returns
  -------
  3d numpy array : 
    warped volume
  """
  # the dimensions of the output volume
  n0, n1, n2 = volume.shape

  # the dimenstion of the input volume
  n0_in, n1_in, n2_in = volume.shape

  # the sizes of the temporary oversampled array
  # the oversampling is needed in case we go from
  # small voxels to big voxels
  n0_os = n0*os0
  n1_os = n1*os1
  n2_os = n2*os2

  os_output_volume = np.zeros((n0_os, n1_os, n2_os))
  if cval != 0: os_output_volume += cval

  for i in prange(n0_os):
    for j in range(n1_os):
      for k in range(n2_os):
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

          if (x0 >= 0) and (x1 < n0_in) and (y0 >= 0) and (y1 < n1_in) and (z0 >= 0) and (z1 < n2_in):
            xd = (tmp_x - x0)
            yd = (tmp_y - y0)
            zd = (tmp_z - z0)

            c00 = volume[x0,y0,z0]*(1 - xd) + volume[x1,y0,z0]*xd
            c01 = volume[x0,y0,z1]*(1 - xd) + volume[x1,y0,z1]*xd
            c10 = volume[x0,y1,z0]*(1 - xd) + volume[x1,y1,z0]*xd
            c11 = volume[x0,y1,z1]*(1 - xd) + volume[x1,y1,z1]*xd

            c0 = c00*(1 - yd) + c10*yd
            c1 = c01*(1 - yd) + c11*yd
 
            os_output_volume[i,j,k] = c0*(1 - zd) + c1*zd

        else:
          # no interpolation mode
          x = round(tmp_x)
          y = round(tmp_y)
          z = round(tmp_z)

          if ((x >= 0) and (x < n0_in) and (y >= 0) and (y < n1_in) and (z >= 0) and (z < n2_in)):
            os_output_volume[i,j,k] = volume[x,y,z]

  if os0 == 1 and os1 == 1 and os2 == 1:
    # case without oversampling
    output_volume = os_output_volume
  else:
    output_volume = np.zeros((n0, n1, n2))
    # case with oversampling, we have to average neighbors
    for i in prange(n0):
      for j in range(n1):
        for k in range(n2):
          for ii in range(os0):
            for jj in range(os1):
              for kk in range(os2):
                output_volume[i,j,k] += os_output_volume[i*os0 + ii, j*os1 + jj, k*os2 + kk] / (os0*os1*os2)

  return output_volume
