import numpy as np
from numba import njit, jit

def mincostpath(img, poscost, cyclic = False, wrapcols = None):
  """ compute a minimum cost path in a cost image, assuming the path consists of a single point per column

  The cost of the entire path is the sum of the local costs. The
  local cost in a pixel has two components:
    1) the cost value of the pixel, obtained from the cost image
    2) the transition cost: a cost can be assigned to the row
       difference of consecutive path points to encourage horizontal lines.

  Parameters
  ----------
  img : 2D numpy array
      the local cost image.
 
  poscost : 1D numpy array
      with the extra cost assigned to the postion of the left
      neighbor:
          poscost[0] is the cost for the direct left neighbor (same row).
          poscost[1] is the cost for the neighbors above and below
                     the direct left neighbor (1 row difference).
          poscost[2] for the left "neighbors" with a 2 row difference.
      and so on.
      The positional costs should be NON-DECREASING with row difference.
      It is recommended to set poscost[0] = 0, and 
      poscost[i+1] >= poscost[i]

  cyclic: bool, default False 
    when set, it is assumed that the image is horizontally
    periodic. To ensure that the last path point connects
    with the first one, a temporary image is created by
    concatenating the first WRAPCOLS columns to the right  and
    the last WRAPCOLS columns to the left.
    This mode allows computation of circular paths by
    applying NImincostpath after polar transform.

  wrapcols: int
    the number of columns used to model the cyclic nature
    of the image. Default is (number_of_columns)/2, so the
    temporary image is twice as wide as the original one.

  Returns
  -------
  1D numpy array
    the minimum cost path consisting of the column indices 

  Note
  ----
  Python reimplementation of J. Nuyts' nimincostpath.pro
  """

  if cyclic:
    if wrapcols is None: wrapcols = img.shape[1] // 2
    tmpimg = np.pad(img, ((0,0),(wrapcols,wrapcols)), 'wrap')
  else:
    tmpimg = img

  res =  mincostpath_backend(tmpimg, poscost)

  if cyclic:
    res = res[wrapcols: (wrapcols + img.shape[1])]

  return res


#-------------------------------------------------------------------------------------------------------  

@njit()
def mincostpath_backend(img, poscost):
  """ compute a minimum cost path in a cost image, assuming the path consists of a single point per column

  The cost of the entire path is the sum of the local costs. The
  local cost in a pixel has two components:
    1) the cost value of the pixel, obtained from the cost image
    2) the transition cost: a cost can be assigned to the row
       difference of consecutive path points to encourage horizontal lines.

  Parameters
  ----------
  img : 2D numpy array
      the local cost image.
 
  poscost
      a 1D numpy array with the extra cost assigned to the postion of the left
      neighbor:
          poscost[0] is the cost for the direct left neighbor (same row).
          poscost[1] is the cost for the neighbors above and below
                     the direct left neighbor (1 row difference).
          poscost[2] for the left "neighbors" with a 2 row difference.
      and so on.
      The positional costs should be NON-DECREASING with row difference.
      It is recommended to set poscost[0] = 0, and 
      poscost[i+1] >= poscost[i]

  Returns
  -------
  1D numpy array
    the minimum cost path consisting of the column indices 

  Note
  ----
  Python reimplementation of J. Nuyts' NCmincostpath.c
  """
  costimg  = img.copy()
  n0, n1   = costimg.shape
  nposcost = poscost.shape[0] 
  colimg   = np.zeros(costimg.shape, dtype = np.int32)
  
  for j in range(1,n1):
    for i in range(0, n0):
      # calculate cost of immediate left neighbor
      mincost = costimg[i,j-1] + poscost[0]
      minleft = i
  
      # compare to "upper" left neighbors
      npos = min(nposcost - 1, i)
      for offset in range(1, npos + 1):
        cost = costimg[i - offset, j-1] + poscost[offset]
  
        if cost < mincost:
          mincost = cost
          minleft = i - offset
  
      # compare to "lower" left neighbors
      npos = min(nposcost - 1, n0 - i - 1)
      for offset in range(1, npos + 1):
        cost = costimg[i + offset, j-1] + poscost[offset]
  
        if cost < mincost:
          mincost = cost
          minleft = i + offset
  
      costimg[i,j] += mincost
      colimg[i,j]   = minleft
  
  # now costimg contains the costs of the paths from the left, and
  # colimg contains the min cost left neighbor of each pixel.
  # so the last point of the contour can be found by selecting
  # the pixel with min cost in the last column.
  
  path       = np.zeros(n1, dtype = np.int32)
  last_point = np.argmin(costimg[:,-1])
  path[-1]   = last_point
  
  for jback in range(1,n1):
    last_point           = colimg[last_point, n1 - jback]
    path[n1 - jback - 1] = last_point 

  return path
