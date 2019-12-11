import math
import numpy
from numba import njit, stencil

@stencil
def fwd_diff2d_0(x):
  return x[1,0] - x[0,0]

@stencil
def back_diff2d_0(x):
  return x[0,0] - x[-1,0]

@stencil
def fwd_diff2d_1(x):
  return x[0,1] - x[0,0]

@stencil
def back_diff2d_1(x):
  return x[0,0] - x[0,-1]

#-----------------------------------------------------------------------------

@stencil
def fwd_diff3d_0(x):
  return x[1,0,0] - x[0,0,0]

@stencil
def back_diff3d_0(x):
  return x[0,0,0] - x[-1,0,0]

@stencil
def fwd_diff3d_1(x):
  return x[0,1,0] - x[0,0,0]

@stencil
def back_diff3d_1(x):
  return x[0,0,0] - x[0,-1,0]

@stencil
def fwd_diff3d_2(x):
  return x[0,0,1] - x[0,0,0]

@stencil
def back_diff3d_2(x):
  return x[0,0,0] - x[0,0,-1]

#-----------------------------------------------------------------------------

@stencil
def fwd_diff4d_0(x):
  return x[1,0,0,0] - x[0,0,0,0]

@stencil
def back_diff4d_0(x):
  return x[0,0,0,0] - x[-1,0,0,0]

@stencil
def fwd_diff4d_1(x):
  return x[0,1,0,0] - x[0,0,0,0]

@stencil
def back_diff4d_1(x):
  return x[0,0,0,0] - x[0,-1,0,0]

@stencil
def fwd_diff4d_2(x):
  return x[0,0,1,0] - x[0,0,0,0]

@stencil
def back_diff4d_2(x):
  return x[0,0,0,0] - x[0,0,-1,0]

@stencil
def fwd_diff4d_3(x):
  return x[0,0,0,1] - x[0,0,0,0]

@stencil
def back_diff4d_3(x):
  return x[0,0,0,0] - x[0,0,0,-1]


#-----------------------------------------------------------------------------

@njit(parallel = True)
def grad2d(x, g):
  fwd_diff2d_0(x, out = g[0,]) 
  fwd_diff2d_1(x, out = g[1,]) 

@njit(parallel = True)
def grad3d(x, g):
  fwd_diff3d_0(x, out = g[0,]) 
  fwd_diff3d_1(x, out = g[1,]) 
  fwd_diff3d_2(x, out = g[2,]) 

@njit(parallel = True)
def grad4d(x, g):
  fwd_diff4d_0(x, out = g[0,]) 
  fwd_diff4d_1(x, out = g[1,]) 
  fwd_diff4d_2(x, out = g[2,]) 
  fwd_diff4d_3(x, out = g[3,]) 

def grad(x,g):
  """
  Calculate the gradient of 2d,3d, or 4d array via the finite forward diffence

  Arguments
  ---------
  
  x ... a 2d, 3d, or 4d numpy array
  g ... (output) array of size ((x.ndim,), x.shape) used to store the ouput

  Examples
  --------

  import numpy
  import pynucmed
  x = numpy.random.rand(20,20,20)
  g = numpy.zeros((x.ndim,) + x.shape) 
  pynucmed.misc.grad(x,g)
  y = pynucmed.misc.div(g) 

  Note
  ----

  This implementation uses the numba stencil decorators in combination with
  jit in parallel nopython mode

  """
  ndim = x.ndim
  if   ndim == 2: grad2d(x, g)
  elif ndim == 3: grad3d(x, g)
  elif ndim == 4: grad4d(x, g)
  else          : raise TypeError('Invalid dimension of input') 

#-----------------------------------------------------------------------------

def complex_grad(x,g):
  """
  Calculate the gradient of 2d,3d, or 4d complex array via the finite forward diffence

  Arguments
  ---------
  
  x ... a complex numpy arrays represented by 2 float arrays
        2D ... x.shape = [n0,n1,2]
        3D ... x.shape = [n0,n1,n2,2]
        4D ... x.shape = [n0,n1,n2,n3,2]

  g ... (output) array of size ((2*x[...,0].ndim,), x[...,0].shape) used to store the ouput

  Note
  ----

  This implementation uses the numba stencil decorators in combination with
  jit in parallel nopython mode.
  The gradient is calculated separately for the real and imag part and
  concatenated together.

  """
  ndim = x[...,0].ndim
  if   ndim == 2: 
    grad2d(x[...,0], g[:ndim,...])
    grad2d(x[...,1], g[ndim:,...])
  elif ndim == 3: 
    grad3d(x[...,0], g[:ndim,...])
    grad3d(x[...,1], g[ndim:,...])
  elif ndim == 4: 
    grad4d(x[...,0], g[:ndim,...])
    grad4d(x[...,1], g[ndim:,...])
  else          : raise TypeError('Invalid dimension of input') 

#-----------------------------------------------------------------------------

@njit(parallel = True)
def div2d(g):
  tmp = numpy.zeros(g.shape)
  back_diff2d_0(g[0,], out = tmp[0,]) 
  back_diff2d_1(g[1,], out = tmp[1,]) 

  return tmp[0,] + tmp[1,]

@njit(parallel = True)
def div3d(g):
  tmp = numpy.zeros(g.shape)
  back_diff3d_0(g[0,], out = tmp[0,]) 
  back_diff3d_1(g[1,], out = tmp[1,]) 
  back_diff3d_2(g[2,], out = tmp[2,]) 
 
  return tmp[0,] + tmp[1,] + tmp[2,]

@njit(parallel = True)
def div4d(g):
  tmp = numpy.zeros(g.shape)
  back_diff4d_0(g[0,], out = tmp[0,]) 
  back_diff4d_1(g[1,], out = tmp[1,]) 
  back_diff4d_2(g[2,], out = tmp[2,]) 
  back_diff4d_3(g[3,], out = tmp[3,]) 
 
  return tmp[0,] + tmp[1,] + tmp[2,] + tmp[3,]

def div(g):
  """
  Calculate the divergence of 2d, 3d, or 4d array via the finite backward diffence

  Arguments
  ---------
  
  g ... a gradient array of size ((x.ndim,), x.shape)

  Returns
  -------

  an array of size g.shape[1:]

  Examples
  --------

  import numpy
  import pynucmed
  x = numpy.random.rand(20,20,20)
  g = numpy.zeros((x.ndim,) + x.shape) 
  pynucmed.misc.grad(x,g)
  y = pynucmed.misc.div(g) 

  Note
  ----

  This implementation uses the numba stencil decorators in combination with
  jit in parallel nopython mode

  See also
  --------

  pynucmed.misc.grad
  """
  ndim = g.shape[0]
  if   ndim == 2: return div2d(g)
  elif ndim == 3: return div3d(g)
  elif ndim == 4: return div4d(g)
  else          : raise TypeError('Invalid dimension of input') 


def complex_div(g):
  """
  Calculate the divergence of 2d, 3d, or 4d "complex" array via the finite backward diffence

  Arguments
  ---------
  
  g ... a gradient array of size (2*(x.ndim,), x.shape)

  Returns
  -------

  a real array of shape (g.shape[1:] + (2,)) representing the complex array by 2 real arrays

  Note
  ----

  This implementation uses the numba stencil decorators in combination with
  jit in parallel nopython mode

  See also
  --------

  pynucmed.misc.grad
  """

  ndim = g.shape[0] // 2
  tmp  = numpy.zeros(g.shape[1:] + (2,))

  if ndim == 2: 
    tmp[...,0] = div2d(g[:ndim,...])
    tmp[...,1] = div2d(g[ndim:,...])
  elif ndim == 3: 
    tmp[...,0] = div3d(g[:ndim,...])
    tmp[...,1] = div3d(g[ndim:,...])
  elif ndim == 4: 
    tmp[...,0] = div4d(g[:ndim,...])
    tmp[...,1] = div4d(g[ndim:,...])
  else: raise TypeError('Invalid dimension of input') 

  return tmp
