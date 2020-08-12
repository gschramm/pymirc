import numpy as np

from scipy.optimize import minimize

from pymirc.image_operations import kul_aff, aff_transform
from pymirc.metrics.cost_functions import neg_mutual_information, regis_cost_func

def rigid_registration(vol_float, vol_fixed, aff_float, aff_fixed,
                       downsample_facs = [4], metric = neg_mutual_information,
                       opts = {'ftol':1e-2,'xtol':1e-2,'disp':True,'maxiter':20,'maxfev':5000},
                       method = 'Powell', metric_kwargs = {}):
  """ rigidly coregister a 3D floating volume to a fixed (reference) volume

  Parameters
  ----------
  vol_float : 3D numpy array
    the floating volume

  vol_fixed : 3D numpy array
    the fixed (reference) volume

  aff_float : 2D 4x4 numpy array
    affine transformation matrix that maps from pixel to world coordinates for floating volume

  aff_fixed : 2D 4x4 numpy array
    affine transformation matrix that maps from pixel to world coordinates for fixed volume

  downsample_facs : None of array_like
    perform registrations on downsampled grids before registering original volumes
    (multi-resolution approach)

  metric : function(x,y, **kwargs)
    metric that compares transformed floating and fixed volume
 
  metric_kwargs : dict
    keyword arguments passed to the metric function

  opts : dictionary
    passed to scipy.optimize.minimize as options

  method : string
    passed to scipy.optimize.minimize as method (which optimizer to use)

  Returns
  -------
  tuple of 3
  - the transformed (coregistered) floating volume
  - the registration affine transformation matrix 
  - the 6 registration parameters (3 translations, 3 rotations) from which the
    affine matrix was derived

  Note
  ----

  To apply the registration affine transformation use:
  new_vol = aff_transform(vol, reg_aff, vol_fixed.shape, cval = vol.min())
  
  """
  # define the affine transformation that maps the floating to the fixed voxel grid
  # we need this to avoid the additional interpolation in case the voxel sizes are not the same
  pre_affine = np.linalg.inv(aff_float) @ aff_fixed
  
  reg_params = np.zeros(6)

  # (1) initial registration with downsampled arrays
  if downsample_facs is not None:
    for dsf in downsample_facs:
      ds_aff = np.diag([dsf,dsf,dsf,1.])
      
      # down sample fixed volume
      vol_fixed_ds = aff_transform(vol_fixed, ds_aff, 
                                        np.ceil(np.array(vol_fixed.shape)/dsf).astype(int))
      
      res = minimize(regis_cost_func, reg_params, method = method, options = opts,
                     args = (vol_fixed_ds, vol_float, True, True, metric, pre_affine @ ds_aff,
                             metric_kwargs))
      
      reg_params = res.x.copy()
      # we have to scale the translations by the down sample factor since they are in voxels
      reg_params[:3] *= dsf
 
 
  # (2) registration with full arrays
  res = minimize(regis_cost_func, reg_params, method = method, options = opts, 
                 args = (vol_fixed, vol_float, True, True, metric, pre_affine,
                         metric_kwargs))
  reg_params = res.x.copy()
  
  
  # define the final affine transformation that maps from the PET grid to the rotated CT grid
  reg_aff  = pre_affine @ kul_aff(reg_params, origin = np.array(vol_fixed.shape)/2)

  # transform the floating volume
  vol_float_coreg = aff_transform(vol_float, reg_aff, vol_fixed.shape, cval = vol_float.min())

  return vol_float_coreg, reg_aff, reg_params
