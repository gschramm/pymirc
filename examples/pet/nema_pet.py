# script to perform NEMA 2008 analysis on small animal IQ phantom

import math
import numpy as np
import pylab as py
import pandas as pd
import matplotlib.patches as patches

import pymirc.metrics     as pymr

from pymirc.image_operations import kul_aff, aff_transform

from scipy.ndimage import label, labeled_comprehension, find_objects, gaussian_filter
from scipy.ndimage import binary_erosion, median_filter
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes

from scipy.special    import erf
from scipy.integrate  import quad

from scipy.signal   import argrelextrema, find_peaks_cwt
from scipy.optimize import minimize

from lmfit    import Model

def find_background_roi(vol, voxsize, Rcenter = 82, edge_margin = 28.):
  """ find the 2D background ROI for a NEMA sphere phantom

  Parameters
  ----------

  vol : 3d numpy array
    containing the volume

  voxsize: 1d numpy array 
    containing the voxel size (mm)

  Rcenter: float (optional)
    the radius (mm) of the sphere part that is not considered - default 82.

  edge_margin : float (optional)
    margin (mm) to stay away from the boarder of the phantom - default 28.

  Returns
  -------

  A tuple containing the indices of the background voxels.
  """

  # find the background value from a histogram analysis
  h  = np.histogram(vol[vol>0.01*vol.max()].flatten(),200)
  bg = 0.5*(h[1][np.argmax(h[0])] + h[1][np.argmax(h[0]) + 1]) 
  
  # get the axial slice with the maximum activity (spheres)
  zprof = vol.sum((0,1))
  sphere_sl = np.argmax(zprof)
  
  sphere_2d_img = vol[...,sphere_sl]
  
  bg_mask = binary_fill_holes(median_filter(np.clip(sphere_2d_img,0,0.8*bg), size = 7) > 0.5*bg)
  
  # erode mask by ca. 2.8 cm to stay away from the boundary
  nerode = int(edge_margin / voxsize[0])
  if (nerode % 2) == 0: nerode += 1
  
  bg_mask = binary_erosion(bg_mask, np.ones((nerode,nerode))) 
  
  # set center where spheres are to 0
  com = center_of_mass(binary_fill_holes(sphere_2d_img < 0.5*bg))
  
  x = voxsize[0]*(np.arange(vol.shape[0]) - com[0]) - 5
  y = voxsize[1]*(np.arange(vol.shape[1]) - com[1])
  
  X,Y = np.meshgrid(x,y)
  R = np.sqrt(X**2 + Y**2)
  
  bg_mask[R<=Rcenter] = 0
  
  # generate voxel indices for background voxels
  tmp = np.zeros(vol.shape, dtype = np.int8)
  tmp[...,sphere_sl] = bg_mask
  bg_inds = np.where(tmp == 1)

  return bg_inds


#----------------------------------------------------------------------  
def cylinder_prof_integrand(eta, z, Z):
  """ Integrand of the convolution of a disk convolved with a 2D Gaussian

  Parameters
  ----------
  eta : float
    integration variable

  z : float
    radial distance from center divided by sigma of Gaussian

  Z : float
    disk radius divided by sigma of Gaussian

  Returns
  -------
  float
  """
  return math.exp(-0.5*eta**2) * (erf( (math.sqrt(Z**2 - eta**2) - z)/math.sqrt(2) ) + 
                                  erf( (math.sqrt(Z**2 - eta**2) + z)/math.sqrt(2)))

#----------------------------------------------------------------------  
def cylinder_prof(z, Z):
  """ Normalized radial profile of a disk convolved with a 2D Gaussian

  Parameters
  ----------
  z : float
    radial distance from center divided by sigma of Gaussian

  Z : float
    disk radius divided by sigma of Gaussian

  Returns
  -------
  float

  Note
  ----
  There is no analytic expression for the convolution.
  The profile is numerically integrated using quad() from scipy.integrate
  """
  return quad(cylinder_prof_integrand, 0, Z, args = (z,Z))[0] / math.sqrt(2*math.pi)

#----------------------------------------------------------------------  
def cylinder_profile(r, S = 1., R = 1., fwhm = 1.):
  """ Radial profile of a disk convolved with a 2D Gaussian

  Parameters
  ----------
  r : 1D numpy float array
    radial distance from center

  S : float
    signal in the disk

  R : float
    radius of the disk

  fwhm : float
    FWHM of the Gaussian smoothing kernel

  Returns
  -------
  1D numpy float array
  """
  sig = fwhm / 2.35
  
  cp = np.frompyfunc(cylinder_prof, 2, 1)

  return S*cp(r/sig, R/sig).astype(float)

#----------------------------------------------------------------------  
def fit_nema_2008_cylinder_profiles(vol, 
                                    voxsize,
                                    Rrod_init  = [2.5,2,1.5,1,0.5],
                                    fwhm_init  = 1.5,
                                    S_init     = 1,
                                    fix_S      = True,
                                    fix_R      = False,
                                    fix_fwhm   = False,
                                    nrods      = 4):
  """ Fit the radial profiles of the rods in a nema 2008 small animal PET phantom

  Parameters
  ----------
  vol : 3D numpy float array
    containing the image

  voxsize : 3 element 1D numpy array
    containing the voxel size

  Rrod_init : list or 1D numpy array of floats, optional
    containing the initial values of the rod radii

  S_init, fwhm_init: float, optional
    initial values for the signal and the FWHM in the fit

  fix_S, fix_R, fix_fwhm : bool, optional
    whether to keep the initial values of signal, radius and FWHM fixed during the fix

  nrods: int, optional
    number of rods to fit

  Returns
  -------
  a list of lmfit fit results

  Note
  ----

  The axial direction should be the right most direction in the 3D numpy array.
  The slices containing the rods are found automatically and summed.
  In the summed image, all rods (disks) are segmented followed by a fit
  of the radial profile.
  """
  roi_vol = nema_2008_small_animal_pet_rois(vol, voxsize)
  
  rod_bbox = find_objects(roi_vol==4)
  
  # find the rods in the summed image
  sum_img = vol[:,:,rod_bbox[0][2].start:rod_bbox[0][2].stop].mean(2)
  
  label_img, nlab = label(sum_img > 0.1*sum_img.max())
  labels = np.arange(1,nlab+1)
  # sort the labels according to volume
  npix   = labeled_comprehension(sum_img, label_img, labels, len, int, 0)
  sort_inds = npix.argsort()[::-1]
  labels    = labels[sort_inds] 
  npix      = npix[sort_inds] 
  
  #----------------------------------------------------------------------  
  ncols = 2
  nrows = int(np.ceil(nrods/ncols))
  fig, ax = py.subplots(nrows,ncols,figsize = (12,7*nrows/2), sharey = True, sharex = True)
 
  retval = []
 
  for irod in range(nrods):
    rod_bbox = find_objects(label_img == labels[irod])
   
    rod_bbox = [(slice(rod_bbox[0][0].start - 2,rod_bbox[0][0].stop + 2),
                 slice(rod_bbox[0][1].start - 2,rod_bbox[0][1].stop + 2))]
   
    rod_img = sum_img[rod_bbox[0]]
    com     = np.array(center_of_mass(rod_img))
    
    x0 = (np.arange(rod_img.shape[0]) - com[0]) * voxsize[0]
    x1 = (np.arange(rod_img.shape[1]) - com[1]) * voxsize[1]
    
    X0, X1 = np.meshgrid(x0, x1, indexing = 'ij')
    RHO    = np.sqrt(X0**2 + X1**2) 
    
    rho    = RHO.flatten()
    signal = rod_img.flatten()
    
    # sort the values according to rho
    sort_inds = rho.argsort()
    rho       = rho[sort_inds]
    signal    = signal[sort_inds]
    
    pmodel = Model(cylinder_profile)
    params = pmodel.make_params(S = S_init, R = Rrod_init[irod], fwhm = fwhm_init)

    if fix_S:
      params['S'].vary = False
    if fix_R:
      params['R'].vary = False
    if fix_fwhm:
      params['fwhm'].vary = False
    
    fitres = pmodel.fit(signal, r = rho, params = params)
    retval.append(fitres)
    fit_report = fitres.fit_report()
   
    iplot = np.unravel_index(irod, ax.shape) 
    ax[iplot].plot(rho,signal,'k.')
    
    rfit = np.linspace(0,rho.max(),100)
    ax[iplot].plot(rfit,fitres.eval(r = rfit),'r-')
    ax[iplot].text(0.99, 0.99, fit_report, fontsize = 6, transform = ax[iplot].transAxes, 
                         verticalalignment='top', horizontalalignment = 'right',
                         backgroundcolor = 'white', bbox = {'pad':0, 'facecolor':'white','lw':0})
    ax[iplot].grid()
  
  for axx in ax[-1,:]: axx.set_xlabel('R (mm)')
  for axx in ax[:,0]:  axx.set_ylabel('signal')
  
  fig.tight_layout()
  fig.show()

  return retval

#----------------------------------------------------------------------  
def nema_2008_small_animal_pet_rois(vol, voxsize, lp_voxel = 'max', rod_th = 0.2):
  """ generate a label volume indicating the ROIs needed in the analysis of the
      NEMA small animal PET IQ phantom

  Parameters
  ----------
  vol : 3D numpy float array
    containing the image

  voxsize : 3 element 1D numpy array
    containing the voxel size

  lp_voxel: string, optional
    method of how to compute the pixel used to draw the line profiles
    in the rods. 'max' means the maximum voxels in the summed 2D image.
    anything else means use the center of mass.
 
  rod_th : float, optional
    threshold to find the rod in the summed 2D image relative to the
    mean of the big uniform region
 
  Returns
  -------
  a 3D integer numpy array
    encoding the following ROIs:
    1 ... ROI of the big uniform region
    2 ... first cold insert
    3 ... second cold insert
    4 ... central line profile in 5mm rod
    5 ... central line profile in 4mm rod
    6 ... central line profile in 3mm rod
    7 ... central line profile in 2mm rod
    8 ... central line profile in 1mm rod

  Note
  ----
  The rod ROIs in the summed 2D image are found by thresholding.
  If the activity in the small rods is too low, they might be missed.
  """
  roi_vol = np.zeros(vol.shape, dtype = np.uint)
  
  # calculate the summed z profile to place the ROIs
  zprof      = vol.sum(0).sum(0)
  zprof_grad = np.gradient(zprof)
  zprof_grad[np.abs(zprof_grad) < 0.1*np.abs(zprof_grad).max()] = 0
  
  rising_edges  = argrelextrema(zprof_grad, np.greater, order = 10)[0]
  falling_edges = argrelextrema(zprof_grad, np.less, order = 10)[0]
  
  # define and analyze the big uniform ROI
  uni_region_start_slice  = rising_edges[1]
  uni_region_end_slice    = falling_edges[1]
  uni_region_center_slice = 0.5*(uni_region_start_slice + uni_region_end_slice) 
  
  uni_roi_start_slice = int(np.floor(uni_region_center_slice - 5./voxsize[2]))
  uni_roi_end_slice   = int(np.ceil(uni_region_center_slice  + 5./voxsize[2]))
  
  uni_com = np.array(center_of_mass(vol[:,:,uni_roi_start_slice:(uni_roi_end_slice+1)]))
  
  x0 = (np.arange(vol.shape[0]) - uni_com[0]) * voxsize[0]
  x1 = (np.arange(vol.shape[1]) - uni_com[1]) * voxsize[1]
  x2 = (np.arange(vol.shape[2]) - uni_com[2]) * voxsize[2]
  
  X0,X1,X2 = np.meshgrid(x0,x1,x2,indexing='ij')
  RHO      = np.sqrt(X0**2 + X1**2)
  
  uni_mask = np.zeros(vol.shape, dtype = np.uint)
  uni_mask[RHO <= 11.25] = 1
  uni_mask[:,:,:uni_roi_start_slice]   = 0
  uni_mask[:,:,(uni_roi_end_slice+1):] = 0
  
  uni_inds = np.where(uni_mask == 1)
  roi_vol[uni_inds] = 1
  
  # define and analyze the two cold ROIs
  
  insert_region_start_slice  = falling_edges[1]
  insert_region_end_slice    = falling_edges[2]
  insert_region_center_slice = 0.5*(insert_region_start_slice + insert_region_end_slice) 
  
  insert_roi_start_slice = int(np.floor(insert_region_center_slice - 3.75/voxsize[2]))
  insert_roi_end_slice   = int(np.ceil(insert_region_center_slice  + 3.75/voxsize[2]))
  
  # sum the insert slices and subtract them from the max to find the two cold inserts
  sum_insert_img = vol[:,:,insert_roi_start_slice:(insert_roi_end_slice+1)].mean(2)
  
  insert_label_img, nlab_insert = label(sum_insert_img <= rod_th*vol[uni_inds].mean())
  insert_labels = np.arange(1,nlab_insert+1)
  # sort the labels according to volume
  npix_insert   = labeled_comprehension(sum_insert_img, insert_label_img, insert_labels, len, int, 0)
  insert_sort_inds = npix_insert.argsort()[::-1]
  insert_labels    = insert_labels[insert_sort_inds] 
  npix_insert      = npix_insert[insert_sort_inds] 
  
  for i_insert in [1,2]:
    tmp = insert_label_img.copy()
    tmp[insert_label_img != insert_labels[i_insert]] = 0
    com_pixel = np.round(np.array(center_of_mass(tmp)))
  
    x0 = (np.arange(vol.shape[0]) - com_pixel[0]) * voxsize[0]
    x1 = (np.arange(vol.shape[1]) - com_pixel[1]) * voxsize[1]
    x2 = (np.arange(vol.shape[2])) * voxsize[2]
    
    X0,X1,X2 = np.meshgrid(x0,x1,x2,indexing='ij')
    RHO      = np.sqrt(X0**2 + X1**2)
  
    insert_mask = np.zeros(vol.shape, dtype = np.uint)
    insert_mask[RHO <= 2] = 1
    insert_mask[:,:,:insert_roi_start_slice]   = 0
    insert_mask[:,:,(insert_roi_end_slice+1):] = 0
  
    insert_inds = np.where(insert_mask == 1)
    roi_vol[insert_inds] = i_insert + 1
  
  # find the rod z slices
  rod_start_slice = falling_edges[0]
  rod_end_slice   = rising_edges[1]
  rod_center      = 0.5*(rod_start_slice + rod_end_slice)
  
  rod_roi_start_slice = int(np.floor(rod_center - 5./voxsize[2]))
  rod_roi_end_slice   = int(np.ceil(rod_center  + 5./voxsize[2]))
  
  # sum the rod slices
  sum_img = vol[:,:,rod_roi_start_slice:(rod_roi_end_slice+1)].mean(2)
  
  # label the summed image
  label_img, nlab = label(sum_img > 0.12*sum_img.max())
  labels = np.arange(1,nlab+1)
  # sort the labels according to volume
  npix   = labeled_comprehension(sum_img, label_img, labels, len, int, 0)
  sort_inds = npix.argsort()[::-1]
  labels    = labels[sort_inds] 
  npix      = npix[sort_inds] 
  
  # find the center for the line profiles
  for i, lab in enumerate(labels):
    rod_sum_img = sum_img.copy()
    rod_sum_img[label_img != lab] = 0
 
    if lp_voxel == 'max':
      central_pixel = np.unravel_index(rod_sum_img.argmax(),rod_sum_img.shape)
    else:
      central_pixel = np.round(np.array(center_of_mass(rod_sum_img))).astype(np.int)
  
    roi_vol[central_pixel[0],central_pixel[1],rod_roi_start_slice:(rod_roi_end_slice+1)] = i + 4

  return roi_vol

#--------------------------------------------------------------------
def nema_2008_small_animal_iq_phantom(voxsize, shape):
  """ generate a digital version of the upper part of the NEMA small animal PET
      IQ phantom that can be used to align a NEMA scan

  Parameters
  ----------
  voxsize : 3 element 1D numpy array
    containing the voxel size

  shape: 3 element tuple of integers
    shape of the volume
 
  Returns
  -------
    a 3D numpy array
  """
  x0 = (np.arange(shape[0]) - 0.5*shape[0] - 0.5) * voxsize[0]
  x1 = (np.arange(shape[1]) - 0.5*shape[1] - 0.5) * voxsize[1]
  x2 = (np.arange(shape[2]) - 0.5*shape[2] - 0.5) * voxsize[2]
  
  X0,X1,X2 = np.meshgrid(x0,x1,x2,indexing='ij')
  RHO      = np.sqrt(X0**2 + X1**2)
    
  phantom = np.zeros(shape)
  phantom[RHO <= 30./2] = 1
  phantom[X2 < 0]       = 0
  phantom[X2 > 33.]     = 0
  
  RHO1 = np.sqrt((X0 - 7.5)**2 + X1**2)
  RHO2 = np.sqrt((X0 + 7.5)**2 + X1**2)
  
  phantom[np.logical_and(RHO1 <= 9.2/2, X2 > 15)] = 0
  phantom[np.logical_and(RHO2 <= 9.2/2, X2 > 15)] = 0

  return phantom

#--------------------------------------------------------------------
def align_nema_2008_small_animal_iq_phantom(vol, voxsize, ftol = 1e-2, xtol = 1e-2, maxiter = 10, maxfev = 500):
  """ align a reconstruction of the NEMA small animal PET IQ phantom to its digital version

  Parameters
  ----------
  vol : 3D numpy float array
    containing the image

  voxsize : 3 element 1D numpy array
    containing the voxel size

  ftol, xtol, maxiter, maxfev : float / int
    parameter for the optimizer used to minimze the cost function

  Returns
  -------
    a 3D numpy array

  Note
  ----
  This routine can be useful to make sure that the rods in the NEMA scan are
  parallel to the axial direction.
  """
  phantom  = nema_2008_small_animal_iq_phantom(voxsize, vol.shape)
  phantom *= vol[vol>0.5*vol.max()].mean()

  reg_params = np.zeros(6)

  # registration of down sampled volumes
  dsf    = 3
  ds_aff = np.diag([dsf,dsf,dsf,1.])
  
  phantom_ds = aff_transform(phantom, ds_aff, np.ceil(np.array(phantom.shape)/dsf).astype(int))

  res = minimize(pymr.regis_cost_func, reg_params, 
                 args = (phantom_ds, vol, True, True, lambda x,y: ((x-y)**2).mean(), ds_aff), 
                 method = 'Powell', 
                 options = {'ftol':ftol, 'xtol':xtol, 'disp':True, 'maxiter':maxiter, 'maxfev':maxfev})

  reg_params = res.x.copy()
  # we have to scale the translations by the down sample factor since they are in voxels
  reg_params[:3] *= dsf

  res = minimize(pymr.regis_cost_func, reg_params, 
                 args = (phantom, vol, True, True, lambda x,y: ((x-y)**2).mean()), 
                 method = 'Powell', 
                 options = {'ftol':ftol, 'xtol':xtol, 'disp':True, 'maxiter':maxiter, 'maxfev':maxfev})
  
  regis_aff   = kul_aff(res.x, origin = np.array(vol.shape)/2)
  vol_aligned = aff_transform(vol, regis_aff, vol.shape) 
  
  return vol_aligned

#--------------------------------------------------------------------
def nema_2008_small_animal_iq_phantom_report(vol, roi_vol):
  """ generate the report for the NEMA 2008 small animal PET IQ phantom analysis

  Parameters
  ----------
  vol : 3D numpy float array
    containing the image

  roi_vol : 3D numpy float array
    containing the ROI label image with following ROIs:
    1 ... ROI of the big uniform region
    2 ... first cold insert
    3 ... second cold insert
    4 ... central line profile in 5mm rod
    5 ... central line profile in 4mm rod
    6 ... central line profile in 3mm rod
    7 ... central line profile in 2mm rod
    8 ... central line profile in 1mm rod

  Returns
  -------
    a 3D numpy array

  Note
  ----
  The ROIs for the smaller rods are optional.
  """
  np.set_printoptions(precision=3)
  
  # get the ROI values of the big uniform ROI
  uni_values    = vol[roi_vol == 1]
  uni_mean      = uni_values.mean()
  uni_max       = uni_values.max()
  uni_min       = uni_values.min()
  uni_std       = uni_values.std()
  uni_perc_std  = 100 * uni_std / uni_mean
  
  print("\nuniform ROI results")
  print("------------------------------")
  print("mean ...:", "%.3f" % uni_mean)
  print("max  ...:", "%.3f" % uni_max)
  print("min  ...:", "%.3f" % uni_min)
  print("%std ...:", "%.3f" % uni_perc_std, "\n")
  
  # get the ROI values of the 2 cold inserts
  insert_mean = np.zeros(2)
  insert_std  = np.zeros(2)
  
  insert1_values = vol[roi_vol == 2]
  insert_mean[0] = insert1_values.mean()
  insert_std[0]  = insert1_values.std()
  
  insert2_values = vol[roi_vol == 3]
  insert_mean[1] = insert2_values.mean()
  insert_std[1]  = insert2_values.std()
  
  insert_ratio    = insert_mean / uni_mean
  insert_perc_std = 100 * np.sqrt((insert_std/insert_mean)**2 + (uni_std/uni_mean)**2)
  
  print("\ncold insert results")
  print("------------------------------")
  print("spill over ratio ...:", insert_ratio)
  print("%std             ...:", insert_perc_std, "\n")
  
  # analyze the rod profiles
  nrods   = int(roi_vol.max() - 3)
  lp_mean = np.zeros(nrods)
  lp_std  = np.zeros(nrods)
  
  # find the center for the line profiles
  for i in range(nrods):
    lp_values  = vol[roi_vol == i + 4]
    lp_mean[i] = lp_values.mean()
    lp_std[i]  = lp_values.std()
  
  lp_rc = lp_mean / uni_mean
  lp_perc_std = 100 * np.sqrt((lp_std/lp_mean)**2 + (uni_std/uni_mean)**2)
  
  print("\nrod results")
  print("------------------------------")
  print("recovery coeff...:", lp_rc)
  print("%std          ...:", lp_perc_std, "\n")
  
  
  np.set_printoptions(precision=None)

#--------------------------------------------------------------------------------------------------

def gausssphere_profile(z = np.linspace(0, 2, 100), 
                        Z = 0.8):
    """ Radial profile of a sphere convolved with a 3D radial symmetric Gaussian

     Parameters
     ----------
     z : 1D numpy float array 
       normalized radial coordinate    (r / (sqrt(2) * sigma))

     Z : float
       normalized radius of the sphere (R / (sqrt(2) * sigma))

    Returns
    -------
    1D numpy array
    """

    sqrtpi = np.sqrt(np.pi)
    
    P = np.zeros_like(z)

    inds0 = np.argwhere(z == 0)
    inds1 = np.argwhere(z != 0)
    
    P[inds0] = erf(Z) - 2 * Z * np.exp(-Z**2) / sqrtpi

    P[inds1] = ( 0.5 * (erf(z[inds1] + Z) - erf(z[inds1] - Z)) - 
                (0.5/sqrtpi) * ((np.exp(-(z[inds1] - Z)**2) - np.exp(-(z[inds1] + Z)**2)) / z[inds1]))
    
    return P

#--------------------------------------------------------------------------------------------------

def glasssphere_profile(r,
                        R    = 18.5,
                        FWHM = 5,
                        d    = 1.5,
                        S    = 10.0,
                        B    = 1.0):
    """ Radial profile of a hot sphere with cold glass wall in warm background

    Parameters
    ----------
    r : 1D numpy float array 
      array with radial coordinates

    R : float, optional
      the radius of the sphere

    FWHM : float, optional
      the full width at half maximum of the points spread function

    d : float, optional
      the thickness (diameter) of the cold glass wall

    S : float, optional
      the signal in the sphere

    B : float, optional
      the signal in the background

    Returns
    -------
    1D numpy float array
    """
    sqrt2 = np.sqrt(2)

    sigma = FWHM / (2*np.sqrt(2*np.log(2)))  
    Z     = R / (sigma*sqrt2)
    w     = d / (sigma*sqrt2)
    z     = r / (sigma*sqrt2)

    P = S*gausssphere_profile(z, Z) - B*gausssphere_profile(z, Z + w) + B

    return P

#--------------------------------------------------------------------------------------------------

def plot1dprofiles(vol,
                   voxsizes):
    """ Plot profiles along the x, y and z axis through the center of mass of a sphere

    Parameters
    ----------
    vol : 3d numpy array 
      containing the volume

    voxsizes :  3 component array 
      with the voxel sizes
    """ 
    # now we have to find the activity weighted center of gravity of the sphere
    # to do so we do a coarse delineation of the sphere (30% over bg)
    bg    = np.mean(vol[:,:,0])
    absth = relth*(vol.max() - bg) + bg

    mask              = np.zeros_like(vol, dtype = np.uint8)
    mask[vol > absth] = 1

    i0, i1, i2 = np.indices(vol.shape)
    i0         = i0*voxsizes[0]
    i1         = i1*voxsizes[1]
    i2         = i2*voxsizes[2]

    # calculate the maxmimum radius of the subvolumes
    # all voxels with a distance bigger than rmax will not be included in the fit
    rmax = np.min((i0.max(),i1.max(),i2.max()))/2

    # first try to get the center of mass via the coarse delineation 
    weights       = vol[mask == 1]
    summedweights = np.sum(weights)
 
    c0 = np.sum(i0[mask == 1]*weights) / summedweights  
    c1 = np.sum(i1[mask == 1]*weights) / summedweights  
    c2 = np.sum(i2[mask == 1]*weights) / summedweights  

    r  = np.sqrt((i0 - c0)**2 + (i1 - c1)**2 + (i2 - c2)**2)

    # second try to get the center of mass
    # use weights from a smoothed volume 
    sigmas = 4 / (2.355*voxsizes)
    vol_sm = gaussian_filter(vol, sigma = sigmas)

    weights       = vol_sm[r <= rmax]
    summedweights = np.sum(weights)

    d0 = np.sum(i0[r <= rmax]*weights) / summedweights  
    d1 = np.sum(i1[r <= rmax]*weights) / summedweights  
    d2 = np.sum(i2[r <= rmax]*weights) / summedweights  

    r  = np.sqrt((i0 - d0)**2 + (i1 - d1)**2 + (i2 - d2)**2)

    if plot1dprofiles:
        spherecenter = np.unravel_index(np.argmin(r),r.shape)
        prof0 = vol[:, spherecenter[1], spherecenter[2]]
        prof1 = vol[spherecenter[0], :, spherecenter[2]]
        prof2 = vol[spherecenter[0], spherecenter[1], :]

        dims = vol.shape

        prof02 = vol.sum(axis = (1,2)) / (dims[1]*dims[2])
        prof12 = vol.sum(axis = (0,2)) / (dims[0]*dims[2])
        prof22 = vol.sum(axis = (0,1)) / (dims[0]*dims[1])

        c0 = sum(prof0*voxsizes[0]*np.arange(len(prof0))) / sum(prof0)
        c1 = sum(prof1*voxsizes[1]*np.arange(len(prof1))) / sum(prof1)
        c2 = sum(prof2*voxsizes[2]*np.arange(len(prof2))) / sum(prof2)

        #bg = 0.5*(prof2[0:3].mean() + prof2[-3:].mean())

        #fwhm0 = np.argwhere(prof0 - bg > 0.5*(prof0.max() - bg))[:,0].ptp()*voxsizes[0]
        #fwhm1 = np.argwhere(prof1 - bg > 0.5*(prof1.max() - bg))[:,0].ptp()*voxsizes[1]
        #fwhm2 = np.argwhere(prof2 - bg > 0.5*(prof2.max() - bg))[:,0].ptp()*voxsizes[2]

        fig1d, ax1d = py.subplots(1)
        ax1d.plot(voxsizes[0]*np.arange(len(prof0)) - c0, prof0, label = 'x')
        ax1d.plot(voxsizes[1]*np.arange(len(prof1)) - c1, prof1, label = 'y')
        ax1d.plot(voxsizes[2]*np.arange(len(prof2)) - c2, prof2, label = 'z')
        py.legend()

#--------------------------------------------------------------------------------------------------

def get_sphere_center(vol, 
                      voxsizes,
                      relth = 0.25):
    """ Get the center of gravity of a single hot sphere in a volume
    
    Parameters
    ----------
    vol : 3d numpy array 
      containing the volume

    voxsizes : 3 component array 
      with the voxel sizes

    relth : float, optional
      the relative threshold (signal over background) for the first coarse 
      delination of the sphere - default 0.25
    """
    # now we have to find the activity weighted center of gravity of the sphere
    # to do so we do a coarse delineation of the sphere (30% over bg)
    bg    = np.mean(vol[:,:,0])
    absth = relth*(vol.max() - bg) + bg

    mask              = np.zeros_like(vol, dtype = np.uint8)
    mask[vol > absth] = 1

    i0, i1, i2 = np.indices(vol.shape)
    i0         = i0*voxsizes[0]
    i1         = i1*voxsizes[1]
    i2         = i2*voxsizes[2]

    # calculate the maxmimum radius of the subvolumes
    # all voxels with a distance bigger than rmax will not be included in the fit
    rmax = np.min((i0.max(),i1.max(),i2.max()))/2

    # first try to get the center of mass via the coarse delineation 
    weights       = vol[mask == 1]
    summedweights = np.sum(weights)
 
    c0 = np.sum(i0[mask == 1]*weights) / summedweights  
    c1 = np.sum(i1[mask == 1]*weights) / summedweights  
    c2 = np.sum(i2[mask == 1]*weights) / summedweights  

    r  = np.sqrt((i0 - c0)**2 + (i1 - c1)**2 + (i2 - c2)**2)

    # second try to get the center of mass
    # use weights from a smoothed volume 
    sigmas = 4 / (2.355*voxsizes)
    vol_sm = gaussian_filter(vol, sigma = sigmas)

    weights       = vol_sm[r <= rmax]
    summedweights = np.sum(weights)

    d0 = np.sum(i0[r <= rmax]*weights) / summedweights  
    d1 = np.sum(i1[r <= rmax]*weights) / summedweights  
    d2 = np.sum(i2[r <= rmax]*weights) / summedweights  

    sphere_center = np.array([d0, d1, d2])

    return sphere_center

#--------------------------------------------------------------------------------------------------

def fitspheresubvolume(vol,
                       voxsizes,
                       relth          = 0.25,
                       Rfix           = None,
                       FWHMfix        = None,
                       dfix           = None,
                       Sfix           = None,
                       Bfix           = None,
                       wm             = 'dist',
                       cl             = False,
                       sphere_center  = None):
    """Fit the radial sphere profile of a 3d volume containg 1 sphere

    Parameters
    ----------
    vol : 3d numpy array 
      containing the volume

    voxsizes : 3 component array 
      with the voxel sizes

    relth : float, optional
      the relative threshold (signal over background) for the first coarse 
      delination of the sphere

    dfix, Sfix, Bfix, Rfix : float, optional 
      fixed values for the wall thickness, signal, background and radius

    wm : string, optinal   
      the weighting method of the data (equal, dist, sqdist)

    cl : bool, optional
      bool whether to compute the confidence limits (this takes very long)
 
    sphere_center : 3 element np.array 
      containing the center of the spheres in mm
      this is the center of in voxel coordiantes multiplied by the voxel sizes

    Returns
    -------
    Dictionary
      with the fitresults (as returned by lmfit)
    """                   

    if sphere_center is None: sphere_center = get_sphere_center(vol, voxsizes, relth = relth)
      
    i0, i1, i2 = np.indices(vol.shape)
    i0         = i0*voxsizes[0]
    i1         = i1*voxsizes[1]
    i2         = i2*voxsizes[2]

    rmax = np.min((i0.max(),i1.max(),i2.max()))/2
    r  = np.sqrt((i0 - sphere_center[0])**2 + (i1 - sphere_center[1])**2 + (i2 - sphere_center[2])**2)

    data = vol[r <= rmax].flatten()
    rfit = r[r <= rmax].flatten()

    if (Rfix == None): Rinit = 0.5*rmax
    else: Rinit = Rfix
 
    if (FWHMfix == None): FWHMinit = 2*voxsizes[0]  
    else: FWHMinit = FWHMfix
 
    if (dfix == None): dinit = 0.15  
    else: dinit = dfix

    if (Sfix == None): Sinit = data.max() 
    else: Sinit = Sfix  
 
    if (Bfix == None): Binit = data.min()
    else: Binit = Bfix 

    # lets do the actual fit
    pmodel = Model(glasssphere_profile)
    params = pmodel.make_params(R = Rinit, FWHM = FWHMinit, d = dinit, S = Sinit, B = Binit)

    # fix the parameters that should be fixed
    if Rfix    != None: params['R'].vary    = False
    if FWHMfix != None: params['FWHM'].vary = False
    if dfix    != None: params['d'].vary    = False
    if Sfix    != None: params['S'].vary    = False
    if Bfix    != None: params['B'].vary    = False

    params['R'].min    = 0
    params['FWHM'].min = 0
    params['d'].min    = 0
    params['S'].min    = 0
    params['B'].min    = 0

    if   wm == 'equal' : weigths = np.ones_like(rfit)
    elif wm == 'sqdist': weights = 1.0/(rfit**2) 
    else               : weights = 1.0/rfit 

    weights[weights == np.inf] = 0

    fitres = pmodel.fit(data, r = rfit, params = params, weights = weights)
    fitres.rdata = rfit
    if cl: fitres.cls   = fitres.conf_interval()

    # calculate the a50 mean
    fitres.a50th     = fitres.values['B'] + 0.5*(vol.max() - fitres.values['B'])
    fitres.mean_a50  = np.mean(data[data >= fitres.a50th])

    # calculate the mean
    fitres.mean = np.mean(data[rfit <= fitres.values['R']]) 

    # calculate the max
    fitres.max = data.max()

    # add the sphere center to the fit results
    fitres.sphere_center = sphere_center

    return fitres

#--------------------------------------------------------------------------------------------------

def plotspherefit(fitres, ax = None, xlim = None, ylim = None, unit = 'mm', showres = True):
    """Plot the results of a single sphere fit 

    Parameters
    ----------
    fitres : dictionary
      the results of the fit as returned by fitspheresubvolume

    ax : matplotlib axis, optional
      to be used for the plot

    xlim, ylim : float, optional
      the x/y limit

    unit : str, optional
      the unit of the radial coordinate

    showres : bool, optional
      whether to add text about the fit results in the plot
    """                   

    rplot = np.linspace(0, fitres.rdata.max(),100)
    
    if ax   == None: fig, ax = py.subplots(1)
    if xlim == None: xlim = (0,rplot.max())

    ax.plot(fitres.rdata, fitres.data, 'k.', ms = 2.5)
    
    ax.add_patch(patches.Rectangle((0, 0), fitres.values['R'], fitres.values['S'], 
                  facecolor = 'lightgrey', edgecolor = 'None'))
    x2  = fitres.values['R'] + fitres.values['d']
    dx2 = xlim[1] - x2 
    ax.add_patch(patches.Rectangle((x2, 0), dx2, fitres.values['B'], 
                  facecolor = 'lightgrey', edgecolor = 'None'))

    ax.plot(rplot, fitres.eval(r = rplot), 'r-')
    ax.set_xlabel('R (' + unit + ')')
    ax.set_ylabel('signal')

    ax.set_xlim(xlim)
    if ylim != None: ax.set_ylim(ylim)

    if showres:
        ax.text(0.99, 0.99, fitres.fit_report(), fontsize = 6, transform = ax.transAxes, 
                            verticalalignment='top', horizontalalignment = 'right')

#--------------------------------------------------------------------------------------------------

def NEMASubvols(input_vol,    
                voxsizes,     
                relTh    = 0.2,
                minvol   = 300,
                margin   = 9,  
                nbins    = 100,
                zignore  = 38,
                bgSignal = None):
    """ Segment a complete NEMA PET volume with several hot sphere in different subvolumes containing
        only one sphere

    Parameters
    ----------
    input_vol : 3D numpy array
      the volume to be segmented

    voxsizes  a 1D numpy array 
      containing the voxelsizes

    relTh : float, optional
      the relative threshold used to find spheres

    minvol : float, optional
      minimum volume of spheres to be segmented (same unit as voxel size^3)

    margin : int, optional
      margin around segmented spheres (same unit as voxel size)

    nbins : int, optional
      number of bins used in histogram for background determination

    zignore : float, optional
     distance to edge of FOV that is ignored (same unit as voxelsize) 

    bgSignal : float or None, optional
      the signal intensity of the background
      if None, it is auto determined from a histogram analysis

    Returns
    -------
    list
      of slices to access the subvolumes from the original volume
    """ 

    vol = input_vol.copy()

    xdim, ydim, zdim = vol.shape
    
    minvol = int(minvol / np.prod(voxsizes))
    
    dx = int(np.ceil(margin / voxsizes[0]))
    dy = int(np.ceil(margin / voxsizes[1]))
    dz = int(np.ceil(margin / voxsizes[2]))

    nzignore = int(np.ceil(zignore / voxsizes[2]))
    vol[:,:,:nzignore]  = 0   
    vol[:,:,-nzignore:] = 0   
 
    # first do a quick search for the biggest sphere (noisy edge of FOV can spoil max value!)
    histo = py.histogram(vol[vol > 0.01*vol.max()], nbins) 
    #bgSignal = histo[1][argrelextrema(histo[0], np.greater)[0][0]]
    if bgSignal is None:
      bgSignal = histo[1][find_peaks_cwt(histo[0], np.arange(nbins/6,nbins))[0]]
    thresh   = bgSignal + relTh*(vol.max() - bgSignal)
    
    vol2               = np.zeros(vol.shape, dtype = np.int)
    vol2[vol > thresh] = 1
    
    vol3, nrois = label(vol2)
    rois = np.arange(1, nrois + 1)
    roivols = labeled_comprehension(vol, vol3, rois, len, int, 0)
   
    i = 1
    
    for roi in rois: 
        if(roivols[roi-1] < minvol): vol3[vol3 == roi] = 0
        else:
            vol3[vol3 == roi] = i
            i = i + 1
    
    nspheres     = vol3.max()
    spherelabels = np.arange(1, nspheres + 1)
    
    bboxes = find_objects(vol3)
    
    nmaskvox = list()
    slices   = list()    
 
    for bbox in bboxes:
        xstart = max(0, bbox[0].start - dx)
        xstop  = min(xdim, bbox[0].stop + dx + 1)
    
        ystart = max(0, bbox[1].start - dy)
        ystop  = min(xdim, bbox[1].stop + dy + 1)
    
        zstart = max(0, bbox[2].start - dz)
        zstop  = min(xdim, bbox[2].stop + dz + 1)
    
        slices.append([slice(xstart,xstop,None), slice(ystart,ystop,None), slice(zstart,zstop,None)])

        nmaskvox.append((xstop-xstart)*(ystop-ystart)*(zstop-zstart))

    # sort subvols acc to number of voxel
    slices   = [ slices[kk] for kk in np.argsort(nmaskvox)[::-1] ]

    return slices

#--------------------------------------------------------------------------------------------------

def findNEMAROIs(vol, voxsizes, R = None, relth = 0.25, bgth = 0.5):
  """
  image-based ROI definition in a NEMA IQ scan 

  Arguments
  ---------

  vol      ... the volume to be segmented

  voxsizes ... a numpy array of voxelsizes in mm


  Keyword arguments
  ----------------
  
  R        ... (np.array or list) with the sphere radii in mm
               default (None) means [18.5,14.,11.,8.5,6.5,5.]       

  relth    ... (float) relative threshold to find the spheres
               above background (default 0.25)

  bgth     ... (float) relative threshold to find homogenous
               background around spheres

  Returns
  -------

  A volume containing labels for the ROIs:
  1      ... background
  2 - 7  ... NEMA spheres (large to small)
  """

  if R is None: R = np.array([18.5,14.,11.,8.5,6.5,5.])

  slices   = NEMASubvols(vol, voxsizes)
  labelvol = np.zeros(vol.shape, dtype = np.uint8) 
  bgmask   = np.zeros(vol.shape, dtype = np.uint8) 
  
  for i in range(len(slices)):
    subvol        = vol[slices[i]]
    sphere_center = get_sphere_center(subvol, voxsizes, relth = relth)
  
    i0, i1, i2 = np.indices(subvol.shape)
    i0         = i0*voxsizes[0]
    i1         = i1*voxsizes[1]
    i2         = i2*voxsizes[2]
  
    r  = np.sqrt((i0 - sphere_center[0])**2 + (i1 - sphere_center[1])**2 + (i2 - sphere_center[2])**2)
  
    if i == 0: bg = subvol[r > (R[i] + 7)].mean()
  
    submask = np.zeros(subvol.shape, dtype = np.uint8)
    submask[r <= R[i]] = i + 2
  
    labelvol[slices[i]] = submask
  
  # calculate the background
  bgmask[vol > bgth*bg]  = 1
  bgmask[labelvol >= 2] = 0
  
  bgmask_eroded = binary_erosion(bgmask, iterations = int(15/voxsizes[0]))
  labelvol[bgmask_eroded] = 1

  return(labelvol)

#--------------------------------------------------------------------------------------------------

def fit_WB_NEMA_sphere_profiles(vol,        
                                voxsizes,
                                sm_fwhm   = 0,
                                margin    = 9.0,
                                dfix      = 1.5,
                                Sfix      = None,
                                Rfix      = None,
                                FWHMfix   = None,
                                wm           = 'dist',
                                nmax_spheres = 6,
                                sameSignal   = False):
    """ Analyse the sphere profiles of a NEMA scan

    Parameters
    ----------
    vol : 3D numpy array
      the volume to be segmented

    voxsizes :  a 3 element numpy array 
      of voxelsizes in mm

    sm_fwhm : float, optional
      FWHM of the gaussian used for post-smoothing (mm)

    dfix, Sfix : float, optional
      fixed values for the wall thickness, and signal

    Rfix : 1D numpy array, optional
      a 6 component array with fixed values for the sphere radii (mm)

    margin: float, optional
      margin around segmented spheres (same unit as voxel size)

    wm : str, optional
      the weighting method of the data (equal, dist, sqdist)

    nmax_spheres: int (optional)
      maximum number of spheres to consider (default 6)

    sameSignal : bool, optional
       whether to forace all spheres to have the signal from the biggest sphere

    Returns
    -------
    Dictionary
        containing the fitresults
    """ 

    unit = 'mm'   

    if sm_fwhm > 0:
        print('\nPost-smoothing with ' + str(sm_fwhm) + ' mm')
        sigma    = sm_fwhm / 2.355
        sigmas   = sigma / voxsizes
        vol      = gaussian_filter(vol, sigma = sigmas)

    # find the 2D background ROI
    bg_inds = find_background_roi(vol, voxsizes)
    bg_mean = vol[bg_inds].mean()
    bg_cov  = vol[bg_inds].std() / bg_mean

    slices  = NEMASubvols(vol, voxsizes, margin = margin, bgSignal = bg_mean)

    subvols = list()
    for iss, ss in enumerate(slices): 
      if iss < nmax_spheres:
        subvols.append(vol[ss])
     
    if Rfix == None: Rfix = [None] * len(subvols)
    if len(Rfix) < len(subvols): Rfix = Rfix + [None] * (len(subvols) - len(Rfix))

    # initial fit to get signal in the biggest sphere
    if (Sfix == None) and (sameSignal == True):
        initfitres = fitspheresubvolume(subvols[0], voxsizes, dfix = dfix, Bfix = bg_mean, 
                                        FWHMfix = FWHMfix, Rfix = Rfix[0], wm = wm)
        Sfix = initfitres.params['S'].value

    # fit of all spheres
    fitres = []
    for i in range(len(subvols)):
      fitres.append(fitspheresubvolume(subvols[i], voxsizes, dfix = dfix, Sfix = Sfix, 
                                       Bfix = bg_mean, FWHMfix = FWHMfix, Rfix = Rfix[i]))

    # summary of results
    fwhms = np.array([x.values['FWHM'] for x in fitres])
    Rs    = np.array([x.values['R'] for x in fitres])
    Bs    = np.array([x.values['B'] for x in fitres])
    Ss    = np.array([x.values['S'] for x in fitres])
   
    sphere_mean_a50  = np.array([x.mean_a50  for x in fitres])  
    sphere_mean      = np.array([x.mean      for x in fitres])  
    sphere_max       = np.array([x.max       for x in fitres])  

    sphere_results = pd.DataFrame({'R':Rs,'FWHM':fwhms,'signal':Ss, 'mean_a50':sphere_mean_a50,
                                   'mean':sphere_mean, 'max':sphere_max, 
                                   'background_mean': bg_mean, 'background_cov': bg_cov}, 
                                    index = np.arange(1,len(subvols)+1))

    return fitres, sphere_results

#-------------------------------------------------------------------------------------------
def show_WB_NEMA_profiles(fitres):

  nsph = len(fitres)

  rmax = fitres[0].rdata.max()
  fig, axes = py.subplots(2,3, figsize = (18,8.3))

  ymax = 1.05*max([x.max for x in fitres])

  for i in range(nsph):
    plotspherefit(fitres[i], ylim = (0, ymax), 
                  ax = axes[np.unravel_index(i,axes.shape)], xlim = (0,1.5*rmax))
  if nsph < 6:
    for i in np.arange(nsph,6):
        ax = axes[np.unravel_index(i,axes.shape)]
        ax.set_axis_off()

  fig.tight_layout()
  fig.show()

  return fig

#-------------------------------------------------------------------------------------------
def show_WB_NEMA_recoveries(sphere_results, true_activity, earlcolor = 'lightgreen'):

  unit = 'mm'   

  a50RCs = sphere_results['mean_a50'].values / true_activity
  maxRCs = sphere_results['max'].values / true_activity
  Rs     = sphere_results['R'].values

  fig2, axes2 = py.subplots(1,2, figsize = (6,4.), sharex = True)
  RCa50_min  = np.array([0.76, 0.72, 0.63, 0.57, 0.44, 0.27]) 
  RCa50_max  = np.array([0.89, 0.85, 0.78, 0.73, 0.60, 0.43]) 
  RCmax_min  = np.array([0.95, 0.91, 0.83, 0.73, 0.59, 0.34]) 
  RCmax_max  = np.array([1.16, 1.13, 1.09, 1.01, 0.85, 0.57]) 

  # add the EARL limits
  if earlcolor != None:
      for i in range(len(Rs)): 
          axes2[0].add_patch(patches.Rectangle((Rs[i] - 0.5, RCa50_min[i]), 1, 
                             RCa50_max[i] - RCa50_min[i], 
                             facecolor = earlcolor, edgecolor = 'None'))


  axes2[0].plot(Rs, a50RCs, 'ko')
  axes2[0].set_ylim(min(0.25,0.95*a50RCs.min()), max(1.02,1.05*a50RCs.max()))
  axes2[0].set_xlabel('R (' + unit + ')')
  axes2[0].set_ylabel('RC a50')

  # add the EARL limits
  if earlcolor != None:
      for i in range(len(Rs)): 
          axes2[1].add_patch(patches.Rectangle((Rs[i] - 0.5, RCmax_min[i]), 1, 
                             RCmax_max[i] - RCmax_min[i], 
                             facecolor = earlcolor, edgecolor = 'None'))

  axes2[1].plot(Rs, maxRCs, 'ko')
  axes2[1].set_ylim(min(0.29,0.95*maxRCs.min()), max(1.18,1.05*maxRCs.max()))
  axes2[1].set_xlabel('R (' + unit + ')')
  axes2[1].set_ylabel('RC max')

  fig2.tight_layout()
  fig2.show()

  return fig2
