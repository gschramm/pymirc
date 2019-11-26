import numpy as np
from   scipy.ndimage import map_coordinates

def resample_img_cont(img, cntcols, cntrows, nroutrows,
                           stepsize = 1., outside_above = True,
                           interp_contour = False, resamplestruct = None):
  """ Resample an image along a contour.

  The input is an image and a contour. The output is a resampled
  version of the input image, in which the original contour becomes
  a horizontal line in the center.
  The main aim is to generate an input for NImincostpath, which
  computes a minimum cost path contour from left to right in the
  resampled image.
  The combination of NIresample_img_cont, NImincostpath and
  NIresample_cont_cont can be used to refine the input contour.
  
  Parameters
  ----------
  img : 2d numpy array
    image which must be resampled
  
  cntcols : 1d numpy array
    column coordinates of the input contour
  
  cntrows : 1d array 
    row coordinates of the input contour
  
  nroutrows : int
    the number of IDL 'rows' of the output image (in python the 1 dimention), 
    which is a resampled version of the input image. The resampled image will
    be a strip along the input contour with a width of NROUTROWS *
    STEPSIZE, so NROUTROWS must be increased if you choose a smaller
    STEPSIZE.
 
  stepsize : float, optional
    specifying how fine the input image must be sampled
    perpendicular to the original contour. A value 0.5 means that
    the vertical pixel size in the resampled image corresponds to
    half the pixel size in the original image.
  
  interp_countour : bool, optional
    !!! not implemented yet !!!
    scalar, specifying how fine the input contour must be
    sampled. If set to zero or not supplied, the input contour is
    not resampled. If set to a non-zero value, the input image is
    sampled along the contour with steps of INTERP_CONTOUR. A
    smaller value for INTERP_CONTOUR will produce an output image
    with more columns.
  
  resamplestruct : dict, optional
    If set to zero or non-existing, it receives a structure which
    contains all parameters required for resampling. This can be
    used to resample another image OF THE SAME SIZE in exactly the
    same way.
    If it is a structure, all inputs except IMG are ignored, IMG is
    resampled as prescribed by RESAMPLESTRUCT.
    Can also be used in a call to resample_cont_cont.
  
  outside_above : bool, optional
    If set, then it is assumed that the contour is closed, and the
    inside of the contour will be at the bottom in the resampled
    image. If not set, the resampling direction depends on the order
    of the coordinates. For a closed contour in clockwise direction,
    the inside of the image will be at the bottom. For the same
    contour in counter clockwise direction, it will be at the top.
  
  Returns
  -------
  2d numpy array
    an image of NROUTROWS with dimension (cntcols.shape[0], nroutrows) 
  
  Note
  ----
  only for 2D images. When RESAMPLESTRUCT is supplied, the input
  image must have the same size as that in the call that returned
  RESAMPLESTRUCT.
  
  Python reimplementation J. Nuyts' NIresample_img_cont
  """

  if resamplestruct is not None:
    ncols     = resamplestruct['ncols']
    nrows     = resamplestruct['nrows']
    nroutrows = resamplestruct['nroutrows']
    cencols   = resamplestruct['cencols']
    cenrows   = resamplestruct['cenrows']
    normcols  = resamplestruct['normcols']
    normrows  = resamplestruct['normrows']
    steps     = resamplestruct['steps']
    npoints   = cencols.shape[0]
  else:
    if interp_contour:
      #NIlin_contour, cencols, cenrows, cntcols, cntrows, interp_contour
      raise Exception('interp_contour not implemented yet')
    else:
      cencols = cntcols
      cenrows = cntrows
  
    nrows, ncols = img.shape
    offset = 1
    
    npoints   = cencols.shape[0]
    index     = np.arange(npoints)
    right     = np.clip(index + offset, None, npoints - 1)
    left      = np.clip(index - offset, 0, None)
    normcols  = cenrows[right] - cenrows[left]
    normrows  = -(cencols[right] - cencols[left])
    normval   = np.sqrt(normcols**2 + normrows**2)
    normcols /= normval
    normrows /= normval
  
    # This way, the inside of a cyclic contour is at the bottom in the
    # resampled image if the contour is given in clockwise direction.
    #----------------------------
    steps  = -np.arange(nroutrows, dtype = np.float) * stepsize
    steps -= steps.mean()
  
    if outside_above:
      testval = (normcols * (cencols - cencols.mean()) + normrows * (cenrows - cenrows.mean())).sum()
      if testval < 0: steps = np.flip(steps)
  
    resamplestruct = {'ncols'     : ncols,
                      'nrows'     : nrows,
                      'nroutrows' : nroutrows,
                      'cencols'   : cencols, 
                      'cenrows'   : cenrows, 
                      'normcols'  : normcols, 
                      'normrows'  : normrows, 
                      'stepsize'  : stepsize, 
                      'steps'     : steps}
  
  interpcols = np.zeros((npoints, nroutrows))
  interprows = np.zeros((npoints, nroutrows))
  
  for col in range(npoints):
    interpcols[col,:] = cencols[col] + steps * normcols[col]
    interprows[col,:] = cenrows[col] + steps * normrows[col]
  
  outimg = map_coordinates(img, [interprows, interpcols], order = 1, prefilter = False)

  return outimg, resamplestruct
