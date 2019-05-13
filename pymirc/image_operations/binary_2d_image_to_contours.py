import numpy as np

from copy import deepcopy
from matplotlib.patches import Polygon 

from skimage import measure

from scipy.ndimage.morphology   import binary_fill_holes
from scipy.ndimage.measurements import find_objects
from scipy.ndimage              import label

#---------------------------------------------------------------------------
def cheese_to_contours(img, connect_holes = True):
  """Conversion of a 2d binary image "cheese" image into a set of contours

  Parameters
  ----------
  img : 2d binary numpy array
    containing the pixelized segmentation

  connect_holes : bool, optional
    whether to connect inner holes to their outer parents contour - default: True
    this connection is needed to show holes correctly in MIM

  Returns
  -------
  list
    of Nx2 numpy arrays containg the contours

  Notes
  -----
  A binary cheese image contains an object with holes, but nothing inside those holes.

  Finding the contours is performed via the marching squares algorithm from skimage.
  First, the contours of the image with filled holes are generated. 
  Second (optional), the contours of the holes are generated and connected to
  their parent contours.
  """
  # we have to 0 pad the image, otherwise the contours are not closed
  # at the outside
  bin_img = np.pad(img, 1, 'constant')
  
  bin_img_filled = binary_fill_holes(bin_img)
  
  bin_holes = bin_img_filled - bin_img
  
  # get the outer contours of the filled image (no holes)
  outer_contours = measure.find_contours(bin_img_filled == 1, 0.5, positive_orientation = 'high')
  # get the inner contours (contours around holes)
  inner_contours = measure.find_contours(bin_holes == 1, 0.5, positive_orientation = 'low')
  
  # test in which outer contour a given inner contour lies
  if connect_holes:
    contours = deepcopy(outer_contours)
    outer_polys = []
    for i in range(len(outer_contours)):
      outer_polys.append(Polygon(outer_contours[i], True))
    
    parent_contour_number = np.zeros(len(inner_contours), dtype = int)
    for j in range(len(inner_contours)):
      for i in range(len(outer_contours)):
        if outer_polys[i].contains_point(inner_contours[j][0,:]):
          parent_contour_number[j] = i
    
    # get the closest point on the outer contours for the start point
    # of the inner contour
    
    closest_outer_points = []
    closest_outer_points_is_in_contour = []
    for j in range(len(inner_contours)):
      oc  = deepcopy(outer_contours[parent_contour_number[j]])
    
      ip    = inner_contours[j][0,:]
      dist  = np.apply_along_axis(lambda x: (x[0] - ip[0])**2 + (x[1] - ip[1])**2, 1, oc)
      closest_outer_points.append(oc[np.argmin(dist),:])
    
      cop = deepcopy(closest_outer_points[j])
    
      # check if closet point is already in contour
      closest_outer_points_is_in_contour.append(np.any(np.all(np.isin(oc,cop,True),axis=1)))
    
      con = deepcopy(contours[parent_contour_number[j]])
    
      i_point = np.where(np.apply_along_axis(lambda x: np.array_equal(x,cop),1,con))[0][0]
     
      contours[parent_contour_number[j]] = np.concatenate([con[:(i_point+1),:],
                                                           inner_contours[j],
                                                           np.array(cop).reshape(1,2),  
                                                           con[i_point:,:]])
  else:
    contours = outer_contours + inner_contours
  
  # subtract 1 from the contour coordinates because we had to use 0 padding
  for i in range(len(contours)):
     contours[i] -= 1

  return contours

#---------------------------------------------------------------------------
def binary_2d_image_to_contours(img, connect_holes = True):
  """Conversion of a 2d binary image image into a set of contours

  Parameters
  ----------
  img : 2d binary numpy array
    containing the pixelized segmentation

  connect_holes : bool, optional
    whether to connect inner holes to their outer parents contour - default: True
    this connection is needed to show holes correctly in MIM

  Returns
  -------
  list
    of Nx2 numpy arrays containg the contours

  Notes
  -----
  Finding the contours is performed via the marching squares algorithm from skimage.
  Special care is taken of holes and object inside holes.
  """
  img_copy = img.copy()
  
  regions, nrois = label(img_copy)
  obj_slices     = find_objects(regions)
  
  contours = []
  
  while nrois > 0:
    # test if found objets only have one label
    for i in range(len(obj_slices)):
      obj = img_copy[obj_slices[i]]
      lab = label(obj)
      if lab[1] == 1:
        # if there is only one label, the object is a "cheese image"
        # where we can calculate the contours 
        tmp = cheese_to_contours(obj, connect_holes = connect_holes)[0]
        tmp[:,0] += obj_slices[i][0].start
        tmp[:,1] += obj_slices[i][1].start
       
        contours.append(tmp) 
        
        # delete the processed rois from the label array
        img_copy[regions == (i+1)] = 0
    
    regions, nrois_new = label(img_copy)
    obj_slices         = find_objects(regions)

    if nrois_new != nrois:
      nrois = nrois_new
    else:
      for i in range(nrois):
        obj = (regions == (i + 1)).astype(int)

        for tmp in cheese_to_contours(obj, connect_holes = connect_holes)[:1]:
          contours.append(tmp) 
          
        # delete the processed rois from the label array
        img_copy[regions == (i+1)] = 0

      regions, nrois = label(img_copy)
      obj_slices     = find_objects(regions)

  return contours
