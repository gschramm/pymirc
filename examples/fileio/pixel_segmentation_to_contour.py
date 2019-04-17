import numpy as np
from   copy import deepcopy

import matplotlib.pyplot as plt

plt.ion()

from matplotlib.patches import Polygon 

from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes

from shapely.geometry import Point, LinearRing

#TODO: zero pad image to avoid open contours at edges

bin_img = np.array([[0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,1,1,1,0],
                    [0,0,0,1,1,1,0,1,0,1,0],
                    [0,0,0,1,1,1,0,1,1,1,0],
                    [0,0,0,0,0,0,0,1,1,1,0],
                    [0,1,1,1,0,0,0,1,0,1,0],
                    [0,1,0,1,0,0,0,1,0,1,0],
                    [0,1,1,1,0,0,0,1,1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0]])

bin_img_filled = binary_fill_holes(bin_img)

bin_holes = bin_img_filled - bin_img

# get the outer contours of the filled image (no holes)
outer_contours = measure.find_contours(bin_img_filled == 1, 0.5, positive_orientation = 'high')
# get the inner contours (contours around holes)
inner_contours = measure.find_contours(bin_holes == 1, 0.5, positive_orientation = 'low')

contours = deepcopy(outer_contours)

# test in which outer contour a given inner contour lies
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
  tmp = LinearRing(outer_contours[parent_contour_number[j]])
  d   = tmp.project(Point(inner_contours[j][0,:]))
  p   = tmp.interpolate(d)
  closest_outer_points.append(list(p.coords)[0])

  # check if closet point is already in contour
  closest_outer_points_is_in_contour.append(np.any(np.all(np.isin(outer_contours[parent_contour_number[j]],closest_outer_points[j],True),axis=1)))

  if closest_outer_points_is_in_contour[j]:
    i_point = np.where(np.all(np.isin(outer_contours[parent_contour_number[j]],closest_outer_points[j]), 
                              axis = 1))[0][0]

    contours[parent_contour_number[j]] = np.concatenate([contours[parent_contour_number[j]][:(i_point+1),:],
                                                         inner_contours[j],
                                                         np.array(closest_outer_points[j]).reshape(1,2),  
                                                         contours[parent_contour_number[j]][i_point:,:]])

    print(closest_outer_points[j], outer_contours[parent_contour_number[j]][i_point,:])

#--------------------------------------------------------------------
# Display the image and plot all contours found
fig, ax = plt.subplots(1,2, figsize = (10,5))

cols = ['r','b','g','y']

ax[0].imshow(bin_img, interpolation='nearest', cmap=plt.cm.gray)
for n, contour in enumerate(contours):
  ax[0].plot(contour[:, 1], contour[:, 0], color = cols[n])

ax[1].imshow(bin_holes, interpolation='nearest', cmap=plt.cm.gray)
for n, contour in enumerate(inner_contours):
  ax[1].plot(contour[:, 1], contour[:, 0], color = cols[parent_contour_number[n]])
  ax[0].plot(closest_outer_points[n][1],closest_outer_points[n][0],'o', color = cols[parent_contour_number[n]])

ax[1].axis('image')

