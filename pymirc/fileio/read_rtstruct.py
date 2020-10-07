import pydicom
import warnings
import numpy as np
import pylab as py

from scipy.spatial      import ConvexHull
from matplotlib.patches import Polygon 

#---------------------------------------------------------------------
def contour_orientation(c):
  """ Orientation of a 2D closed Polygon

  Parameters
  ----------
  c : (n,2) numpy array
    containing the x,y coordinates of the contour points

  Returns
  -------
  bool
    meaning counter-clockwise and clockwise orientation

  References
  ----------
  https://en.wikipedia.org/wiki/Curve_orientation

  """
  cc = ConvexHull(c[:,:2])
  cc.vertices.sort()
  x = c[cc.vertices,0]  
  y = c[cc.vertices,1]  

  ic = 2
  d  = 0 

  while d == 0:
    d = (x[1] - x[0])*(y[ic] - y[0]) - (x[ic] - x[0])*(y[1] - y[0])
    ic = (ic + 1) % c.shape[0]

  ori = (d > 0)

  return ori

#---------------------------------------------------------------------

def read_rtstruct_contour_data(rtstruct_file,
                               roinames = None):
  """Read dicom RTSTRUCT contour data

  Parameters
  ----------
  rtstruct_file : str
    a dicom RTSTRUCT file

  roinames : list of strings
    ROINames to read - default None means read all ROIs

  Returns
  -------
  list of length n
    with the contour data of all n ROIs saved in the RTSTRUCT.
    Every element of the list is a dictionary wih several keys.
    The actual contour points are saved in the key 'contour_points'
    which itself is a list of (x,3) numpy arrays containg the coordinates
    of the 2D planar contours.

  Note
  ----
  The most important dicom fields for RTSTRUCT are:
  -FrameOfReferenceUID
  -ROIContourSequence  (1 element for every ROI)
    -ReferenceROINumber
    -ContourSequence   (1 element for every 2D contour in a given ROI)
      -ContourData
      -GeometruCType
  """
  ds = pydicom.read_file(rtstruct_file)
  
  # get the Frame of Reference UID
  FrameOfReferenceUID = [x.FrameOfReferenceUID for x in ds.ReferencedFrameOfReferenceSequence]
  
  ctrs = ds.ROIContourSequence
  
  contour_data = []
 
  allroinames = [x.ROIName if 'ROIName' in x else '' for x in ds.StructureSetROISequence]

  if roinames is None: roinames = allroinames.copy()

  for roiname in roinames:
    i = allroinames.index(roiname)

    if 'ContourSequence' in ctrs[i]:
      contour_seq = ctrs[i].ContourSequence
    else:
      warnings.warn(f"The ROI with name '{roiname}' appears to be empty.")
      contour_seq = []


    contour_points = []
    contour_orientations = []
  
    for cs in contour_seq:
      cp = np.array(cs.ContourData).reshape(-1,3)
      if cp.shape[0] >= 3:
        contour_points.append(cp)
        contour_orientations.append(contour_orientation(cp[:,:2]))

    if len(contour_points) > 0: 
      cd = {'contour_points':       contour_points, 
            'contour_orientations': contour_orientations, 
            'GeometricType':        cs.ContourGeometricType,
            'Number':               ctrs[i].ReferencedROINumber,
            'FrameOfReferenceUID':  FrameOfReferenceUID}
  
      for key in ['ROIName','ROIDescription','ROINumber','ReferencedFrameOfReferenceUID','ROIGenerationAlgorithm']:
        if key in ds.StructureSetROISequence[i]: cd[key] = getattr(ds.StructureSetROISequence[i], key)
  
      contour_data.append(cd)


  return contour_data

#----------------------------------------------------------------------------------------------------

def convert_contour_data_to_roi_indices(contour_data, 
                                        aff, 
                                        shape, 
                                        radius = None, 
                                        use_contour_orientation = True):
  """Convert RTSTRUCT 2D polygon contour data to 3D indices

  Parameters
  ----------
  contour_data : list
    of contour data as returned from read_rtstruct_contour_data()

  aff: 2d 4x4 numpy array
    affine matrix that maps from voxel to world coordinates
    of volume where ROIs should be applied

  shape : 3 element tuple
    shape of the of volume where ROIs should be applied

  radius : float, optional
    passed to matplotlib.patches.Polygon.contains_point()
  
  use_contour_orientation: bool
    whether to use the orientation of a contour (clockwise vs counter clockwise)
    to determine whether a contour defines a ROI or a holes "within" an ROI.
    This approach is used by some vendors to store "holes" in 2D slices of 3D segmentations.

  Returns
  -------
  list
    containing the voxel indices of all ROIs

  Note
  ----
  (1) matplotlib.patches.Polygon.contains_point() is used to determine whether
  a voxel is inside a 2D RTSTRUCT polygon. There is ambiguity for voxels that only
  lie partly inside the polygon.
   
  Example
  -------
  dcm = pymirc.fileio.DicomVolume('mydcm_dir/*.dcm')
  vol = dcm.get_data()  

  contour_data = pymirc.fileio.read_rtstruct_contour_data('my_rtstruct_file.dcm')
  roi_inds     = pymirc.fileio.convert_contour_data_to_roi_indices(contour_data, dcm.affine, vol.hape)

  print('ROI name.....:', [x['ROIName']   for x in contour_data])
  print('ROI number...:', [x['ROINumber'] for x in contour_data])
  print('ROI mean.....:', [vol[x].mean()  for x in roi_inds])
  """
  roi_inds = []
  
  for iroi in range(len(contour_data)):
    contour_points       = contour_data[iroi]['contour_points']
    contour_orientations = np.array(contour_data[iroi]['contour_orientations'])
   
    roi_number = int(contour_data[iroi]['Number'])
   
    roi_inds0 = []
    roi_inds1 = []
    roi_inds2 = []
 
    # calculate the slices of all contours
    sls = np.array([int(round((np.linalg.inv(aff) @ np.concatenate([x[0,:],[1]]))[2])) for x in contour_points])
    sls_uniq = np.unique(sls)

    for sl in sls_uniq:
      sl_inds =  np.where(sls == sl)[0]

      if np.any(np.logical_not(contour_orientations[sl_inds])):
        # case where we have negative contours (holes) in the slices
        bin_img = np.zeros(shape[:2], dtype = np.int16)
        for ip in sl_inds:
          cp = contour_points[ip] 
  
          # get the minimum and maximum voxel coordinate of the contour in the slice
          i_min = np.floor((np.linalg.inv(aff) @ np.concatenate([cp.min(axis=0),[1]]))[:2]).astype(int)
          i_max = np.ceil((np.linalg.inv(aff) @ np.concatenate([cp.max(axis=0),[1]]))[:2]).astype(int)
  
          n_test = i_max + 1 - i_min
  
          poly = Polygon(cp[:,:-1], True)
 
          contour_orientation = contour_orientations[ip] 
 
          for i in np.arange(i_min[0], i_min[0] + n_test[0]):
            for j in np.arange(i_min[1], i_min[1] + n_test[1]):
              if poly.contains_point((aff @ np.array([i,j,sl,1]))[:2], radius = radius):
                if use_contour_orientation:
                  if contour_orientation:
                    bin_img[i,j] += 1
                  else:
                    bin_img[i,j] -= 1
                else:
                  bin_img[i,j] += 1
        inds0, inds1 = np.where(bin_img > 0)
        inds2 = np.repeat(sl,len(inds0))

        roi_inds0 = roi_inds0  + inds0.tolist() 
        roi_inds1 = roi_inds1  + inds1.tolist()
        roi_inds2 = roi_inds2  + inds2.tolist()

      else:
        # case where we don't have negative contours (holes) in the slices
        for ip in sl_inds:
          cp = contour_points[ip] 
  
          # get the minimum and maximum voxel coordinate of the contour in the slice
          i_min = np.floor((np.linalg.inv(aff) @ np.concatenate([cp.min(axis=0),[1]]))[:2]).astype(int)
          i_max = np.ceil((np.linalg.inv(aff) @ np.concatenate([cp.max(axis=0),[1]]))[:2]).astype(int)
  
          n_test = i_max + 1 - i_min
  
          poly = Polygon(cp[:,:-1], True)
 
          for i in np.arange(i_min[0], i_min[0] + n_test[0]):
            for j in np.arange(i_min[1], i_min[1] + n_test[1]):
              if poly.contains_point((aff @ np.array([i,j,sl,1]))[:2], radius = radius):
                roi_inds0.append(i)
                roi_inds1.append(j)
                roi_inds2.append(sl)

    roi_inds.append((np.array(roi_inds0), np.array(roi_inds1), np.array(roi_inds2)))

  return roi_inds
