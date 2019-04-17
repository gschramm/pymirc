import pydicom
import numpy as np
import pylab as py

from matplotlib.patches import Polygon 
#import shapely.geometry as shg

def read_rtstruct_contour_data(rtstruct_file):
  """Read dicom RTSTRUCT contour data

  Parameters
  ----------
  restruct_file : str
    a dicom RTSTRUCT file

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
  
  for i in range(len(ctrs)):
    contour_seq    = ctrs[i].ContourSequence
    contour_points = []
  
    for cs in contour_seq:
      cp = np.array(cs.ContourData).reshape(-1,3)
      contour_points.append(cp)
  
    cd = {'contour_points':      contour_points, 
          'GeometricType':       cs.ContourGeometricType,
          'Number':              ctrs[i].ReferencedROINumber,
          'FrameOfReferenceUID': FrameOfReferenceUID}
  
    for key in ['ROIName','ROIDescription','ROINumber','ReferencedFrameOfReferenceUID','ROIGenerationAlgorithm']:
      if key in ds.StructureSetROISequence[i]: cd[key] = getattr(ds.StructureSetROISequence[i], key)
  
    contour_data.append(cd)

  return contour_data

#----------------------------------------------------------------------------------------------------

def convert_contour_data_to_roi_indices(contour_data, aff, shape, radius = None):
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
    contour_points = contour_data[iroi]['contour_points']
   
    roi_number = int(contour_data[iroi]['Number'])
  
    roi_inds0 = []
    roi_inds1 = []
    roi_inds2 = []
  
    for cp in contour_points:
      # get the slice of the current contour
      sl = int(round((np.linalg.inv(aff) @ np.concatenate([cp[0,:],[1]]))[2]))
  
      # get the minimum and maximum voxel coordinate of the contour in the slice
      i_min = np.floor((np.linalg.inv(aff) @ np.concatenate([cp.min(axis=0),[1]]))[:2]).astype(int)
      i_max = np.ceil((np.linalg.inv(aff) @ np.concatenate([cp.max(axis=0),[1]]))[:2]).astype(int)
  
      n_test = i_max + 1 - i_min
  
      poly = Polygon(cp[:,:-1], True)
      #poly = shg.Polygon(cp[:,:-1])
  
      for i in np.arange(i_min[0], i_min[0] + n_test[0]):
        for j in np.arange(i_min[1], i_min[1] + n_test[1]):
          if poly.contains_point((aff @ np.array([i,j,sl,1]))[:2], radius = radius):
          #if poly.contains(shg.Point((aff @ np.array([i,j,sl,1]))[:2])):
            roi_inds0.append(i)
            roi_inds1.append(j)
            roi_inds2.append(sl)
  
    roi_inds.append((np.array(roi_inds0), np.array(roi_inds1), np.array(roi_inds2)))

  return roi_inds
