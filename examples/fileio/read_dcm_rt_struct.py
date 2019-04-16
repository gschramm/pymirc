import pydicom
import numpy as np
import pylab as py

from matplotlib.patches import Polygon

#-------------------------------------------------------------------------------------
import sys, os
pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)
import pymirc.fileio as pymf

ct_dcm = pymf.DicomVolume('../../data/nema_petct/CT/*.dcm')
ct_vol = ct_dcm.get_data()

aff   = ct_dcm.affine
shape = ct_vol.shape 

#-------------------------------------------------------------------------------------
ds = pydicom.read_file('../../data/nema_petct/sphere_rt_struct/rt_struct.dcm')

ctrs = ds.ROIContourSequence

contour_data = []

for i in range(len(ctrs)):
  contour_seq    = ctrs[i].ContourSequence
  contour_points = []

  for cs in contour_seq:
    cp = np.array(cs.ContourData).reshape(-1,3)
    contour_points.append(cp)

  cd = {'contour_points': contour_points, 
        'GeometricType':  cs.ContourGeometricType,
        'Number':         ctrs[i].ReferencedROINumber}

  for key in ['ROIName','ROIDescription']:
    if key in ds.StructureSetROISequence[i]: cd[key] = getattr(ds.StructureSetROISequence[i], key)

  contour_data.append(cd)

#--------------------------------------------------------------------
# convert contour data to index array

roi_inds = []

for iroi in range(len(contour_data)):
  contour_points = contour_data[iroi]['contour_points']
 
  roi_number = int(contour_data[iroi]['Number'])

  roi_inds0 = []
  roi_inds1 = []
  roi_inds2 = []

  for cp in contour_points:
    # get the slice of the current contour
    sl = int(round((np.linalg.inv(ct_dcm.affine) @ np.concatenate([cp[0,:],[1]]))[2]))

    # get the minimum and maximum voxel coordinate of the contour in the slice
    i_min = np.floor((np.linalg.inv(ct_dcm.affine) @ np.concatenate([cp.min(axis=0),[1]]))[:2]).astype(int)
    i_max = np.ceil((np.linalg.inv(ct_dcm.affine) @ np.concatenate([cp.max(axis=0),[1]]))[:2]).astype(int)

    n_test = i_max + 1 - i_min

    poly = Polygon(cp[:,:-1], True)

    for i in np.arange(i_min[0], i_min[0] + n_test[0]):
      for j in np.arange(i_min[1], i_min[1] + n_test[1]):
        if poly.contains_point((aff @ np.array([i,j,sl,1]))[:2]):
          roi_inds0.append(i)
          roi_inds1.append(j)
          roi_inds2.append(sl)


  roi_inds.append((np.array(roi_inds0), np.array(roi_inds1), np.array(roi_inds2)))

#---------------------------------------------------------------------------
roi_vol = np.zeros(ct_vol.shape)

for i in range(len(roi_inds)):
  roi_vol[roi_inds[i]] = i + 1

#---------------------------------------------------------------------------
import pymirc.viewer as pymv
vi = pymv.ThreeAxisViewer([ct_vol,roi_vol], voxsize=ct_dcm.voxsize)


