import pymirc.image_operations as pymi
import numpy                   as np
import pydicom
import os
import warnings
import datetime

from scipy.ndimage import label, find_objects

from .read_dicom import DicomVolume

# ------------------------------------------------------------------------
def labelvol_to_rtstruct(roi_vol,
                         aff,
                         refdcm_file,
                         filename,
                         uid_base          = '1.2.826.0.1.3680043.9.7147.',
                         seriesDescription = 'test rois',
                         structureSetLabel = 'RTstruct',
                         structureSetName  = 'my rois',
                         connect_holes     = True,
                         roinames          = None,
                         roidescriptions   = None,
                         roigenerationalgs = None,
                         roi_colors        = [['255','0','0'],  ['0',  '0','255'],['0',  '255','0'],
                                            ['255','0','255'],['255','255','0'],['0','255','255']],
                         tags_to_copy      = ['PatientName','PatientID','AccessionNumber','StudyID',
                                              'StudyDescription','StudyDate','StudyTime',
                                              'SeriesDate','SeriesTime'],
                         tags_to_add       = None):

  """Convert a 3D array with integer ROI label to RTstruct

  Parameters
  ---------

  roi_vol : 3d numpy integer array 
    in LPS orientation containing the ROI labels
    0 is considered background
    1  ...          ROI-1
    n  ...          ROI-n


  aff : 2d 4x4 numpy array
    affine matrix that maps from voxel to (LPS) world coordinates
  
  refdcm_file : string or list
    A single reference dicom file or a list of reference files (multiple CT slices)
    From this file several dicom tags are copied (e.g. the FrameOfReferenceUID).
    In case a list of files is given, the dicom tags are copied from the first file,
    however all SOPInstanceUIDs are added to the ContourImageSequence (needed for some
    RT systems)

  filename : string
    name of the output rtstruct file

  uid_base : string, optional
    uid base used to generate some dicom UID

  seriesDescription : string, optional
    dicom series description for the rtstruct series

  structureSetLabel, structureSetName : string, optional
    Label and Name of the structSet

  connect_holes : bool, optional
    whether to connect inner holes to their outer parents contour - default: True
    this connection is needed to show holes correctly in MIM

  roinames, roidescriptions, roigenerationalgs : lists, optional
    containing strings for ROIName, ROIDescription and ROIGenerationAlgorithm
  
  roi_colors: list of lists containing 3 integer strings (0 - 255), optional
    used as ROI display colors

  tags_to_copy: list of strings, list optional
    extra dicom tags to copy from the refereced dicom file

  tags_to_add: dictionary
    with valid dicom tags to add in the header
  """

  roinumbers = np.unique(roi_vol)
  roinumbers = roinumbers[roinumbers > 0]
  nrois  = len(roinumbers)

  if isinstance(refdcm_file, list):
    refdcm = pydicom.read_file(refdcm_file[0]) 
  else:
    refdcm = pydicom.read_file(refdcm_file) 
 
  file_meta = pydicom.Dataset()
  
  file_meta.ImplementationClassUID     = uid_base + '1.1.1'
  file_meta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.481.3'
  file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid('1.2.840.10008.5.1.4.1.1.481.3.')
  
  ds = pydicom.FileDataset(filename, {}, file_meta = file_meta, preamble=b"\0" * 128)

  ds.Modality          = 'RTSTRUCT'
  ds.SeriesDescription = seriesDescription

  #--- copy dicom tags from reference dicom file
  for tag in tags_to_copy:
    if tag in refdcm:
      setattr(ds,tag, refdcm.data_element(tag).value)
    else:
      warnings.warn(tag + ' not in reference dicom file -> will not be written')

 
  ds.StudyInstanceUID  = refdcm.StudyInstanceUID
  ds.SeriesInstanceUID = pydicom.uid.generate_uid(uid_base)
  
  ds.SOPClassUID    = '1.2.840.10008.5.1.4.1.1.481.3'
  ds.SOPInstanceUID = pydicom.uid.generate_uid(uid_base)
  
  ds.StructureSetLabel = structureSetLabel
  ds.StructureSetName  = structureSetName
  ds.StructureSetTime  = datetime.datetime.now().time()
  ds.StructureSetDate  = datetime.datetime.now().date()

  dfr = pydicom.Dataset()
  dfr.FrameOfReferenceUID = refdcm.FrameOfReferenceUID
  
  ds.ReferencedFrameOfReferenceSequence = pydicom.Sequence([dfr])

  if tags_to_add is not None: 
    for tag, value in tags_to_add.items():
      setattr(ds, tag, value)
 
  #######################################################################
  #######################################################################
  # write the ReferencedFrameOfReferenceSequence
 
  contourImageSeq = pydicom.Sequence()

  if isinstance(refdcm_file, list):
    # in case we got all reference dicom files we add all SOPInstanceUIDs
    # otherwise some RT planning systems refuse to read the RTstructs

    # sort the reference dicom files according to slice position
    dcmVol = DicomVolume(refdcm_file)
    # we have to read the data to get dicom slices properly sorted
    # the actual returned image is not needed
    dummyvol = dcmVol.get_data()

    # calculate the slice offset between the image and ROI volume
    sl_offset = int(round((np.linalg.inv(dcmVol.affine) @ aff[:,-1])[2]))

    # find the bounding box in the last direction
    ob_sls  = find_objects(roi_vol > 0)
    z_start = min([x[2].start for x in ob_sls]) 
    z_end   = max([x[2].stop  for x in ob_sls]) 

    for i in np.arange(z_start,z_end):
      tmp = pydicom.Dataset()
      tmp.ReferencedSOPClassUID    = dcmVol.sorted_SOPClassUIDs[i + sl_offset]
      tmp.ReferencedSOPInstanceUID = dcmVol.sorted_SOPInstanceUIDs[i + sl_offset]
      contourImageSeq.append(tmp) 
  else: 
    dcmVol = None
    tmp = pydicom.Dataset()
    tmp.ReferencedSOPClassUID    = refdcm.SOPClassUID
    tmp.ReferencedSOPInstanceUID = refdcm.SOPInstanceUID
    contourImageSeq.append(tmp) 
  
  tmp2 = pydicom.Dataset()
  tmp2.SeriesInstanceUID    = refdcm.SeriesInstanceUID
  tmp2.ContourImageSequence = contourImageSeq
  
  tmp3 = pydicom.Dataset()
  tmp3.ReferencedSOPClassUID    = '1.2.840.10008.3.1.2.3.1' 
  # TODO SOP just copied from MIM rtstructs 
  tmp3.ReferencedSOPInstanceUID = refdcm.StudyInstanceUID
  tmp3.RTReferencedSeriesSequence = pydicom.Sequence([tmp2])
  
  tmp4 = pydicom.Dataset()
  tmp4.FrameOfReferenceUID = refdcm.FrameOfReferenceUID
  tmp4.RTReferencedStudySequence = pydicom.Sequence([tmp3])
  
  ds.ReferencedFrameOfReferenceSequence = pydicom.Sequence([tmp4])
  
  #######################################################################
  #######################################################################
  
  ds.StructureSetROISequence = pydicom.Sequence()
  ds.ROIContourSequence      = pydicom.Sequence()
 
  if roinames          is None: roinames          = ['ROI-' + str(x) for x in roinumbers]
  if roidescriptions   is None: roidescriptions   = ['ROI-' + str(x) for x in roinumbers]
  if roigenerationalgs is None: roigenerationalgs = len(roinumbers) * ['MANUAL']

  # loop over the ROIs
  for iroi, roinumber in enumerate(roinumbers):
    dssr = pydicom.Dataset()
    dssr.ROINumber      = roinumber
    dssr.ROIName        = roinames[iroi]
    dssr.ROIDescription = roidescriptions[iroi]
    dssr.ROIGenerationAlgorithm        = roigenerationalgs[iroi]
    dssr.ReferencedFrameOfReferenceUID = dfr.FrameOfReferenceUID
    
    ds.StructureSetROISequence.append(dssr)
    
    #######################################################################
    #######################################################################
    # write ROIContourSequence containing the actual 2D polygon points of the ROI
    
    # generate binary volume for the current ROI 
    bin_vol = (roi_vol == dssr.ROINumber).astype(int)
  
    # find the bounding box in the last direction
    ob_sls  = find_objects(bin_vol)
    z_start = min([x[2].start for x in ob_sls]) 
    z_end   = max([x[2].stop  for x in ob_sls]) 

    ds_roi_contour = pydicom.Dataset()
    ds_roi_contour.ROIDisplayColor     = roi_colors[iroi % len(roi_colors)]
    ds_roi_contour.ReferencedROINumber = dssr.ROINumber
    ds_roi_contour.ContourSequence     = pydicom.Sequence()
  
    # loop over the slices in the 2 direction to create 2D polygons 
    for sl in np.arange(z_start, z_end):  
      ds_contour = pydicom.Dataset()

      if dcmVol is None:
        ds_contour.ReferencedSOPInstanceUID = refdcm.SOPInstanceUID
        ds_contour.ReferencedSOPClassUID    = refdcm.SOPClassUID
      else:
        ds_contour.ReferencedSOPInstanceUID = dcmVol.sorted_SOPInstanceUIDs[sl + sl_offset]
        ds_contour.ReferencedSOPClassUID    = dcmVol.sorted_SOPClassUIDs[sl + sl_offset]

      bin_slice = bin_vol[:,:,sl]
  
      if bin_slice.max() > 0:
        contours = pymi.binary_2d_image_to_contours(bin_slice, connect_holes = connect_holes)
  
        for ic in range(len(contours)):
          npoints  = contours[ic].shape[0]
          
          contour = np.zeros((npoints,3))
          
          for ipoint in range(npoints):
            contour[ipoint,:] = (aff @ np.concatenate((contours[ic][ipoint,:],[sl,1])))[:-1]  
          
          dsci = pydicom.Dataset()
          dsci.ContourGeometricType     = 'CLOSED_PLANAR'
          dsci.NumberOfContourPoints    = contour.shape[0]
          dsci.ContourImageSequence     = pydicom.Sequence([ds_contour])
          dsci.ContourData              = contour.flatten().tolist()
  
          # ContourImageSequence contains 1 element per 2D contour
          ds_roi_contour.ContourSequence.append(dsci)
    
    # has to contain one element per ROI
    ds.ROIContourSequence.append(ds_roi_contour)
  
  #######################################################################
  #######################################################################
  
  pydicom.filewriter.write_file(os.path.join('.',filename), 
                                ds, write_like_original = False)


