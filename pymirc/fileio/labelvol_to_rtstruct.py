import pymirc.image_operations as pymi
import numpy                   as np
import pydicom
import os
import warnings

from scipy.ndimage import label, find_objects

# ------------------------------------------------------------------------
def labelvol_to_rtstruct(roi_vol,
                         aff,
                         refdcm_file,
                         filename,
                         uid_base          = '1.2.826.0.1.3680043.9.7147.',
                         seriesDescription = 'test rois',
                         structureSetLabel = 'RTstruct',
                         structureSetName  = 'my rois',
                         roi_colors        = [['255','0','0'],  ['0',  '0','255'],['0',  '255','0'],
                                            ['255','0','255'],['255','255','0'],['0','255','255']],
                         tags_to_copy      = ['PatientName','PatientID','AccessionNumber','StudyID',
                                              'StudyDescription','StudyDate','StudyTime',
                                              'SeriesDate','SeriesTime']):

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
  
  refdcm_file : string
    a single reference dicom file
    from this file several dicom tags are copied (e.g. the FrameOfReferenceUID)

  filename : string
    name of the output rtstruct file

  uid_base : string, optional
    uid base used to generate some dicom UID

  seriesDescription : string, optional
    dicom series description for the rtstruct series

  structureSetLabel, structureSetName : string, optional
    Label and Name of the structSet
  
  roi_colors: list of lists containing 3 integer strings (0 - 255)
    used as ROI display colors

  tags_to_copy: list of strings
    extra dicom tags to copy from the refereced dicom file
  """

  roinumbers = np.unique(roi_vol)
  roinumbers = roinumbers[roinumbers > 0]
  nrois  = len(roinumbers)

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

  dfr = pydicom.Dataset()
  dfr.FrameOfReferenceUID = refdcm.FrameOfReferenceUID
  
  ds.ReferencedFrameOfReferenceSequence = pydicom.Sequence([dfr])
  
  #######################################################################
  #######################################################################
  # write the ReferencedFrameOfReferenceSequence
  
  tmp = pydicom.Dataset()
  tmp.ReferencedSOPClassUID    = refdcm.SOPClassUID
  tmp.ReferencedSOPInstanceUID = refdcm.SOPInstanceUID
  
  tmp2 = pydicom.Dataset()
  tmp2.SeriesInstanceUID    = refdcm.SeriesInstanceUID
  tmp2.ContourImageSequence = pydicom.Sequence([tmp])
  
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
  
  # loop over the ROIs
  for iroi in roinumbers:
    dssr = pydicom.Dataset()
    dssr.ROINumber      = iroi
    dssr.ROIName        = 'ROI-' + str(dssr.ROINumber)
    dssr.ROIDescription = 'ROI-' + str(dssr.ROINumber)
    dssr.ROIGenerationAlgorithm        = 'MANUAL'
    dssr.ReferencedFrameOfReferenceUID = dfr.FrameOfReferenceUID
    
    ds.StructureSetROISequence.append(dssr)
    
    #######################################################################
    #######################################################################
    # write ROIContourSequence containing the actual 2D polygon points of the ROI
    
    ds_contour = pydicom.Dataset()
    ds_contour.ReferencedSOPClassUID    = refdcm.SOPClassUID
    ds_contour.ReferencedSOPInstanceUID = refdcm.SOPInstanceUID
    
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
      bin_slice = bin_vol[:,:,sl]
  
      if bin_slice.max() > 0:
        contours = pymi.binary_2d_image_to_contours(bin_slice)
  
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


