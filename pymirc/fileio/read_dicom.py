import os
import numpy as np
import scipy.ndimage
import glob
import warnings

import pydicom as dicom

#--------------------------------------------------------------

class DicomVolume:
  """get 3D or 4D numpy arrays from a list of 2D dicom files

  Parameters
  ----------
  filelist : list or str
    either:
    (1) a list of 2d dicom files containing the image data of a 3D/4D dicom series
    (2) a string containing a pattern passed to glob.glob to generate the file list in (1)

   dicomlist: list of pydicom FileDatasets
     instead of specifing filelist, the list of pydicom FileDatasets can also be
     given directly. In this case filelist must not be given!

  fallback_series_type : 2 element tuple
    series type to use if not given in the header as tag SeriesType.
    Valid values for the 1st element are: "STATIC", "DYNAMIC", "GATED", "WHOLE BODY"
    Valid values for the 2nd element are: "IMAGE", "REPROJECTION"

  verbose: bool
    print verbose output

  Note
  ----
  The aim of this class is to get 3D/4D numpy arrays from a set of 2D dicom files of a dicom series
  in defined orientation (LPS).  

  Example
  -------
  dcm_vol = DicomVolume('mydicom_dir/*.dcm')
  img_arr = dcm_vol.get_data()
  img_aff = dcm_vol.affine
  dcm_hdr = dcm_vol.firstdcmheader
  """
  def __init__(self, filelist = None, dicomlist= None, fallback_series_type = ('STATIC','IMAGE'), verbose = True):
   
    self.verbose = verbose
 
    if   isinstance(filelist,list): self.filelist = filelist
    elif isinstance(filelist,str):  self.filelist = glob.glob(filelist)
    else:                           self.filelist = None

    self.dicomlist = dicomlist

    # throw error if neither filelist nor dicomlist are given
    if (self.filelist is None) and (self.dicomlist is None):
      raise InputError('Either filelist or dicomlist must be given as input') 

    # throw error if both filelist and dicomlist are given
    if (self.filelist is not None) and (self.dicomlist is not None):
      raise InputError('Either filelist or dicomlist must be given as input') 

    # attach the first dicom header to the object
    if self.filelist is not None:
      self.firstdcmheader = dicom.read_file(self.filelist[0])
    else: 
      self.firstdcmheader = self.dicomlist[0]

    # the extra check if the dimension of the pixel array is bigger than 2 is needed
    # because there are GE CT with erroneouly contain NumberOfFrames in classical slice by
    # slice dicom files
    if ('NumberOfFrames' in self.firstdcmheader) and (self.firstdcmheader.pixel_array.ndim > 2):
      # case of multi slice data (3d array in 1 dicom file
      # getting the ImageOrientationPatient attribute is not trivial
      # since it is stored in different tags by different vendors

      if 'DetectorInformationSequence' in self.firstdcmheader:     
        # this is for multi slice data of the Siemens symbia spect
        iop =  self.firstdcmheader.DetectorInformationSequence[0].ImageOrientationPatient
      else:
        try:
          # this is for multi slice data of molecubes
          iop = self.firstdcmheader.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
        except AttributeError:
          # this is for multi slice data from PMOD
          try:
            iop = self.firstdcmheader.PerFrameFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
          except AttributeError:
            # this is for multi slice data from RayStation
            iop = self.firstdcmheader.ImageOrientationPatient

      self.x = np.array(iop[:3], dtype = np.float) 
      self.y = np.array(iop[3:], dtype = np.float) 

      # set member variable that shows whether data has been read in
      self.read_all_dcms = True
      
    else:
      self.x = np.array(self.firstdcmheader.ImageOrientationPatient[0:3], dtype = np.float) 
      self.y = np.array(self.firstdcmheader.ImageOrientationPatient[3:] , dtype = np.float) 

      # set member variable that shows whether data has been read in
      self.read_all_dcms = False

    self.n = np.cross(self.x,self.y)

    # get the row and column pixelspacing 
    if 'PixelSpacing' in self.firstdcmheader:
      self.pixelspacing = np.array(self.firstdcmheader.PixelSpacing, dtype = np.float) 
    else:
      self.pixelspacing = np.array(self.firstdcmheader.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing)
     
    self.dr           = self.pixelspacing[0]
    self.dc           = self.pixelspacing[1]

    # approximately transform slices in patient coord. system
    self.normaxis  = np.argmax(np.abs(self.n))
    self.normdir   = np.sign(self.n[self.normaxis])
    self.rowaxis   = np.argmax(np.abs(self.x))
    self.rowdir    = np.sign(self.x[self.rowaxis])
    self.colaxis   = np.argmax(np.abs(self.y))
    self.coldir    = np.sign(self.y[self.colaxis])

    # read the number of frames (time slices)
    if 'NumberOfTimeSlices' in self.firstdcmheader: self.NumTimeSlices = self.firstdcmheader.NumberOfTimeSlices
    else: self.NumTimeSlices = 1


    # get the dicom series type to see whether we have a static or dynamic acq.
    if "SeriesType" in self.firstdcmheader:
      self.series_type = self.firstdcmheader.SeriesType
    else:
      self.series_type = dicom.multival.MultiValue(str, fallback_series_type)
      warnings.warn(f'Cannot find SeriesType in first dicom header. Setting it to {fallback_series_type}')

  #------------------------------------------------------------------------------------------------------
  def reorient_volume(self, patvol):
    """reorient the raw dicom volume to LPS orientation

    Parameters
    ----------
    patvol : 3d numpy array

    Returns
    -------
    3d numpy array
       reoriented numpy array in LPS orientation
    """

    # check the directions of the norm, col and row dir and revert some axis if necessary
    if(self.normdir == -1):
        patvol = patvol[::-1,:,:]
        self.offset = self.offset + (self.n0 - 1)*self.v0
        self.v0     = -1.0*self.v0
    if(self.coldir == -1):
        patvol = patvol[:,::-1,:]
        self.offset = self.offset + (self.n1 - 1)*self.v1
        self.v1     = -1.0*self.v1
    if(self.rowdir == -1):
        patvol = patvol[:,:,::-1]
        self.offset = self.offset + (self.n2 - 1)*self.v2
        self.v2     = -1.0*self.v2

    # now we want to make sure that the 0, 1, 2 axis of our 3d volume corrrespond
    # to the x, y, z axis in the patient coordinate system
    # therefore we might need to swap some axis
    if(self.normaxis == 0 and self.colaxis == 1 and self.rowaxis == 2):
        self.yvoxsize, self.zvoxsize = self.dr, self.dc
        self.xvoxsize                = self.sliceDistance
    elif(self.normaxis == 0 and self.colaxis == 2 and self.rowaxis == 1):
        if self.verbose: print('--- swapping axis 1 and 2')
        patvol                  = np.swapaxes(patvol,1,2) 
        self.v1, self.v2             = self.v2, self.v1
        self.zvoxsize, self.yvoxsize = self.dr, self.dc
        self.xvoxsize                = self.sliceDistance
    elif(self.normaxis == 1 and self.colaxis == 0 and self.rowaxis == 2):
        if self.verbose: print('--- swapping axis 0 and 1')
        patvol                  = np.swapaxes(patvol,0,1) 
        self.v0, self.v1             = self.v1, self.v0
        self.xvoxsize, self.zvoxsize = self.dr, self.dc
        self.yvoxsize                = self.sliceDistance
    elif(self.normaxis == 1 and self.colaxis == 2 and self.rowaxis == 0):
        if self.verbose: print('--- swapping axis 0 and 1')
        if self.verbose: print('--- swapping axis 0 and 2')
        patvol                  = np.swapaxes(np.swapaxes(patvol,0,1),0,2) 
        self.v0, self.v1             = self.v1, self.v0
        self.v0, self.v2             = self.v2, self.v0
        self.zvoxsize, self.xvoxsize = self.dr, self.dc
        self.yvoxsize                = self.sliceDistance
    elif(self.normaxis == 2 and self.colaxis == 1 and self.rowaxis == 0):
        if self.verbose: print('--- swapping axis 0 and 2')
        patvol                  = np.swapaxes(patvol,0,2) 
        self.v0, self.v2             = self.v2, self.v0
        self.yvoxsize, self.xvoxsize = self.dr, self.dc
        self.zvoxsize                = self.sliceDistance
    elif(self.normaxis == 2 and self.colaxis == 0 and self.rowaxis == 1):
        if self.verbose: print('--- swapping axis 0 and 2')
        if self.verbose: print('--- swapping axis 0 and 1')
        patvol                  = np.swapaxes(np.swapaxes(patvol,0,2),0,1) 
        self.v0, self.v2             = self.v2, self.v0
        self.v0, self.v1             = self.v1, self.v0
        self.xvoxsize, self.yvoxsize = self.dr, self.dc
        self.zvoxsize                = self.sliceDistance

    # update the volume dimensions 
    self.n0, self.n1, self.n2 = patvol.shape

    return patvol

  #------------------------------------------------------------------------------------------------------
  def get_data(self, frames = None):
    """get the actual 3D or 4D image data 

    Parameters
    ----------
    frames : list of ints, optional
      if the data is 4D this can be a list of frame number to be read
      the default None means read all frames

    Note
    ----
    This is a high level function that call the underlying function for
    reading 3D, 4D or multislice data sets.

    Returns
    -------
    a 3D or 4D numpy array
      array containing the data
    """
    if not self.read_all_dcms:
      if self.verbose: print('Analyzing dicom headers')

      if self.dicomlist is None:
        self.dicomlist = [dicom.read_file(x) for x in self.filelist] 

      # check if some images have a SOPclassUID that does not belong to images and drop them
      # SOPClassUID '1.2.840.10008.5.1.4.1.1.66' means Raw Data Storage
      #             '1.2.840.10008.5.1.4.1.1.66.x' for x in (1,2,3,4) are also not images
      self.dicomlist = [x for x in self.dicomlist if not x.SOPClassUID.startswith('1.2.840.10008.5.1.4.1.1.66')]

      self.read_all_dcms = True

      self.TemporalPositionIdentifiers = []

      # to figure out which 2d dicom file belongs to which time frame
      # we use the TemporalPositionIdentifier (not very common) or the acquisition date time
      for dcm in self.dicomlist:
        if (dcm.Modality == 'MR') and ('AcquisitionNumber' in dcm):
          self.TemporalPositionIdentifiers.append(dcm.AcquisitionNumber)
        elif (dcm.Modality == 'MR') and ('EchoNumbers' in dcm):
          self.TemporalPositionIdentifiers.append(dcm.EchoNumbers)
        elif 'TemporalPositionIdentifier' in dcm:
          self.TemporalPositionIdentifiers.append(dcm.TemporalPositionIdentifier)
        else:
          if 'AcquisitionDate' in dcm:
            acq_d = dcm.AcquisitionDate
          else:
            acq_d = '19700101'

          if 'AcquisitionTime' in dcm:
            acq_t = dcm.AcquisitionTime
          else:
            acq_t = '000000'

          # if the trigger time is in the data we add it to the acq. time
          # this is needed to read GE gated PET data
          if 'TriggerTime' in dcm:
            acq_t += ('.' + str(dcm.TriggerTime)) 

          self.TemporalPositionIdentifiers.append(acq_d + acq_t)

      self.TemporalPositionIdentifiers = np.array(self.TemporalPositionIdentifiers)
      self.uniq_TemporalPositionIdentifiers = np.unique(self.TemporalPositionIdentifiers)
      self.uniq_TemporalPositionIdentifiers.sort()

      # if an MR data contains multiple echos, we interpret it as dynamic data
      if ((self.dicomlist[0].Modality == 'MR') and (len(self.uniq_TemporalPositionIdentifiers) > 1)):
        self.series_type[0] = 'DYNAMIC'
        warnings.warn(f'Found multiple Temporal Positions in MR data set. Setting series type to DYNAMIC')

    # read static image
    if (self.series_type[0] == 'STATIC') or (self.series_type[0] == 'WHOLE BODY'):
      self.nframes = 1

      # the extra check if the dimension of the pixel array is bigger than 2 is needed
      # because there are GE CT with erroneouly contain NumberOfFrames in classical slice by
      # slice dicom files
      if 'NumberOfFrames' in self.firstdcmheader and (self.firstdcmheader.pixel_array.ndim > 2):
        # read multi slice data (the whole 3d volume is in one dicom file)
        if self.filelist is not None:
          data = self.get_multislice_3d_data(dicom.read_file(self.filelist[0]))
        else:
          data = self.get_multislice_3d_data(self.dicomlist[0])
      else:
        # read 3d data stored in multiple 2d dicom files
        data = self.get_3d_data(self.dicomlist)

    # read dynamic / gated images
    else: 
      self.nframes = len(self.uniq_TemporalPositionIdentifiers)
      if frames is None: frames = np.arange(self.nframes) + 1

      data = []
      self.AcquisitionTimes = np.empty(self.nframes, dtype = object)
      self.AcquisitionDates = np.empty(self.nframes, dtype = object)

      for frame in frames:
        if self.verbose: print('Reading frame ' + str(frame) + ' / ' + str(self.nframes))
        inds = np.where(self.TemporalPositionIdentifiers == self.uniq_TemporalPositionIdentifiers[frame - 1])[0]
        data.append(self.get_3d_data([self.dicomlist[i] for i in inds]))

        # add the acuqisiton date and time of every frame
        if 'AcquisitionTime' in self.dicomlist[inds[0]]: 
          self.AcquisitionTimes[frame - 1] = self.dicomlist[inds[0]].AcquisitionTime
        if 'AcquisitionDate' in self.dicomlist[inds[0]]: 
          self.AcquisitionDates[frame - 1] = self.dicomlist[inds[0]].AcquisitionDate

      data = np.squeeze(np.array(data))

    return data 
   
  #------------------------------------------------------------------------------------------------------
  def get_multislice_3d_data(self, dcm_data):
    """get data from a multislice 3D dicom file (as e.g. used in SPECT or molecubes dicoms)

    Parameters
    ----------
    dcm_data : pydicom FileDataset 
      as returned by pydicom.read_file

    Returns
    -------
    a 3D numpy array
    """
    pixelarray = dcm_data.pixel_array.copy()

    self.Nslices, self.Nrows, self.Ncols = pixelarray.shape

    if 'RescaleSlope' in dcm_data: 
      pixelarray = pixelarray * dcm_data.RescaleSlope 
    elif 'SharedFunctionalGroupsSequence' in dcm_data:
      try:
        # molecubes multi slice data
        pixelarray = pixelarray * float(dcm_data.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0].RescaleSlope)
      except AttributeError:
        # pmod multi slice data 
        pixelarray = pixelarray * float(dcm_data.PerFrameFunctionalGroupsSequence[0].PixelValueTransformationSequence[0].RescaleSlope)
   

    if 'RescaleIntercept' in dcm_data: 
      pixelarray = pixelarray +  dcm_data.RescaleIntercept
    elif 'SharedFunctionalGroupsSequence' in dcm_data:
      try:
        # molecubes multi slice data
        pixelarray = pixelarray + float(dcm_data.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0].RescaleIntercept)
      except AttributeError:
        # pmod multi slice data 
        pixelarray = pixelarray + float(dcm_data.PerFrameFunctionalGroupsSequence[0].PixelValueTransformationSequence[0].RescaleIntercept)

    if 'SliceThickness' in dcm_data:
      self.sliceDistance = float(dcm_data.SliceThickness)
    else:
      # PMOD multi slice data
      self.sliceDistance = float(dcm_data.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness)
     
    self.n0, self.n1, self.n2 = pixelarray.shape

    # generate the directional vectors and the offset
    self.v1 = np.array([self.y[0]*self.dr,
                        self.y[1]*self.dr,
                        self.y[2]*self.dr])

    self.v2 = np.array([self.x[0]*self.dc,
                        self.x[1]*self.dc,
                        self.x[2]*self.dc])

    self.v0  = np.cross(self.v2, self.v1)
    self.v0 /= np.sqrt((self.v0**2).sum()) 
    self.v0 *= self.sliceDistance 

    # heuristic modification of v0 and normdir if SpacingBetweenSlices is negative
    # tested on Siemens SPECT data
    if 'SpacingBetweenSlices' in dcm_data:
      if float(dcm_data.SpacingBetweenSlices) < 0:
        self.v0 *= -1
        self.normdir *= -1

    ipp = None

    if 'DetectorInformationSequence' in dcm_data:
      if 'ImagePositionPatient' in dcm_data.DetectorInformationSequence[0]:
        ipp = dcm_data.DetectorInformationSequence[0].ImagePositionPatient
        self.offset = np.array(ipp, dtype = np.float)
    elif 'PerFrameFunctionalGroupsSequence' in dcm_data:
      if 'PlanePositionSequence' in dcm_data.PerFrameFunctionalGroupsSequence[0]:
        # this is for molecubes dicom data
        ipp = dcm_data.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient
        self.offset = np.array(ipp, dtype = np.float)
    elif 'ImagePositionPatient' in dcm_data:
      # this is for RayStation dicom data
      ipp = dcm_data.ImagePositionPatient
      self.offset = np.array(ipp, dtype = np.float)

    if ipp is None:
      self.offset = np.zeros(3)
      warnings.warn('Cannot find ImagePositionPatient in dicom header. Setting it to [0,0,0]')

    # reorient the patient volume to standard LPS orientation
    patvol = self.reorient_volume(pixelarray)

    self.voxsize = np.array([self.xvoxsize, self.yvoxsize, self.zvoxsize])

    self.affine       = np.eye(4)
    self.affine[:3,0] = self.v0
    self.affine[:3,1] = self.v1
    self.affine[:3,2] = self.v2
    self.affine[:3,3] = self.offset


    return patvol
    

  #------------------------------------------------------------------------------------------------------
  def get_3d_data(self, dicomlist):
    """get the 3D data from a list of dicom data sets

    Parameters
    ----------
    dicomframes : list 
      list of dicom objects from pydicom

    Returns
    -------
    a 3D numpy array
    """
    d = [self.distanceMeasure(x) for x in dicomlist]

    # sort the list according to the distance measure
    dicomlistsorted        = [x for (y,x) in sorted(zip(d,dicomlist))]
    pixelarraylistsorted   = [x.pixel_array for x in dicomlistsorted]

    # store the sorted list of SOPInstanceUIDs which is needed when writing RTstructs
    self.sorted_SOPClassUIDs    = [x.SOPClassUID for x in dicomlistsorted]
    self.sorted_SOPInstanceUIDs = [x.SOPInstanceUID for x in dicomlistsorted]

    self.Nslices           = len(dicomlistsorted)
    self.Nrows, self.Ncols = pixelarraylistsorted[0].shape

    if 'RescaleSlope' in dicomlistsorted[0]: 
        RescaleSlopes  = [np.float(x.RescaleSlope) for x in dicomlistsorted]
    else:
        RescaleSlopes = [1.0] * len(dicomlistsorted)

    if 'RescaleIntercept' in dicomlistsorted[0]: 
        RescaleIntercepts = [np.float(x.RescaleIntercept) for x in dicomlistsorted]
    else:
        RescaleIntercepts = [0.0] * len(dicomlistsorted)
    
    # rescale the pixelarrays with the rescale slopes and intercepts
    for i in range(len(pixelarraylistsorted)): 
        pixelarraylistsorted[i] = pixelarraylistsorted[i]*RescaleSlopes[i] + RescaleIntercepts[i]

    # get the first and last ImagePositionPatient vectors
    self.T1 = np.array(dicomlistsorted[0].ImagePositionPatient , dtype = np.float)
    self.TN = np.array(dicomlistsorted[-1].ImagePositionPatient, dtype = np.float)
    self.dT = self.T1 - self.TN

    # get the distance between the dicom slices
    self.sliceDistance = sorted(d)[1] - sorted(d)[0]
  
    # calculate the (x,y,z) offset of voxel [0,0,0]  
    self.offset = 1.0*self.T1

    # now we calculate the direction vectors when moving 1 voxel along axis 0, 1, 2
    self.v0     = np.array([1.0*self.dT[0]/ (1 - self.Nslices),
                            1.0*self.dT[1]/ (1 - self.Nslices),
                            1.0*self.dT[2]/ (1 - self.Nslices)])

    self.v1 = np.array([self.y[0]*self.dr,
                        self.y[1]*self.dr,
                        self.y[2]*self.dr])

    self.v2 = np.array([self.x[0]*self.dc,
                        self.x[1]*self.dc,
                        self.x[2]*self.dc])

    # generate a 3d volume from the sorted list of 2d pixelarrays
    patvol               = np.array(pixelarraylistsorted) 
    self.n0, self.n1, self.n2 = patvol.shape

    # reorient the patient volume to standard LPS orientation
    patvol = self.reorient_volume(patvol)

    self.voxsize = np.array([self.xvoxsize, self.yvoxsize, self.zvoxsize])

    # create affine matrix
    self.affine       = np.eye(4)
    self.affine[:3,0] = self.v0
    self.affine[:3,1] = self.v1
    self.affine[:3,2] = self.v2
    self.affine[:3,3] = self.offset

    return patvol

  #--------------------------------------------------------------------------
  def get_3d_overlay_img(self, tag = 0x6002):
    """ Read dicom overlay information and convert it to a binary image.
    
    Parameters
    ----------
    tag : int in hex, optional
      overlay tag to use, default 0x6002 

    Note
    ----
    (1) up to 8 overlays can be saved in the tags 0x6000, 0x6002, 0x6004, 0x6006, 0x6008, 0x600a, 0x600c, 0x600e

    (2) the generation of the binary label image was only tested for transaxial CT overlays so far

    (3) so far it only works for negative origins

    Returns
    -------
    3d numpy array
      a binary array containing the overlay information
    """
    # up to know we assume that the input dicom list is a 3D volume
    # so we use all dicom files as input for the overlay

    if not self.read_all_dcms:
      if self.dicomlist is None:
        self.dicomlist = [dicom.read_file(x) for x in self.filelist] 

      self.read_all_dcms = True

    d = [self.distanceMeasure(x) for x in self.dicomlist]

    nrows = self.firstdcmheader.Rows
    ncols = self.firstdcmheader.Columns

    # sort the list according to the distance measure
    dicomlistsorted        = [x for (y,x) in sorted(zip(d,self.dicomlist))]

    overlay_imgs = []
 
    for dcm in dicomlistsorted: 
      if [tag,0x3000]  in dcm: 
        # read the number of rows and columns for the overlay image
        orows = dcm[tag,0x0010].value  
        ocols = dcm[tag,0x0011].value  

        # read the overlay origin
        orig  = dcm[tag,0x0050].value

        # read the overlay data
        overlay = dcm[tag,0x3000].value

        # the bit order of np.unpackbits is not the one of the dicom overlay standard
        # which is why we need to reverse it (middle reshape)
        tmp = np.unpackbits(np.frombuffer(overlay, dtype = 'uint8')).reshape(-1,8)[:,::-1].flatten()[:(orows*ocols)].reshape(orows,ocols)
      
        # crop the image to the correct dimensions
        # if the origin is negative, we have to crop the image
        if orig[0] < 0:
          tmp = tmp[-orig[0]:,:]  

        if orig[1] < 0:
          tmp = tmp[:,-orig[1]:]  

        r = min(nrows,tmp.shape[0])
        c = min(ncols,tmp.shape[1])

        tmp2 = np.zeros((nrows,ncols), dtype = 'uint8')
        tmp2[:r,:c] = tmp[:r,:c]
 
        #tmp = tmp[-orig[0]:(-orig[0]+nrows),-orig[1]:(-orig[1]+ncols)]

        overlay_imgs.append(tmp2)
      else:
        overlay_imgs.append(np.zeros((nrows,ncols), dtype = 'uint8'))

    return np.swapaxes(np.array(overlay_imgs),0,2)

  #--------------------------------------------------------------------------
  def distanceMeasure(self,dicomslice):
    # see http://nipy.org/nibabel/dicom/dicom_orientation.html
    # d = position vector in the direction of the normal vector
    T  = np.array(dicomslice.ImagePositionPatient, dtype = np.float)
    d = np.dot(T,self.n)
    return d

  #def setAttibute(self, attribute, value):
  #  for dcm in dicomlist: setattr(dcm,attribute,value) 
  #  for dcm2 in dicomlistsorted: setattr(dcm2,attribute,value)

  #def write(self):
  #  for i in xrange(len(self.filelist)):
  #      if self.verbose: print("\nWriting dicom file: ", self.filelist[i])
  #      dicom.write_file(self.filelist[i],dicomlist[i])

################################################################################
################################################################################
################################################################################

class DicomSearch:
  
  def __init__(self, path, pattern = '*.dcm'):
    self.path     = path
    self.pattern  = pattern 
    self.allfiles = glob.glob(os.path.join(self.path,self.pattern)) 

    self.UIDs     = []

    # first read all dicom images to get the UIDs
    for fname in self.allfiles:
      dicomfile = dicom.read_file(fname, force = True)
      if 'SeriesInstanceUID' not in dicomfile:
        continue
      self.UIDs.append(dicomfile.SeriesInstanceUID)
      dicomfile.clear()
    
    # now lets remove all duplicates
    self.uniqueUIDs    = list(set(self.UIDs))

    self.inds       = []
    self.files      = []
    self.SeriesDescription = []
    self.AcquisitionDate   = []
    self.AcquisitionTime   = []
    self.PatientName       = []
    self.Modality          = []

    # now read 1 dicom file of each unique UID and extract some usefule information
    for uid in self.uniqueUIDs:
      self.inds.append([i for i in range(len(self.UIDs)) if self.UIDs[i] == uid])
      self.files.append([self.allfiles[x] for x in self.inds[-1]])
     
      dicomfile = dicom.read_file(self.files[-1][0])
      if 'SeriesDescription' in dicomfile : self.SeriesDescription.append(dicomfile.SeriesDescription)
      else                                : self.SeriesDescription.append(None)
      if 'AcquisitionDate'   in dicomfile : self.AcquisitionDate.append(dicomfile.AcquisitionDate)
      else                                : self.AcquisitionDate.append(None)
      if 'AcquisitionTime'   in dicomfile : self.AcquisitionTime.append(dicomfile.AcquisitionTime)
      else                                : self.AcquisitionTime.append(None)
      if 'PatientName'       in dicomfile : self.PatientName.append(dicomfile.PatientName)
      else                                : self.PatientName.append(None)
      if 'Modality'          in dicomfile : self.Modality.append(dicomfile.Modality)
      else                                : self.Modality.append(None)

      dicomfile.clear()
