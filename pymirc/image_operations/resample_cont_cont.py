def resample_cont_cont(contourrows, resamplestruct):
  """ compute a contour as a set of columns and rows

  from the output of mincostpath (which was applied to an image resampled with
  resample_img_cont)
  
  Parameters
  ----------
  contourrows : 1d numpy array
    specifying a contour produced by NImincostpath. This
    contour goes from left to right and has a single point in every
    column. Therefore, it is specified as a 1D array, giving the
    row coordinate for every column.
  
  resamplestruct : dictionary
    returned resample_img_cont. The idea is to
    first resample an image with resample_img_cont, then compute
    a minimum cost path contour on the resampled image, and then
    convert that contour back to the 2D image coordinates with
    resample_cont_cont.

  Returns
  -------
  (1d numpy array, 1d numpy array)
    contraining the contour indices in the space after the inverse resampling

  Note
  ----
  Python reimplementation of J. Nuyts' Niresample_cont_cont
  """

  inrows = contourrows - (resamplestruct['nroutrows'] - 1.0) / 2.0
  
  if resamplestruct["steps"][resamplestruct["nroutrows"] - 1] < resamplestruct["steps"][0]:
    inrows = -inrows
  
  inrows *= resamplestruct["stepsize"]
  
  outcols = resamplestruct["cencols"] + inrows * resamplestruct["normcols"]
  outrows = resamplestruct["cenrows"] + inrows * resamplestruct["normrows"]
  
  return outcols, outrows
