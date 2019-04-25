import sys, os
pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import numpy as np
import pymirc.fileio as pymf

from scipy.io    import readsav
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('kul_sav_file', help = 'filename of IDL sav file with KUL recon')
parser.add_argument('ref_dcm_dir',  help = 'directory containing the reference dicom files')
parser.add_argument('output_dir',   help = 'directory containing the reference dicom files')

parser.add_argument('--ref_dcm_pat',        help = 'file pattern for reference dicom files', default = '*')
parser.add_argument('--dcm_tag_file',       help = 'txt file with dcm tags to copy', 
                    default = 'ct_dcm_tags_to_copy.txt')
parser.add_argument('--kul_var_name',       help = 'name of recon variable in sav file', default = 'recon')
parser.add_argument('--series_desc_prefix', help = 'prefix for dcm series description', 
                    default = '(KUL motion corrected)')
parser.add_argument('--importer_computer',  help = 'PACS import computer', default = None)
parser.add_argument('--importer_login',     help = 'PACS import login',    default = None)

args       = parser.parse_args()
output_dir = args.output_dir

#-----------------------------------------------------------------------------------------

if os.path.isdir(output_dir):
  raise FileExistsError('output directory ' + output_dir + ' already exists')

# load the list of dicom tags to copy from the reference header from an input text file
with open(args.dcm_tag_file,'r') as f:
  tags_to_copy = [x.strip() for x in f.read().splitlines()]

# read the reference dicom volume
ref_dcm = pymf.DicomVolume(os.path.join(args.ref_dcm_dir, args.ref_dcm_pat))
ref_vol = ref_dcm.get_data()

# restore the KUL recon from save file
kul_recon = readsav(args.kul_sav_file)[args.kul_var_name]

# due to the memory conventions we have to reverse the axis order
kul_recon = np.swapaxes(kul_recon, 0, 2)

# to get the KUL recon in LPS we have to reverse the last axix
kul_recon = np.flip(kul_recon,2)

# create the dictionary of tags and values that are copied from the reference dicom header
dcm_header_kwargs = {}
for tag in tags_to_copy:
  if tag in ref_dcm.firstdcmheader:
    dcm_header_kwargs[tag] = ref_dcm.firstdcmheader.data_element(tag).value

# write the dicoms
# the number of tags to be copied from the original recon can be extented
new_series_desc = args.series_desc_prefix + ' ' + ref_dcm.firstdcmheader.SeriesDescription
dcm_out_fnames = pymf.write_3d_static_dicom(kul_recon, output_dir, 
                                            affine            = ref_dcm.affine,
                                            SeriesDescription = new_series_desc,
                                            modality          = ref_dcm.firstdcmheader.Modality,
                                            **dcm_header_kwargs)

# for PACS import we have to write two additional text files
if (args.importer_computer is not None) and (args.importer_login is not None):
  dcm_props = {}
  dcm_props['import.type']       = 'cd'
  dcm_props['study.size']        = kul_recon.shape[2] 
  dcm_props['version']           = 2
  dcm_props['study.uid']         = ref_dcm.firstdcmheader.StudyInstanceUID
  dcm_props['institution']       = 'KUL'
  dcm_props['importer.computer'] = args.importer_computer
  dcm_props['study.date']        = ref_dcm.firstdcmheader.StudyDate[:8]
  dcm_props['importer.login']    = args.importer_login
  dcm_props['patient.ead']       = ref_dcm.firstdcmheader.PatientID
  dcm_props['file.list']         = 'filelist.txt'
  dcm_props['werkomgeving']      = 'UZL'
  dcm_props['study.time']        = ref_dcm.firstdcmheader.StudyTime[:6]
  dcm_props['modality']          = ref_dcm.firstdcmheader.Modality
  dcm_props['study.description'] = ref_dcm.firstdcmheader.StudyDescription
  
  with open(os.path.join(output_dir, 'dicomimport.properties'), 'w') as f:
    for key, value in dcm_props.items():
      f.write(key + '=' + str(value) + '\n')

  with open(os.path.join(output_dir, 'filelist.txt'), 'w') as f:
    for fname in dcm_out_fnames:
      f.write(os.path.basename(fname) + '\n')
    f.write('COMPLETED')
