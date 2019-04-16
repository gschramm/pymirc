import sys, os
pymirc_path = os.path.join('..','..')
if not pymirc_path in sys.path: sys.path.append(pymirc_path)

import pymirc.fileio as pymf
import pymirc.viewer as pymv

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('dcm_path', help = 'dicom folder')
parser.add_argument('--dcm_pat', default = '*.dcm')
parser.add_argument('--overlay_tag', default = 0x6002)

args       = parser.parse_args()

#---------------------------------------------------------------------

dcm_data = pymf.DicomVolume(os.path.join(args.dcm_path,args.dcm_pat))
vol      = dcm_data.get_data()

voxsize = dcm_data.voxsize

oli = dcm_data.get_3d_overlay_img(tag=args.overlay_tag)

vi = pymv.ThreeAxisViewer([vol,oli], voxsize = voxsize)
