""" 
============== 
Reading dose information from an RTDose Dicom file
==============

Extracts the dose matrix from an RTDose Dicom.
""" 

import numpy as np
import rtdsm
from rtdsm import examples
Filename = examples.RDfile()

###############################################################################
# RT Dose Dicom files can be read into numpy arrays using the method get_doseinfo()

dosegrid, xs, ys, zs = rtdsm.get_doseinfo(Filename)

###############################################################################
# The variable dosegrid is the RTDose matrix created during the radiotherapy
# treatment planning  process. The variables xs, ys, and zs are arrays that
# specify where the centers of the voxels in  dosegrid are located in the patient
# coordinate system. They are necessary to identify the dose  to an arbitrary
# point P on an ROI surface object. 
#
# IMPORTANT: The dicom standard encodes the dosegrid such that M[i,j,k] is
# formatted as M[z,y,x]. This is the reverse order of how contour point data
# is stored [x,y,z]

# We can determine the location of the center of the voxel dosegrid[40,81,80] as follows
loc = [ xs[80], ys[50], zs[40] ]
print('Dose of voxel at [40,50,80] in the grid:',dosegrid[40,50,80],
	'Gy\nLocation in the patient coordinate system:',loc,'mm')