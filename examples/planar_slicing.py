""" 
============== 
Using the planar slicing method
==============

Defines points on an ROI surface mesh to create a dose-surface map from using the
planar slicing method.
""" 

import pyvista as pv
import numpy as np
import rtdsm
from rtdsm import examples

###############################################################################
# Read in an ROI from a Dicom file and generate it’s surface mesh.
Filename = examples.RSfile()
pointdata, centroiddata = rtdsm.get_pointcloud('Rectum', Filename)
imgres = [0.908, 0.908, 2.0]   # the voxel resolution of the ROI’s image
verts, faces, pvfaces = rtdsm.get_cubemarch_surface(pointdata, imgres)
mesh = pv.PolyData(verts, faces=pvfaces)
surf = mesh.extract_geometry()  #extracts the outer surface of the mesh
surf = surf.smooth(n_iter=200)

###############################################################################
# Now we need to determine where along the ROI we will sample it. For the planar
# slicing method the convention is to sample the mesh on the same slices as those
# in the original image the ROI was contoured on. This means we can simply define
# our sampling points as all points in the  centroiddata array. If we wanted to
# only use every second slice we could define a different list of points, as
# shown below.

sampallslices = np.arange(len(centroiddata))
samphalfslices = np.arange(0,len(centroiddata),2)
print(sampallslices[:5], samphalfslices[:5])

###############################################################################
# With the locations where we will create slices defined, all that is left is to
# decide how many points to sample around the circumference of these slices and
# to specify the shared axis of along which all the slices will be taken. In most
# cases this will be the Z axis of the original medical image,  which is defined
# by the vector [0,0,1]. 

# Create slices of the ROI mesh composed of 30 points each and aligned with
# longitudinal axis of the image. 
slices = rtdsm.sample_planar_slices(surf, centroiddata, sampallslices, 30, [0,0,1])

###############################################################################
# The output dictionary, slices, houses the data for each slice using the indices
# in sampallslices as keys. Each slice entry contains the following:
#   - “PlnNorm”: The unit normal vector of the plane used to create the slice.
#   - “Org”: The point from which ray casting to define the slice originated from.
#   -”Slice”: A pyvista PolyData object of the slice itself, formatted as a disk.
#    Contains the points around the circumference of the slice used to create a
#    DSM, plus the point “Org” to facilitate the creation of a triangulated mesh.
#    “Org” is always the first point in the mesh’s point data array.

# Plot the slices to visualize them
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(azim=10,elev=5)
ax.set_zlim(-35,40)
ax.set_ylim(20,70)
ax.set_xlim(-25,25)
for s in sampallslices:
    Points = slices[s]['Slice'].points[1:]
    ax.plot(Points[:,0], Points[:,1], Points[:,2],color='#0000FF60')
plt.show()