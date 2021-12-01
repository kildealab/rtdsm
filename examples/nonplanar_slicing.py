""" 
============== 
Using the non-planar slicing method
==============

Defines points on an ROI surface mesh to create a dose-surface map from using the
non- planar slicing method.
""" 

import pyvista as pv
import numpy as np
import rtdsm
from rtdsm import examples

###############################################################################
# Read in an ROI from a Dicom file and generate its surface mesh.
Filename = examples.RSfile()
pointdata, centroiddata = rtdsm.get_pointcloud('Rectum', Filename)
imgres = [0.908, 0.908, 2.0]   # the voxel resolution of the ROI’s image
verts, faces, pvfaces = rtdsm.get_cubemarch_surface(pointdata, imgres)
mesh = pv.PolyData(verts, faces=pvfaces)
surf = mesh.extract_geometry()  #extracts the outer surface of the mesh
surf = surf.smooth(n_iter=200)

###############################################################################
# The next step is to define a smooth path through the middle of the ROI from
# which we’ll establish  our slicing planes. Because the slicing function
# necessitates a densely sampled, but also smooth spline in order to accurately
# determine the spline’s tangents, we use a two step approach.

# First, create a spline with 1/5th the default number of knots to smooth the spline
spl0 = pv.Spline(centroiddata, n_points = int(len(centroiddata)/5) ) 
# Next, use the knots of this spline to create a smooth, densely sampled one
spline = pv.Spline(spl0.points, n_points = 1000) 

###############################################################################
# With our spline created we can now determine the points along the spline at
# which slices should be defined. For this example let us define slices every
# 3 mm along its length. 

samplocs = rtdsm.sample_centerline(spline, 3, stepType = 'dist')

###############################################################################
# With the locations where we will create slices defined, all that is left is to
# decide how many points to sample around the circumference of these slices.
# Let’s use 30.

# Create slices of the ROI mesh composed of 30 points each 
slices = rtdsm.sample_nonplanar_slices(surf, spline, samplocs, 30)

###############################################################################
# The output dictionary, slices, houses the data for each slice using the indices
# in samplocs as keys. Each slice entry contains the following:
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
for s in samplocs:
    Points = slices[s]['Slice'].points[1:]
    ax.plot(Points[:,0], Points[:,1], Points[:,2],color='#0000FF60')
plt.show()