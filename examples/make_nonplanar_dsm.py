""" 
============== 
Creating a non-planar DSM
==============

Creates a dose-surface map from dicom data using the non-planar slicing method. 
For a more detailed explanation of each step consult the examples for basic DSM 
creation. 
""" 

###############################################################################
# Begin by loading the relevant packages and the data.

import numpy as np
import pyvista as pv
import rtdsm
from rtdsm import examples
RSfile = examples.RSfile()        # Get the path to the RTStructure dicom
RDfile = examples.RDfile()    # Get the path to the RTDose dicom

###############################################################################
# Use an ROI from the RTStructure file to create a surface mesh of a structure. 
# We do this by retrieving point cloud data for the structure from the dicom file
# and using a smoothed marching cubes algorithmic approach to create a connected
# mesh. 

# Get the point cloud data and slice centroid locations from the dicom file
pointdata, centroiddata = rtdsm.get_pointcloud('Rectum', RSfile)
# Run the algorithm to create a mesh and format as a smoothed surface mesh
imgres = [0.908, 0.908, 2.0]   # the voxel resolution of the ROI’s image
verts, faces, pvfaces = rtdsm.get_cubemarch_surface(pointdata, imgres)
mesh = pv.PolyData(verts, faces=pvfaces) #face data needs to be pyvista formatted
surf = mesh.extract_geometry()  # extracts the outer surface of the mesh
surf = surf.smooth(n_iter=200)  # smooths the mesh so not voxelated

###############################################################################
# For the non-planar slicing method we need a smooth spline representing a path
# through the length of the ROI in order to create slices that follow the shape
# of this path. We can do this by creating a spline fit to the centroids of the
# original slices of the ROI. Once we have this spline we can define points along
# it where we would like to create slices. In this example we will create a new
# slice every 3 mm along the spline’s length. 

spl0 = pv.Spline(centroiddata, n_points = int(len(centroiddata)/5) ) # initial, smoothed spline
spline = pv.Spline(spl0.points, n_points = 1000) # spline redefined with more points
# Determine the set of points along the spline spaced 3 mm apart starting from the bottom.
slicelocs = rtdsm.sample_centerline(spline, 3, stepType = 'dist')

###############################################################################
# Using the slice locations we defined, create slices of the ROI mesh in planes
# that follow the  trajectory of the centerline spline path. The non-planar
# slicing method will check if slices overlap with one another and correct them
# if they do using the method of Witztum et al. For this example each slice ring
# will be composed of 45 points.

slices = rtdsm.sample_nonplanar_slices(surf, spline, slicelocs, 45)

###############################################################################
# Collect the dose matrix information associated with the image of the ROI and
# use it to calculate dose at each point on every slice and output the result as
# a dose-surface map.

dosegrid, xs, ys, zs = rtdsm.get_doseinfo(RDfile)
# Calculate the DSM 
dosemap = rtdsm.sample_dsm(slices, slicelocs, dosegrid, xs, ys, zs)

###############################################################################
# Plot the dose map. 

import matplotlib.pyplot as plt 
fig, ax = plt.subplots()  
f = ax.pcolormesh(dosemap, cmap='inferno')
fig.colorbar(f,ax=ax)
plt.show()