""" 
============== 
Creating a planar DSM
==============

Creates a dose-surface map from dicom data using the planar slicing method. For a
more detailed explanation of each step consult the examples for basic DSM creation. 
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
mesh = pv.PolyData(verts, faces=pvfaces)
surf = mesh.extract_geometry()  # extracts the outer surface of the mesh
surf = surf.smooth(n_iter=200)  # smooths the mesh so not voxelated

###############################################################################
# Identify where we want each slice (row) in the dose-surface map to originate from. 
# In the planar case, we want to create a slice at each location a slice of the 
# ROI existed in the original image.

slicelocs = np.arange(len(centroiddata)) # array indicating which points to use

###############################################################################
# Using the slice locations we set, create slices of the mesh in the same planes 
# as the slices in the original image. Each slice ring will include 45 points.

slices = rtdsm.sample_planar_slices(surf, centroiddata, slicelocs, 45)

###############################################################################
# Collect the dose matrix information associated with the image of the ROI and use
# it to calculate dose at each point on every slice and output the result as a 
# dose-surface map.

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