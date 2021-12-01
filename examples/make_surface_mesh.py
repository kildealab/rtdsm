""" 
============== 
Creating a surface mesh of an ROI
==============

Creates a pyvista.PolyData surface mesh of an ROI.
""" 

import numpy as np
import pyvista as pv
import rtdsm
from rtdsm import examples
# Get the sample RTStructure file and extract the pointcloud data
Filename = examples.RSfile()
pointdata, centroiddata  = rtdsm.get_pointcloud('Rectum', Filename)

###############################################################################
# In order to create quality dose-surface maps with no missing data it is
# extremely important to ensure that the ROI mesh does not contain any holes,
# other than those that exist at the top and bottom of a tubular structure (AKA,
# no leaky pipes). For this reason we recommend smoothing a mesh generated with a
# marching cubes algorithm to ensure no holes are created.  Other methods to
# connect vertices can be used (like Delaunay triangulation), but the mesh
# quality needs to be assessed before moving on to creating slices of it.

###############################################################################
# Because the marching cubes algorithm requires an input mask of 0s and 1s, it is
# important to  provide the voxel resolution of the mask in order to convert the
# vertex coordinates from the Dicom in mm/cm/inches to indices of a 3D array.
# While using the resolution of the medical image the ROI was drawn on is
# recommended (such as [0.908,0.908,2]), a different in-plane resolution can be
# used (eg. [0.85,0.85,2] provided the slice thickness resolution is correct. In
# this example we will  use the image’s actual resolution.

imgres = [0.908, 0.908, 2.0]
verts, faces, pvfaces = rtdsm.get_cubemarch_surface(pointdata, imgres)

###############################################################################
# We now have vertex and face data for the connected mesh. Face data is provided
# in two  differently formatted arrays: faces, in which each row is the set of
# vertex indices that comprise the face, and pvfaces, which also includes the
# additional information of how many vertices comprise each face. This alternate
# format is required in order to create a pyvista.PolyData mesh from face and
# vertex data.

mesh = pv.PolyData(verts, faces=pvfaces)
surf = mesh.extract_geometry()  #extracts the outer surface of the mesh

###############################################################################
# If we plot the surface mesh in its current form we can see it’s slightly
# voxelated due to the mask used to create it. In order to remedy this we can
# apply smoothing to the surface to create a  cleaner mesh. Typically 200-250
# iterations is enough to yield a clean mesh without over smoothing.

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D, art3d
fig = plt.figure()
ax = Axes3D(fig)    # enable 3D plotting
ax.view_init(azim=10,elev=5)
ax.set_zlim(-35,40)
ax.set_ylim(20,70)
ax.set_xlim(-25,25)
# plot the mesh
pc = art3d.Poly3DCollection(verts[faces], facecolors="#9FE2BF08", edgecolor="#27825315") 
ax.add_collection(pc)  
plt.show()

# Smooth the pyvista mesh and plot the updated version
surf = surf.smooth(n_iter=200)
vNew, fNew = surf.points, surf.faces.reshape(-1,4)[:, 1:] #retrive the face and vertex data

fig = plt.figure()
ax = Axes3D(fig)    # enable 3D plotting
ax.view_init(azim=10,elev=5)
ax.set_zlim(-35,40)
ax.set_ylim(20,70)
ax.set_xlim(-25,25)
# plot the mesh
pc = art3d.Poly3DCollection(vNew[fNew], facecolors="#9FE2BF08", edgecolor="#27825315") 
ax.add_collection(pc)  
plt.show()