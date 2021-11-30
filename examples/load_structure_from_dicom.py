""" 
============== 
Loading ROI data from an RT Structure Dicom file
==============

Extracts the pointcloud of a specific ROI’s vertices from an RT Structure type 
Dicom file. 
""" 

###############################################################################
# Begin by loading the relevant packages and example data

import numpy as np
import rtdsm
from rtdsm import examples
# Get the sample RTStructure file
Filename = examples.RSfile()

###############################################################################
# Now that we have the path to our RTStructure file we can request the point
# cloud data for our structure of interest. In this case, the ROI we want is
# 'Rectum'. 

pointdata, centroiddata = rtdsm.get_pointcloud('Rectum',Filename)
pointdata[:5,:]

###############################################################################
# As we can see, we now have a list of N coordinates in 3D space that correspond
# to the vertices of the ROI’s surface mesh, as defined in the Dicom file. We
# also have a list of the locations of the centroids of each slice of the ROI
# that can be useful when determining how to slice the ROI. Let’s visualize the
# data and see how it looks.

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax = Axes3D(fig)    # enable 3D plotting
ax.view_init(azim=10,elev=5)
# plot the vertices in blue
ax.scatter(pointdata[:,0], pointdata[:,1], pointdata[:,2], color = "#0000FF45" )
# plot the slice centroids in red
ax.scatter(centroiddata[:,0], centroiddata[:,1], centroiddata[:,2], color = "#FF000095" )
plt.show()

###############################################################################
# There we go! We have a point cloud of vertex data for the structure, plus some
# points following  a centerline through it. Our next steps will be to use this
# data to create a mesh of the structure.
