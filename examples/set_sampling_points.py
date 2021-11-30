""" 
============== 
Defining sampling points along a spline
==============

Determines a set of sampling points along a spline based on the sampling criteria.
""" 

import pyvista as pv
import numpy as np
import rtdsm

###############################################################################
# Begin by defining an arbitrary pyvista spline in 3D space. We’ll use a half-circle.

angs = np.linspace(0,np.pi,180)
points = np.array([np.cos(angs), np.sin(angs), np.zeros(len(angs)) ]).T
spline = pv.Spline(points, n_points = 400)

###############################################################################
# Now let’s define sampling points along the spline in two different ways:
# (1) specifying the step size between points we’d like, and
# (2) specifying the total number of steps we’d like. 

# Determine sampling point locations with a set step size of 0.5 units
steps1 = rtdsm.sample_centerline(spline, 0.5, stepType='dist')
# Determine 8 equally spaced sampling point locations along the length of the line
steps2 = rtdsm.sample_centerline(spline, 8, stepType='nsteps')

###############################################################################
# We can now plot the sets of sampling points and compare them. 

import matplotlib.pyplot as plt 
fig, ax = plt.subplots()  
line = ax.plot(spline.points[:,0], spline.points[:,1], color='black', label='spline')
distpoints = ax.scatter(spline.points[steps1, 0], spline.points[steps1, 1], color='b',
    marker='v',label='set step size')
ntotpoints = ax.scatter(spline.points[steps2, 0], spline.points[steps2, 1], color='r',
    marker='^',label='N total points')
ax.legend()
plt.show()