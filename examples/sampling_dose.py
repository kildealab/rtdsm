""" 
============== 
Sampling dose from an RTDose matrix
==============

Calculates dose at a set of points and plots the resulting colormap
""" 

import numpy as np
import rtdsm
from rtdsm import examples
Filename = examples.RDfile()

###############################################################################
# Begin by defining a set of points to sample dose at. We'll make three 
# different input data types: (1) a single point, (2) a list of points, and (3)
# a 5 x 5 grid of points.

xvals = np.linspace(-10,10, 5)
yvals = np.linspace(-15,5, 5)
z = 30
pointset = []
for i in range(len(xvals)):
    row = []
    for j in range(len(yvals)):
        row.append([xvals[i], yvals[j], z])
    pointset.append(row) 

grid = np.array(pointset)
line = grid[:,0]
point = grid[0,0]

###############################################################################
# Retrieve the dose grid information from an RTDose file. The dose for the points
# in the pointset can then be determined using regular grid interpolation with
# the sample_dose() method.

dosegrid, xs, ys, zs = rtdsm.get_doseinfo(Filename)

pointdose = rtdsm.sample_dose(point, dosegrid, xs, ys, zs)
linedose = rtdsm.sample_dose(line, dosegrid, xs, ys, zs)
griddose = rtdsm.sample_dose(grid, dosegrid, xs, ys, zs)

###############################################################################
# If we print the results we can compare the dose returned for each method.

print('point:',pointdose,
    '\nlist:',linedose,
    '\ngrid:',griddose)

###############################################################################
# The result returned when a grid of input data is provided is formatted the 
# same as a dose-surface map and can be visualized as such. In practice we use
# another function, sample_dsm(), to format data from the slicing process into
# the correct format for sample_dose(), rather than doing so manually.

# Plot the dose map to see the result.
import matplotlib.pyplot as plt 
fig, ax = plt.subplots()  
f = ax.pcolormesh(griddose, cmap='inferno')
fig.colorbar(f,ax=ax)
plt.show()