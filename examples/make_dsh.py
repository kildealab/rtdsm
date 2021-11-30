""" 
============== 
Creating a DSH from a DSM
==============

Calculates a dose-surface histogram using a dose-surface map.
""" 

import numpy as np
from rtdsm import examples
from rtdsm import dsmprocessing as dp

###############################################################################
# Dose surface histograms (DSHs) can be easily calculated from a DSM by
# specifying dose levels at which to sample it at.

DSM = examples.DSM1()     # load the DSM
x = np.arange(0,40,0.5) # Define the dose range of the DSH
y = dp.sample_dsh(DSM, x) # Calculate the y data

# Plot the DSM and the DSH
import matplotlib.pyplot as plt 
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.pcolormesh(DSM, cmap='inferno')
ax2.plot(x,y)
plt.show()