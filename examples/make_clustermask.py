""" 
============== 
Creating a dose cluster mask
==============

Determines the single largest cluster of voxels in a dose-surface map over the
dose threshold specified.
""" 

from rtdsm import examples
from rtdsm import dsmprocessing as dp

###############################################################################
# Most dose-surface map literature calculate spatial dose metrics based on the
# single largest cluster of voxels above the given dose threshold. In this
# example we’ll load a sample DSM and calculate some cluster masks for a few
# different dose levels.

DSM = examples.DSM1()     # load the DSM
# Calculate cluster masks for the 10, 15, and 20 Gy isodose regions
Clus10 = dp.clustermask(DSM, 10)
Clus15 = dp.clustermask(DSM, 15)
Clus20 = dp.clustermask(DSM, 20)

# Plot the original DSM and the three cluster masks
import matplotlib.pyplot as plt 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.pcolormesh(DSM, cmap='inferno')
ax2.pcolormesh(Clus10, cmap='gray')
ax3.pcolormesh(Clus15, cmap='gray')
ax4.pcolormesh(Clus20, cmap='gray')
for ax in fig.get_axes():
    ax.label_outer()
plt.show()