""" 
============== 
Combining dose-surface maps
==============

Demonstrates how DSMs can be summed, averaged, or substracted from one another.
""" 

from rtdsm import examples
from rtdsm import dsmprocessing as dp
DSM1 = examples.DSM1()
DSM2 = examples.DSM2()

###############################################################################
# In some cases we may be interested in combining two or more DSMs in different
# ways, such as (1) summing the DSMs from multiple treatments, (2) comparing two
# DSMs by calculating the difference map between them, or (3) calculating the 
# average DSM for a certain population. The rtdsm package offers a method
# to do so, provided the DSMs meet the following criteria:
# - The DSMs share the same inferior border (begin at same landmark)
# - The DSMs have the same number of samples per slice (consistent n_columns)
# - The DSMs are all for the same fractionation schem (eg. converted to EQD2)
# - The spacing between slices (rows) is consistent between DSMs (eg. same set
# step size or same number of steps over the full length of the structure )


# Calculate the sum of the two DSMs
totDSM = dp.combine_dsms((DSM1,DSM2),kind='sum')
# Calculate the average of the two DSMs
avgDSM = dp.combine_dsms((DSM1,DSM2),kind='average')
# Calculate the difference of the two DSMs
difDSM = dp.combine_dsms((DSM1,DSM2),kind='difference')

###############################################################################
# If one of the input DSMs has fewer rows than the rest, the output DSM will be
# truncated to the shape of the smaller one, using the assumption that the larger
# DSM has more rows because it extends further along the structure's length.

import matplotlib.pyplot as plt 

# Plot the results of summing the DSMs
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
f1 = ax1.pcolormesh(DSM1, cmap='inferno')
f2 = ax2.pcolormesh(DSM2, cmap='inferno')
f3 = ax3.pcolormesh(totDSM, cmap='inferno')
for ax in fig.get_axes():
    ax.set_ylim(0,max(len(DSM1),len(DSM2)))
fig.colorbar(f1, ax=ax1)
fig.colorbar(f2, ax=ax2)
fig.colorbar(f3, ax=ax3)
fig.suptitle("DSM Summation")
fig.tight_layout()
plt.show()

# Plot the results of averaging the DSMs
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
f1 = ax1.pcolormesh(DSM1, cmap='inferno')
f2 = ax2.pcolormesh(DSM2, cmap='inferno')
f3 = ax3.pcolormesh(avgDSM, cmap='inferno')
for ax in fig.get_axes():
    ax.set_ylim(0,max(len(DSM1),len(DSM2)))
fig.colorbar(f1, ax=ax1)
fig.colorbar(f2, ax=ax2)
fig.colorbar(f3, ax=ax3)
fig.suptitle("DSM Averaging")
fig.tight_layout()
plt.show()

# Plot the results of subtracting the DSMs
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
f1 = ax1.pcolormesh(DSM1, cmap='inferno')
f2 = ax2.pcolormesh(DSM2, cmap='inferno')
f3 = ax3.pcolormesh(difDSM, cmap='bwr')
for ax in fig.get_axes():
    ax.set_ylim(0,max(len(DSM1),len(DSM2)))
fig.colorbar(f1, ax=ax1)
fig.colorbar(f2, ax=ax2)
fig.colorbar(f3, ax=ax3)
fig.suptitle("DSM Subtraction")
fig.tight_layout()
plt.show()