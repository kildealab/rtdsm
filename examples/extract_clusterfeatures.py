""" 
============== 
Calculating the features of a cluster mask
==============

Calculates the most commonly used features of DSM cluster masks from the 
literature. These features are defined as: 
- “area”: The percent of the total DSM area covered by the cluster.
- “centroid”: The center of mass of the cluster, given in terms of indices of the
original DSM.
- “centroidpcnt”: The center of mass of the cluster, given in terms of percent of 
the lateral and longitudinal span of the DSM.
- “latext”: The maximum lateral span of the cluster, given as a percent of the 
total lateral span.
- “longext”: The maximum longitudinal span of the cluster, given as a percent of 
the total longitudinal span.
- “bounds”: The indices of the lateral and longitudinal bounds of the cluster 
((latmin,latmax),(lngmin,lngmax))
""" 

from rtdsm import examples
from rtdsm import dsmprocessing as dp

###############################################################################
# Cluster-based dose-surface map features are often used to characterize the size
# and location of specific dose levels delivered to a structure’s outer surface
# and can be calculated as follows.

# Calculate the features of the 10Gy cluster region
DSM = examples.DSM1()     # load the DSM
cluster = dp.clustermask(DSM, 20)
features = dp.cluster_features(cluster)

###############################################################################
# Now that we’ve calculated the features, we can examine them along with the DSM
# and the cluster mask. The most useful features for further analysis typically
# include the cluster area, its centroid,  and its lateral and longitudinal extent.

print(features)

# Plot the DSM and the cluster mask to visually compare to the feature results
import matplotlib.pyplot as plt 
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.pcolormesh(DSM, cmap='inferno')
ax2.pcolormesh(cluster, cmap='gray')
for ax in fig.get_axes():
    ax.label_outer()
ax1.set_title("DSM")
ax2.set_title("20Gy Cluster")
plt.show()