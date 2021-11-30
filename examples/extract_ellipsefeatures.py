""" 
============== 
Calculating ellipse-based features of a cluster mask
==============

Calculates the features of an ellipse fit to a DSM cluster mask using the 
approach popularized by Buetter et al. These features are defined as: 
- “area”: The percent of the total DSM area covered by the fitted ellipse.
- “eccent”: The eccentricity of the fitted ellipse.
- “anglerad”: The angle of rotation of the fitted ellipse, in radians. 
- “latproj”: The percent of the lateral span of the DSM that the projection of 
the ellipse’s lateral axis covers.
- “longproj”: The percent of the longitudinal span of the DSM that the projection
of the ellipse’s longitudinal axis covers.

Additional properties of the ellipse are also given by the ellipse fitting
function:
- “CoM”: The center of mass, in indices (x,y), of the ellipse.
- “a”,”b”: Half the width and height of the ellipse, respectively.
- ”theta”: The angle (in radians) the ellipse is rotated.
""" 

from rtdsm import examples
from rtdsm import dsmprocessing as dp

###############################################################################
# To begin we need to fit an ellipse to a cluster mask

# Calculate the features of the 10Gy cluster region
DSM = examples.DSM1()     # load the DSM
cluster = dp.clustermask(DSM, 20)
# Fit an ellipse to the cluster
ellip = dp.fit_ellipse(cluster)
print(ellip)

###############################################################################
# As we can see, the variable “ellip” is a dictionary of the variables that define
# the fitted ellipse.  These variables can be used to calculate the ellipse-based
# features of the cluster mask.

features =  dp.ellipse_features(ellip, DSM.shape)

###############################################################################
# Now that we’ve calculated the features, we can examine them along with the DSM
# and the cluster mask. The most useful features for further analysis typically
# include ellipse area, eccentricity, and the lateral and longitudinal spans of
# the projections of the ellipse’s axes. 

print(features)

# Plot the original DSM and the cluster mask to visually compare to the feature results
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.pcolormesh(DSM, cmap='inferno')
ax2.pcolormesh(cluster, cmap='gray')
ellipse = Ellipse(ellip['CoM'],width=ellip['a']*2,height=ellip['b']*2,
    angle=ellip['theta']*180/3.14159,facecolor='none',edgecolor='red')
ax2.add_patch(ellipse)
for ax in fig.get_axes():
    ax.label_outer()
plt.show()