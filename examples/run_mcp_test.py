""" 
============== 
Performing a Multiple Comparisons Permutation Test
==============

Statistical test to quantify the difference between two populations of DSMs (or
a single population and a reference DSM) and identify the pixels in which the
difference is statistically significant.
""" 
import numpy as np
from rtdsm import dsmprocessing as dp

###############################################################################
# The multiple-comparisons permulation test (MCP) can be used to determine if
# two groups of DSMs are statistically different from each other. In this example
# we will compare two randomly generated sets of DSMs.

# Create mean DSMs for each group
g1mean = np.ones((10,10))*30
g2mean = np.ones((10,10))*30    #group 2 has a hotspot in the middle
g1mean[3:7,3:7] = 34
# Define a st. dev array to add noise to the samples
SDmap = np.ones((10,10))
SDmap[[4,8],:],SDmap[[5,7],:],SDmap[6,:] = 1,3,5 #noise increases in a central band

# Create 10 unique DSMs for each group
g1dsms, g2dsms = [],[]
for i in range(10):
    g1dsms.append(np.random.normal(g1mean,SDmap) )
    g2dsms.append(np.random.normal(g2mean,SDmap) )

###############################################################################
# Next, we will structure our DSM data into the appropriate structure for the MCP
# function. This takes the form of an M x N numpy array where M is the number of
# observations/DSMs in a group and N is the number of pixels in a DSM. A built in
# function format_MCP_data() can be used to flatten DSMs into the correct format.

g1_arr, shape1 = dp.format_MCP_data(g1dsms)
g2_arr, shape2 = dp.format_MCP_data(g2dsms)

###############################################################################
# Now we run the MCP test. In this example, we will treat our data as independent
# from each other and test the alternate hypothesis that g2 DSMs received greater
# doses than the g1 DSMs.

results = dp.MCP_test((g1_arr,g2_arr),testType='independent',variation='greater',n_permutations=2000,randomSeed=4)

###############################################################################
# 'Results' provides several valuable pieces of information. For example, while
# 'pvalue' gives a single numerical result of the MCP test, 'obs_statmap_1D' 
# gives the map of test statistic values in the observed sample. This map can be
# combined with the information in 'percentile_thresholds' to identify which 
# subregions receive statistically different doses between the two groups. Lets
# plot this information alongside the original DSMs

print('p-value;', results['pvalue'])

import matplotlib.pyplot as plt 
from matplotlib import colors
fig, ((a1,a2),(a3,a4)) = plt.subplots(2,2, figsize=(5, 4))

# Plot the mean DSM of each group.
g1_avg = np.mean(g1_arr,axis=0).reshape(shape1)
f1 = a1.pcolormesh(g1_avg,cmap='inferno',vmin=25, vmax=40)
g2_avg = np.mean(g2_arr,axis=0).reshape(shape2)
f2 = a2.pcolormesh(g2_avg,cmap='inferno',vmin=25, vmax=40)
cb1 = fig.colorbar(f1, ax=a1)
cb2 = fig.colorbar(f2, ax=a2)
a1.axis('off')
a2.axis('off')
a1.set_title('Avg. DSM in g1')
a2.set_title('Avg. DSM in g2')

# Plot the test statistic map a
f3 = a3.pcolormesh(results['obs_statmap_1D'].reshape(shape1) ,cmap='viridis')
cb3 = fig.colorbar(f3, ax=a3)
a3.axis('off')
a3.set_title('Test statistic map')

# Plot the same information, but masked to different p-value levels
cmapMCP = colors.ListedColormap(['#151515','#404040','#696969', '#ffbdbd','#ff8282','#ff4040'])
thresholds = list(results['percentile_thresholds'][:,1])    #get the statistic values for each threshold
thresholds.append(100)
cmapPvalues = list(results['percentile_thresholds'][:,0]) #get the p-value levels for each threshold
cmapPvalues.append('')
norm = colors.BoundaryNorm(thresholds, cmapMCP.N)   #apply to the colormap
# add the plot
f4 = a4.pcolormesh(results['obs_statmap_1D'].reshape(shape1) ,cmap=cmapMCP, norm=norm)
cb4 = fig.colorbar(f4, ax=a4)
cb4.ax.set_yticklabels(cmapPvalues)
a4.axis('off')
a2.set_title('p-value map')

fig.tight_layout()
plt.show()

###############################################################################
# As a final example, we'll demonstrate how we can perform a single sample test
# using the 'paired' testType. Our alternate hypothesis will be that g1 DSMs are
# different from g1mean (our population value). To do this we'll need to
# calculate a set of dose difference maps to use in our test.

# Calculate the difference maps between the DSMs in g1 and the g2mean
g3_arr = np.copy(g1_arr)
for i in range(10):
    g3_arr[i,:] = g3_arr[i,:] - g2mean.flatten()

# Run the MCP test using the paired testType and a single data group
results_1sample = dp.MCP_test(g3_arr,testType='paired',variation='two-sided',n_permutations=2000,randomSeed=4)
print('p-value;', results['pvalue'])

###############################################################################
# We can plot this data the same as we did before to visualize where the test
# found statistically different pixels.

fig, ((a1,a2,a3)) = plt.subplots(1,3, figsize=(6, 2))

# Plot the mean DSM of each group.
g3_avg = np.mean(g3_arr,axis=0).reshape(shape1)
f1 = a1.pcolormesh(g3_avg,cmap='bwr',vmin=-5, vmax=5)
cb1 = fig.colorbar(f1, ax=a1)
a1.axis('off')
a1.set_title('Avg. diff. map')

# Plot the test statistic map a
f2 = a2.pcolormesh(results_1sample['obs_statmap_1D'].reshape(shape1) ,cmap='viridis')
cb2 = fig.colorbar(f2, ax=a2)
a2.axis('off')
a2.set_title('Test statistic map')

# Plot the same information, but masked to different p-value levels
thresholds = list(results_1sample['percentile_thresholds'][:,1])    #get the statistic values for each threshold
thresholds.append(100)
norm = colors.BoundaryNorm(thresholds, cmapMCP.N)   #apply to the colormap
# add the plot
f3 = a3.pcolormesh(results_1sample['obs_statmap_1D'].reshape(shape1) ,cmap=cmapMCP, norm=norm)
cb3 = fig.colorbar(f3, ax=a3)
cb3.ax.set_yticklabels(cmapPvalues)
a3.axis('off')
a3.set_title('p-value map')

fig.tight_layout()
plt.show()