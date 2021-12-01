""" 
============== 
Converting a DSM to another fractionation scheme
==============

Converts a dose-surface map to its 2 Gy and 3 Gy perfraction Equivalent Doses.
""" 

import numpy as np
from rtdsm import examples
from rtdsm import dsmprocessing as dp

###############################################################################
# In cases where a dose-surface map needs to be converted to its Biologically 
# Effective Dose (BED) or Equivalent Dose (EQD) rtdsm includes methods
# to do so.
# The example DSM is for treatment plan with a prescription of 36.25Gy in 5 
# fractions, which we will convert to its 2Gy per fraction equivalent

DSM = examples.DSM1()     # load the DSM
n_frac = 5 # number of fractions in the original plan
aB = 3 #alpha-beta ratio to use for the conversion

# Calculate the BED of the DSM
BED = dp.bed_calculation(DSM,aB,n_frac)
# Calculate the EQD_2Gy and EQD_3Gy of the DSM
EQD2 = dp.eqd_gy(DSM,aB,n_frac,2)
EQD3 = dp.eqd_gy(DSM,aB,n_frac,3)

# Plot the original DSM and its conversions
import matplotlib.pyplot as plt 
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2)
f1 = ax1.pcolormesh(DSM, cmap='inferno')
f2 = ax2.pcolormesh(BED, cmap='inferno')
f3 = ax3.pcolormesh(EQD2, cmap='inferno')
f4 = ax4.pcolormesh(EQD3, cmap='inferno')
fig.colorbar(f1, ax=ax1)
fig.colorbar(f2, ax=ax2)
fig.colorbar(f3, ax=ax3)
fig.colorbar(f4, ax=ax4)
ax1.set_title("DSM")
ax2.set_title("BED")
ax3.set_title("EQD2")
ax4.set_title("EQD3")
fig.tight_layout()
plt.show()