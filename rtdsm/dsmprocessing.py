import numpy as np
import math
from skimage.measure import label
from skimage.measure import moments
import cv2 as cv

####    -----------------------     ####
#     SPATIAL DOSE METRIC FUNCTIONS    #
####    -----------------------     ####
def clustermask(DSM,value,connectivity=1):
    """
    Creates a mask of the largest connected cluster of voxels above a given dose
    threshold in a DSM.

    Parameters
    ----------
    DSM : numpy.ndarray
        A dose-surface map (DSM, 2D array)
    value : float
        Dose value to use as the threshold for the mask.
    connectivity : int, optional   
        Type of connectivity used to define the cluster (1 = include neighbours one
        step horizontal or vertical, 2 = include neighbours one step horizontal, 
        vertical, or diagonal). Default is 1 (identical to Buettner definition).

    Returns
    -------
    Mask : numpy.ndarray
        Mask of the DSM where the largest continuous cluster of voxels above the 
        threshold = 1, 0 otherwise.

    """
    #STEP1: mask DSM to only include voxels above the dose vslue
    Mask = (DSM>=value).astype(int)
    #STEP2: find all the connected clusters
    clusters = label(Mask,connectivity=connectivity)
    if clusters.max() != 0:
        largest = (clusters == np.argmax(np.bincount(clusters.flat)[1:])+1).astype(int)
        return largest
    else:
        print("WARNING! No voxels greater than",value,"Gy exist in the DSM. Returning 'None' instead of a cluster mask.")
        return None

def fit_ellipse(ClusterMask):
    """
    Fits an ellipse to a cluster mask using the method of Buettner et al (2011).
    
    Parameters
    ----------
    ClusterMask : numpy.ndarray
        Mask of a DSM where the largest continuous cluster of voxels above a 
        threshold = 1, 0 otherwise.

    Returns
    -------
    params : dict
        Dictionary of the characteristics of the fitted ellipse:
        - 'CoM': The center of mass (x,y) of the ellipse.
        - 'a', 'b': Half the width and height of the ellipse, respectively.
        - 'theta': The angle (in rad) the ellipse is rotated.
    """
    #STEP1: Converthe numpy mask into openCV acceptable format and calculate it's outer contour
    imgray = np.array(ClusterMask*255).astype('uint8')
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours,hierarchy = cv.findContours(thresh,1,cv.CHAIN_APPROX_NONE)
    #because there is a chance the cluster could have a hole in the middle (that will have its own contour)
        #we need to ensure that if findContours returns more than 1 contour we take the largest one
    if len(contours)>1:
        print(type(contours))
        cnt = []
        for c in contours:
            if len(c)>len(cnt):
                cnt = c
    else:
        cnt = contours[0]
    #STEP2: Fit an ellipse to the contour
    ellipse = cv.fitEllipse(cnt)
    #STEP3: Return the fit parameters in the form x0,y0,a,b, and theta(in radians) for easy plotting
        #note: openCV returns 2a and 2b rather than a,b and theta in degrees rather than radians
    params = {'CoM':[ellipse[0][0],ellipse[0][1]],'a':ellipse[1][0]/2,'b':ellipse[1][1]/2,'theta':ellipse[2]*math.pi/180}
    return params
    
def ellipse_features(Parms,DSMShape):
    """
    Calculates the spatial dose metrics of an ellipse fit to a DSM as defined by
    Buettner et al (2011) and Moulton et al (YEAR).
    
    Parameters
    ----------
    Parms : dict
        Dictionary of the characteristics of an ellipse fitted to DSM.
    DSMShape : tuple of ints       
        The shape of the DSM array the ellipse was fit to. Required to calculate
        features related to the size of the DSM.

    Returns
    -------
    Features : dict
        Dictionary of the calculated spatial dose metrics:
        - 'area': The percent of the DSM's area covered by the ellipse.
        - 'eccent': The eccentricity of the ellipse (None if a or b = 0)
        - 'anglerad': The rotation of the ellipse, in radians
        - 'latproj': The percent of lateral span of the DSM covered by a projection
         of the ellipse's lateral axis onto the DSM's lateral axis (Buettner).
        - 'longproj': The percent of longitudinal span of the DSM covered by a 
        projection of the ellipse longitudinal axis onto the DSM's longitudinal
        axis(Buettner)
    """
    #STEP1: define a function to find points on the fitted ellipse
    xc,yc = Parms['CoM'][0],Parms['CoM'][1]
    a,b,theta = Parms['a'],Parms['b'],Parms['theta']
    coord = lambda t: [xc + a*np.cos(theta)*np.cos(t) - b*np.sin(theta)*np.sin(t), yc + a*np.sin(theta)*np.cos(t) + b*np.cos(theta)*np.sin(t)]
    #STEP2: Get the coordinates of the points at 0,pi/2,pi,3pi/2)
    axespoints = np.array([coord(0),coord(math.pi/2),coord(math.pi),coord(3*math.pi/2)])
    #STEP3: Calculate the extent of the ellipse axes along lat and long axes of the DSM
    Latlimits = (axespoints[:,0].min(axis=0),axespoints[:,0].max(axis=0))
    Longlimits = (axespoints[:,1].min(axis=0),axespoints[:,1].max(axis=0))
    LatExtent = (Latlimits[1]-Latlimits[0])*100/DSMShape[1]
    LongExtent = (Longlimits[1]-Longlimits[0])*100/DSMShape[0]
    #STEP4: Calculate parameters related to the ellipse equation
    if a==0 or b==0:
        eccent = None
    else:
        abratio = min(a/b,b/a)
        eccent = (1 - (abratio)**2)**0.5
    area = math.pi*a*b /(DSMShape[0]*DSMShape[1])
    #STEP: return a dictionary of features
    Features = {'area':area,'eccent':eccent,'anglerad':theta,'latproj':LatExtent,'lngproj':LongExtent}
    return Features

def cluster_features(ClusterMask):
    """
    Calculates the spatial dose metrics of a cluster mask of a DSM.
    
    Parameters
    ----------
    ClusterMask : numpy.ndarray
        Cluster mask of a dose surface map (DSM).

    Returns
    -------
    Features : dict
        Dictionary of the calculated spatial dose metrics:
        - 'area': The percent area of the DSM covered by the Cluster.
        - 'centroid': The center of mass (in array indices) of the Cluster.
        - 'centroidpcnt': The center of mass (in percent of lat/long dimensions) 
        of the Cluster.
        - 'latext': The percent of the lateral span of the DSM covered by the Cluster.
        - 'longext': Tthe percent of the longitudinal span of the DSM covered by 
        the Cluster.
        - 'bounds': The lateral and longitudinal index bounds of the cluster
    """
    #STEP1: get indices of all non-zero elemts of the mask
    locs = np.transpose(np.nonzero(ClusterMask))
    #STEP2: Find the min/max ind values in each direction
    Longlimits = (locs[:,0].min(axis=0),locs[:,0].max(axis=0))
    Latlimits = (locs[:,1].min(axis=0),locs[:,1].max(axis=0))
    #STEP3: Calculate extent-based metrics
    LatExtent = (Latlimits[1]-Latlimits[0]+1)*100/ClusterMask.shape[1]
    LongExtent = (Longlimits[1]-Longlimits[0]+1)*100/ClusterMask.shape[0]
    #STEP4: Calculate Moment based metrics
    M = moments(ClusterMask)
    Areapcnt = M[0,0]/ClusterMask.size *100
    centroid = (M[0,1]/M[0,0], M[1,0]/M[0,0]) #order is lat long
    CoMPcnt = (centroid[0]/ClusterMask.shape[1]*100,centroid[1]/ClusterMask.shape[0]*100)
    #STEP5: Return all features as a dictionary
    Features = {'area':Areapcnt,'centroid':centroid,'centroidpcnt':CoMPcnt,'latext':LatExtent,'longext':LongExtent,'bounds':(Latlimits,Longlimits)}
    return Features

def sample_dsh(DSM,dosepoints):
    """
    Calculates a dose surface histogram (DSH) from a DSM.

    Parameters
    ----------
    DSM : numpy.ndarray
        The DSM to calculate a DSH from.
    dosepoints : list of numpy.ndarray
        Series of dose values to calculate points on the DSH.

    Returns
    -------
    DSH : numpy.ndarray        
        Array of percent areas >= the values in the dosepoints list.
    """
    #STEP1: Initialize np.array for outputs
    DSH = np.zeros(len(dosepoints))
    area = DSM.size
    #STEP2: Iterate and create DSH
    for i in range(len(dosepoints)):
        DSH[i] = (DSM>=dosepoints[i]).sum()*100/area
    return DSH

####    -----------------------     ####
#   CODE FOR ADDITIONAL DSM PROCESSING     #
####    -----------------------     ####
def combine_dsms(DSMlist,kind='sum'):
    """
    Combines multiple DSMS together using the assumptions that:
    - All DSMs have the same inferior boarder.
    - All DSMs have the same number of samples per slice (row).
    - All DSMs use the same spacing between slices. This means either all DSMs 
    were created with the same number of slices with the 'nsteps' method, or they
    all used the same set distance with the 'dist' method. 

    Parameters
    ----------
    DSMlist : list of numpy.ndarrays 
        A list of the DSMs to combine.
    kind : str, optional   
        What type of combination of the DSMs to create. Valid inputs are 'sum',
        'difference', or 'average'). Defaults to 'sum'.

    Returns
    -------
    combo : numpy.ndarray
        Resulting DSM from the combination method specified.

    Notes
    -------
    If the dose-surface maps do not have the same longitudinal (column) size all
    input DSMs will be truncated to the height of the shortest one, making the 
    output DSM the same shape as the smallest DSM.
    """
    #STEP1: check the longitudinal size of all the DSMs and make an empty matrix the size of the shortest
        #At the same time check DSM shapes and throw exception if incompatible
    minlen = 1000
    ncol = np.shape(DSMlist[0])[-1]
    for DSM in DSMlist:
        minlen = min(minlen,len(DSM))
        if np.shape(DSM)[-1] != ncol:
            raise Exception('Only DSMs with the same number of columns can be combined with this method.')
    combo = np.zeros(shape=(minlen,len(DSMlist[0][0])))

    #STEP2: combine the DSMs
    if kind == 'difference':
        #check that only two DSMs were given, then subtract the second from the first
        if len(DSMlist) > 2:
            print('ISSUE! DSM differences can only be calculated for a pair of DSMs, not a larger list. Please adjust your code.')
            return None
        combo = DSMlist[0][0:minlen] - DSMlist[1][0:minlen]
    else:
        #sum everything together
        for DSM in DSMlist:
            combo = combo + DSM[0:minlen]
    if kind == 'average':
        combo = combo/len(DSMlist)
    #STEP3: return the combined DSMs
    return combo

def bed_calculation(DSM,aB,n):
    """
    Converts a DSM to its Biologically Effective Dose (BED) equivalent.

    Parameters
    ----------
    DSM : numpy.ndarray
        The dose-surface map to convert.
    aB : float
        The alpha-beta ratio to use for the conversion.
    n : int
        The number of fractions worth of dose represented in the DSM.

    Returns
    -------
    BED : numpy.ndarray
        The BED map of the original DSM.

    See Also
    --------
    EQD_gy : Converts a DSM to its Equivalent Dose (EQD) for a specified fraction
    size.

    Notes
    -------
    The BED conversion is performed voxel-wise according to:
    .. math:: BED = nd [1 + d/(\alpha/\beta)]

    where n is the number of fractions, d the voxel-wise dose per fraction, and 
    a/B the alpha-beta ratio.
    """
    #STEP1: copy the original DSM marix and divide by n to get a matrix of d values
    mtx = np.copy(DSM)/n
    #STEP2: divide the array of d values by a/B then add 1 to each element
    mtx = mtx/aB +1
    #STEP3: elementwise multiply the d array with the original matric (which is equal to dn)
    BED = mtx*DSM
    return BED

def eqd_gy(DSM,aB,n,newgy):
    """
    Converts a dose-surface map to its Equivalent Dose (EQD) for the dose per
    fraction specified.

    Parameters
    ----------
    DSM : numpy.ndarray
        The dose-surface map to convert.
    aB : float
        The alpha-beta ratio to use for the conversion.
    n : int
        The number of fractions worth of dose represented in the DSM.
    newgy : float
        The dose per fraction (in Gy) to convert the DSM to.

    Returns
    -------
    EQD : numpy.ndarray
        The EQD_x Gy map of the original DSM.

    Notes
    -------
    The EQD conversion is performed voxel-wise according to:
    .. math:: EQD_x = BED / [1 + x/(\alpha/\beta)] \\
                    = nd [1 + d/(\alpha/\beta)] / [1 + x/(\alpha/\beta)]

    where n is the number of fractions, d the voxel-wise dose per fraction, x the
    prescribed dose per fraction to convert to, and a/B the alpha-beta ratio.
    """
    #STEP1: Get the BED matrixx
    BED = bed_calculation(DSM,aB,n)
    #STEP2: Calculate the denominator of the EQD equation
    denom = 1 + (newgy/aB)
    #STEP3: Calculate the EQD_d Gy matrix
    EQD = BED/denom
    return EQD
