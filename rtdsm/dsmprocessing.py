import numpy as np
import math
from skimage.measure import label
from skimage.measure import moments
import cv2 as cv
from itertools import combinations, permutations, product
from math import factorial, pi

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
    params = {'CoM':[ellipse[0][0],ellipse[0][1]],'a':ellipse[1][0]/2,'b':ellipse[1][1]/2,'theta':ellipse[2]*pi/180}
    return params
    
def ellipse_features(Parms,DSMShape):
    """
    Calculates the spatial dose metrics of an ellipse fit to a DSM as defined by
    Buettner et al (2011) and Moulton et al (2017).
    
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
    axespoints = np.array([coord(0),coord(pi/2),coord(pi),coord(3*pi/2)])
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
    area = pi*a*b /(DSMShape[0]*DSMShape[1])
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
        - 'longext': The percent of the longitudinal span of the DSM covered by 
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

####    -----------------------     ####
#          MCP TEST FUNCTIONS          #
####    -----------------------     ####

def _init_MCP_inputs(data,testType,variation,n_permutations,randomSeed):
    """
    A function to check that all the provided input variables are in the correct format
    """
    #Check that the testType and variation parameters are permitted
    testType, allowed = testType.lower() , {'independent','paired'}    #make it lowercase
    if testType not in allowed:
        raise ValueError(f"`testType` is not an expected value. Allowed values are {allowed}.")
    variation, allowed = variation.lower() , {'two-sided','lesser','greater'}    #make it lowercase
    if variation not in allowed:
        raise ValueError(f"`variation` must be `two-sided`, `lesser`, or `greater`.")
    #Ensure n_permutations is an interger
    if isinstance(n_permutations, int) == False:
        raise ValueError(f"`n_permutations` must be an integer.")
    #Check randomSeed is either an int or None
    if randomSeed != None and isinstance(n_permutations, int) == False:
        raise ValueError(f"`randomSeed` must be either an integer or `None`.")

    #Check that data is correctly formatted
    message = "`data` must be either a tuple of two 2D numpy arrays or a single 2D numpy array."
    #Start by doing the check for 1 sample test scenario
    if isinstance(data,np.ndarray):
        print('Single numpy array detected for `data`.')
        if data.ndim != 2:
            raise ValueError(f"A single numpy array with dimentions other than 2 was detected for `data`.\nIf performing a one sample test please ensure `data` is a 2D numpy array, otherwise provide a tuple of 2D numpy arrays for a two sample test.")
    #Now check for 2 sample test scenarios
    elif isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(message)
        if isinstance(data[0],np.ndarray) !=True or isinstance(data[1],np.ndarray) != True: #ensurenumpy arrays
            raise ValueError(message)
        if data[0].ndim !=2 or data[1].ndim !=2:    #ensure both 2D
            raise ValueError(message)
        if len(data[0])<2 or len(data[1])<2:
            raise ValueError("Each dataset must contain at least two DSMs/observations.")
        if testType == 'paired' and len(data[0]) != len(data[1]):
            raise ValueError("Each dataset must contain the same number of observations to perform a paired test.")
    else:
        raise ValueError(message)
    return data,testType,variation,n_permutations,randomSeed

#-------Permutation generators------
def paired_permutation_generator(n_datapairs,n_permutations,randomGenerator=None):
    #STEP1: Determine the number of sample pairs and total possible permutations
    n_samples = n_datapairs
    max_perms = 2**n_samples
    #STEP2: Based on the requested # of iterations, define a permutation generator
    if max_perms <= n_permutations:
        #Explicitly calculate all possible permutations if an exact test
        exact = True
        n_permutations = max_perms
        #generate all n possible combinations of 0 and 1 lables for the dataset
        perm_generator = product([0, 1], repeat=n_samples)
    else:
        #Generate n random possible permutations
        exact = False
        perm_generator = (randomGenerator.choice([0,1],n_samples) for i in range(n_permutations))
    return perm_generator, int(n_permutations),exact

def indept_permutation_generator(n_obsv,n_permutations,randomGenerator=None):
    #STEP1: Determine the number of samples per group and total possible permutations
    n_obsv_total = np.sum(n_obsv)
    max_perms =  factorial(n_obsv_total)/(factorial(n_obsv[0]) * factorial(n_obsv[1]) )   #N_total choose n in group 1
    #STEP2: Based on the requested # of permutations, define a permutation generator
    if max_perms <= n_permutations:
        #Explicitly calculate all possible data permutations for group1
        exact = True
        n_permutations = max_perms
        perm_generator = combinations(range(n_obsv_total),n_obsv[0]) #generate all possible sets for group 1
    else:
        #Calculate n random possible permutations of index labels for group1
        exact = False
        perm_generator = (randomGenerator.permutation(n_obsv_total)[:n_obsv[0]] for i in range(n_permutations))
    return perm_generator, int(n_permutations),exact

#-------compute test results------
def calc_pvalue_Tstar(iter_dist,obs,exact):
    """
    A function to calculate the p-value of the test and the statistic values for 
    various p-value thresholds.
    """
    #Define the adjustment factor and tolerance for detecting a distinct value
    adjust = 0 if exact else 1
    tol = np.maximum(1e-14, np.abs(1e-14*obs) )
    #Calculate the p-value
    pvalue = ( len(np.where(iter_dist >= (obs+tol) )[0]) + adjust)/(len(iter_dist) + adjust)
    #Identifiy the statistic values for several percentiles
    thresholds = np.array([[0.5,0],[0.1,0],[0.05,0],[0.01,0]])
    for level in thresholds:
        level[1] = np.percentile(iter_dist,100*(1-level[0]))
    return pvalue, thresholds

def MCP_test(data,testType='independent',variation='two-sided',n_permutations=1000,randomSeed=None):
    """
    Performs a multiple comparisons permutation test to determine if a set of
    dose-surface maps are statistically different from another set or baseline
    according to the methodology of Chen et al (doi.org/10.1186/1748-717X-8-293).

    Parameters
    ----------
    data : tuple or numpy.ndarray
        Contains the DSM data one or two populations (tuple of numpy arrays).
        Each population array must be structured so that each row contains
        the pixel values of a single DSM flattend from its usual 2D form and 
        each column corresponds to a pixel in the DSM.
    testType : {â€˜independentâ€™, â€˜pairedâ€™}, optional
        The type of test (and therefore the type of permutations) to perform.
        - 'independent' (default): Observations can be randomly assigned to
        either population. Populations do not need to contain the same number of
        observations. Use this type for indepent sample tests (eg. indept. t-test)
        - 'paired': Observerations in both populations are paired with the same
        indexed observation from the other population. Use this type for paired
        tests (eg. paired t-test) or for single sample tests.
    variation :  {â€˜two-sidedâ€™,â€˜lesserâ€™,â€˜greaterâ€™}, optional
        The non-null hypothesis to be tested. The p-value is defined as follows
        for each variation:
        -'two-sided' (default): The percent of the null distribution that has 
        greater absolute values of the test statistic than the observed sample.
        -'lesser': The percent of the null distribution that is less than or
        equal to the test statistic of the observed sample.
        -'greater': The percent of the null distribution that is greater than or
        equal to the test statistic of the observed sample.
    n_permutations : int, default=1000
        Number of random permutations to use to approximate the null distribution.
        If higher than the number of possible distinct permutations the explicit 
        null distribution will be calculated.
    randomSeed : int or None (default)
        Seed value to use to create a numpy.random.RandomState instance.

    Returns
    -------
    results : dict
        A dictionary of the various results of the test:
        - 'pvalue': The p-value result of the test.
        - 'obs_statmap_1D': A map of the test statistic values for the observed 
        sample, given as a 1D array. It can be returned toa 2D array with numpy
        reshape()
        - 'percentile_thresholds': an array of alpha values and corresponding
        test statistic values for the given test. This information can be used
        with 'obs_statmap_1D' to identify pixels that differ between test
        populations to a given level of significance.
        - 'precision': The minimum change in p-value detectable for the number of
        permutations used for the test. This output is predominantly provided to 
        contextulize the p-value result for tests with small smaple sizes. 

    Notes
    ----------
    As defined by Chen et al, the test statistic used to produce the null 
    distribution for this test is T_max, which is the maximum value of the
    statistic T for all pixels in a given permutation of data. T is defined as
    the mean difference in a pixel's intensity between groups 1 and 2, divided by
    the standard deviation of this mean difference value across all permulations:
    .. math:: T = ( \mu_{g1} - \mu_{g2} ) / \sigma

    The selection of T_max will differ based on the number of sides used for the
    test. If using 'greater' the maximum value of T will be used, the minimum 
    value used for 'lesser', and the maximum absolute value for 'two-sided'.

    When performing a one-sided test ('lesser' or 'greater') the user must take
    into account the way the statistic is calculated to ensure they are providing
    the data in the right order to correctly evaluate the test.
    
    See Also
    ----------
    format_MCP_data : formats DSM data into the structure used by this function.
    """
    #STEP0: Check the inputs are as expected
    data,testType,variation,n_permutations,randomSeed = _init_MCP_inputs(data,testType,variation,n_permutations,randomSeed)
    #STEP1: initialize the random number generator and check inputs
    randomGenerator = np.random.RandomState(randomSeed)

    #STEP2: initialize the data, permutation generator, and statistic based on the testType 
    if testType == 'independent':
        #2a: count the samples per data group and concatenate together
        n_samples = np.array([len(data[0]), len(data[1])])
        data = np.concatenate(data,axis=0)
        #2b: create the appropriate permutation generator for the sample size
        perm_generator, n_permutations,exact = indept_permutation_generator(n_samples,n_permutations,randomGenerator)
        #2c: define a function to calculate the mean difference of a given permutation
        def calc_meandiff(data,g1set):
            # defie indices of samples that belong to groups 1 & 2
            g1set = np.array(g1set)
            g2set = np.delete(np.arange(len(data)), g1set)
            # calculate the mean difference
            meandiff = np.mean(data[g1set],axis=0) - np.mean(data[g2set],axis=0)
            return meandiff
        #2d: calculate the statistic for the observed sample
        d_obs = calc_meandiff(data,np.arange(n_samples[0]) )
    elif testType == 'paired':
        #2a: ensure datasets are the same size, then create difference array
        if type(data) != tuple:
            print('Only one group found. Will proceed with one sample test...')
            n_samples = len(data)
        else: 
            n_samples = len(data[0])
            data = data[0] - data[1]
        #2b:create the appropriate permutation generator for the sample size
        perm_generator, n_permutations,exact = paired_permutation_generator(n_samples,n_permutations,randomGenerator)
        #2c: define a function to calculate the mean difference of a given permutation
        def calc_meandiff(data,signs):
            # create an array of +1/-1 values
            signs = np.power(np.ones(n_samples)*(-1), signs)
            #print(signs)
            # apply to the data
            data = data * signs[:, np.newaxis]
            #calculate the mean of the dataset
            meandiff = np.mean(data,axis=0) 
            return meandiff
        #2d: calculate the statistic for the observed sample
        d_obs = calc_meandiff(data,np.zeros(n_samples) )

    #STEP3: Begin iterating through the permutations
    IterData = np.empty((n_permutations,len(data[0,:]))) #array to store staticstics from permutations
    i = 0
    for indices in perm_generator:
        IterData[i] = calc_meandiff(data,indices)
        i +=1

    #STEP4: Calculate the st.dev of metric values across the permutations and use to normalize values to our statistic T
    std_k = np.std(IterData,axis=0)
    T_iter = IterData/std_k     #the statistic for each pixel for each iteration
    T_obs = d_obs/ std_k        #the statistic for each pixel in the observed sample

    #STEP5: Select the maximum value of T for all pixels in each permuation and based on the type of hypothesis
    if variation == 'two-sided':
        T_max = np.amax(np.abs(T_iter),axis=1)    #identify the most extreme value of T from each iteration
        T_max_obs = np.amax(np.abs(T_obs))        #do the same for the observed sample
        #Calculate the p-value of the observed sample and the statisc values corresponding to various percentiles
        pvalue, percentile_thresholds = calc_pvalue_Tstar(T_max,T_max_obs,exact)
    elif variation == 'greater':
        T_max = np.amax(T_iter,axis=1)    
        T_max_obs = np.amax(T_obs)        
        pvalue, percentile_thresholds = calc_pvalue_Tstar(T_max,T_max_obs,exact)
    elif variation == 'lesser':
        T_max = np.amax((-1)*T_iter,axis=1)    
        T_max_obs = np.amax((-1)*T_obs)      
        pvalue, percentile_thresholds = calc_pvalue_Tstar(T_max,T_max_obs,exact)
        percentile_thresholds[:,1] = percentile_thresholds[:,1]*(-1)    #adjust to be negative values

    return {'pvalue': pvalue, 'obs_statmap_1D':T_obs, 'percentile_thresholds': percentile_thresholds, 'precision': round(1/len(T_max),4) }

def format_MCP_data(listOfDSMs,truncateIndex=None):
    """
    Function to take a set of 2D DSMs and format them into a single 2D array for
    use in the MCP_test() function.

    Parameters
    ----------
    listOfDSMs : list of numpy.ndarrays
        A list of DSMs for a given population. All DSMs must have the same x 
        (AKA angular) resolution to be valid.
    truncateIndex : int or None (default)
        The row (AKA y or slice index) at which all DSMs are to be truncated to
        to ensure all DSMs have the same shape. By default, the function will use
        the height of the shortest DSM in the list.

    Returns
    -------
    data_arr : numpy.ndarray
        A 2D numpy array in which each row contains the flattened pixel 
        information of a given DSM.
    shape : tuple
        The shape of the original DSMs in data_arr before they were flattened. 
        This information can be used to restore dsm data in data_arr to its
        original shape with numpy.reshape().
    
    See Also
    ----------
    MCP_test : The statistical test that uses data_arr as an input.
    """
    #STEP1: Determine the standard DSM dimensions to use
    xdim = None
    for DSM in listOfDSMs:
        i_shape = DSM.shape     #get the shape of the DSM
        if xdim == None:
            xdim = i_shape[1]
        elif i_shape[1] != xdim:   #if the x (aka angular dimension not consistent raise an error
            raise ValueError("Inconsistent DSM shape in angular (x) dimension between DSMs.")
        #identify the shortest DSM length (ithe slicing dimension)
        truncateIndex = min(value for value in [truncateIndex,i_shape[0]] if value is not None) 
    #STEP2: Create an array to hold the data
    data_arr = np.zeros((len(listOfDSMs), xdim*truncateIndex))
    #STEP3: Fill the array with the flattened DSM data
    for i in range(len(listOfDSMs)):
        data_arr[i,:] = listOfDSMs[i][:truncateIndex,:].flatten()
    #STEP4: Return the prepped dataarray and the original shape for reference
    return data_arr, (truncateIndex,xdim)
