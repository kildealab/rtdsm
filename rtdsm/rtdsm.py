import numpy as np
import pyvista as pv
import math
import rtdsm.helpers as helpers
from rtdsm.meshandspline import get_spline_tangents
# packages required to sort slice sets
from operator import itemgetter
from itertools import groupby
# packages required for dose sampling
from scipy.interpolate import RegularGridInterpolator

####    -----------------------     ####
#      GENERAL SLICING FUNCTIONS       #
####    -----------------------     ####
def interpolate_missed_points(samppoints,failvects,centroid):
    """
    Fills in points missed by ray casting in the sample_uniangular method by
    interpolating along a spline loop created from successfully found points.
    Used as an optional subfunction within sample_uniangular.

    Parameters
    ----------
    samppoints : numpy.ndarray
        Array of point coordinates from equiangular ray casting. Failed points 
        given as np.nan.
    failvects : numpy.ndarray      
        Array of ray casting vectors that failed to detect a point on the ROI 
        mesh.
    centroid :  numpy.ndarray       
        Coordinate where ray casting originated from.

    Returns
    -------
    samppoints : numpy.ndarray     
        Updated array of point coordinates with failed points filled with 
        interpolated fix.

    Notes
    -------
    Because this method depends on successfully ray-casted points to perform the 
    fix, accuracy decreases the more points that failed. This method should only 
    be needed to be called for a small number of points (1-4) on one or two slices 
    total. If it is called more frequently it is recommended to check the ROI mesh 
    quality, as it tends to be an indicator of a holey or otherwise poorer 
    quality mesh.

    """
    #STEP1: Create a densely sampled spline with the successfully sampled points    
    successes = samppoints[~np.isnan(samppoints).any(axis=1)]
    failinds = np.argwhere(np.isnan(samppoints[:,0])).ravel()
    points = np.zeros((len(samppoints)-len(failvects)+1,3)) #make placeholder array 1 point longer than successful points
    points[:-1,:],points[-1,:]= successes, successes[0,:]   #fill with ordered list and repeat first point at end to make loop
    intspline = pv.Spline(points,720)   #using 720 points to try to make sure one point per degree at least
    ringpoints = intspline.points
    #STEP2: for each vector that didnt successfully ray trace, find the closest point on the spline
    print('Calculating points missed by ray tracing...')
    Dtoline = lambda X : np.linalg.norm(np.cross((X - centroid),(X - (centroid+vect))) ) / np.linalg.norm((vect)) #distance between point and line in 3D coordinate space
    ii = 0
    for vect in failvects:
        #print(failinds[ii])
        pind = failinds[ii]
        # #get the closest succesful ray tracing point hat came before the failed point
        while pind <= failinds[ii]:
            pind -= 1
            prevspot = samppoints[pind]
            if math.isnan(prevspot[0]) == False:
                break
        dists = np.apply_along_axis(Dtoline, 1, ringpoints) #calculate distances for all points on the spline
        #get the 3 closest dists to the vecorline, then check to make sure you grab one along correct direction of ray (not 180 of intended direction)
        #Sort the list and get the 8 points on the contour that come closest to the sampling ray
        idx = np.argpartition(dists, 8)
        closest = ringpoints[idx[:8],:]
        #Next, calculate how far these points are from the previous point on the contour ring (eg. when looking for point at 30deg, compare to point at 27deg)
        vdists = np.linalg.norm(closest - prevspot, axis=1)
        #find the first element in the array with a low distance value
        ibest = np.argmax(vdists<np.mean(vdists))
        samppoints[failinds[ii],:] = closest[ibest]
        ii += 1
    return samppoints

def sample_uniangular(SurfaceObject,normal,centroid,nsamples,fixfailures=True):
    """
    Finds N equiangularly spaced points on the contour for a given sampling plane
    using ray casting.
    
    Parameters
    ----------
    SurfaceObject : pyvista.PolyData 
        Surface mesh of the ROI.
    normal : numpy.ndarray
        The unit normal vector (x,y,z) of the sampling plane.
    centroid : numpy.ndarray       
        Proposed point from which ray casting should occur (will be adjusted if
            needed).
    nsamples : int      
        Number of points to ray cast for.
    fixfailures : bool,optional    
        If True, will use an interpolation fix to fill in points missed by ray
        casting. If False, will return np.nan for coordinates of missing points.
        Defaults to True.

    Returns
    -------
    samppoints : numpy.ndarray
        Array of point coordinates on the mesh surface in the specified plane
        (M x 3).
    failvects : list
        List of ray casting vectors that failed to intersect the ROI mesh.
    newcent : numpy.ndarray
        Coordinate of the point where all ray casting originated from (approx
        slice centroid).
    
    Notes
    -------
    By design, the first and last point of the output array 'samppoints' are the 
    point directly posterior to the ray tracing centroid (assuming a HFS patient 
    orientation). This is so that the slice is already effectively 'cut open' at 
    the structure's posterior side, as is frequently convention. Changing the cut
    point can be achieved in post-processing by rolling the slice arrays to the 
    appropriate angle.
    """
    #STEP1: create a list of the angles to sample at, then define the sampling vectors around a unit circle in the slicing plane
    sampangs = np.linspace(0,2*math.pi,nsamples)
    basis = helpers.orthonormal_basis(normal)
    rayvects = basis[0]*np.ones((len(sampangs),3))*np.sin(sampangs[:, np.newaxis]) + basis[1]*np.ones((len(sampangs),3))*np.cos(sampangs[:, np.newaxis])
    rayvects = 250*rayvects
    #STEP2: define the centroid point we will perform the ray tracing from
    newcent,openTF = helpers.adjust_centroid(centroid,basis,SurfaceObject)
    centroid = newcent
    #STEP3: begin iterating through the list and performing ray tracking
    failcount = 0
    failvects = []
    samppoints = np.full_like(rayvects,np.nan)
    for i in range(len(rayvects)):
        #project the sampling vector onto the slice plane
        inter,ind = SurfaceObject.ray_trace(centroid,centroid+rayvects[i],first_point=True) #give start and end of ray, and only return first intersection
        if len(inter) == 0:
            failcount += 1
            failvects.append(rayvects[i])
            continue
        samppoints[i,:] = inter     #add to the list fo samp points (failed intersections will appear as nan)
    #STEP4: if ray tracing failed (tends to happen in less densely sapled regions of the mesh), run alternate method to get points
    if failcount > 0 and fixfailures==True:
        print('missed ',failcount,' of ',nsamples,'points for slice at',centroid,'. \nRunning interpolation fix now...')
        samppoints = interpolate_missed_points(samppoints,failvects,centroid)
    return samppoints, failvects, newcent

####    -----------------------     ####
#       PLANAR SLICING FUNCTIONS       #
####    -----------------------     ####
def sample_planar_slices(SurfaceObject,CenterLine,SamplingPoints,SampPerSlice,Normal=[0,0,1]):
    """
    Defines slices of an ROI mesh perpendicular to a single shared axis.

    Parameters
    ----------
    SurfaceObject : pyvista.PolyData
        The surface mesh of the ROI.
    CenterLine : numpy.ndarray or pyvista.PolyData
        An array or spline containing points at the center of each slice of the ROI.
    SamplingPoints : 
        Array of the indices of points in CenterLine where sampling should occur. 
    SampPerSlice : int 
        Number of points to define around each slice.
    Normal : list or numpy.ndarray, optional
        The normal vector to use for all sampling planes. Default value [0,0,1].

    Returns
    -------      
    Slices : 
        A dictionary housing all the data for each slice. Each slice's data is 
        stored under a key corresponding to that slice's index along the 
        centerline (eg. 345) and contains the following:
        - “PlnNorm”: The unit normal vector of the plane used to create the slice.
        - “Org”: The point from which ray casting to define the slice originated 
        from.
        -”Slice”: A pyvista PolyData object of the slice itself, formatted as a 
        disk. Contains the points around the circumference of the slice used to 
        create a DSM, plus the point “Org” to facilitate the creation of a 
        triangulated mesh. “Org” is always the first point in the mesh’s point 
        data array.
    """
    #STEP1: Declare dictionary to store slice data
    Slicedict = {}
    Issuelist = []
    #if a pyvista spline was provided instead of a numpy array extract the point array
    if type(CenterLine)== pv.core.pointset.PolyData:
        CenterLine = CenterLine.points
    #STEP2: Step through the sampling points list and begin taking slices
    for p in SamplingPoints:
        pslice, fails, NC = sample_uniangular(SurfaceObject,Normal,CenterLine[p],SampPerSlice,fixfailures=True)
        slicemesh = helpers.make_slice_mesh(NC, pslice)
        if len(pslice) == 0:
            print('ERROR OCCURED AT SLICE',p,': pyvista slice function failed to find contour points for the given origin and tangent. Check sampling location relative to mesh orientation.')
            Issuelist.append(p)
        #add the slice and point data to the dictionary
        Slicedict[p] = {'Slice':slicemesh,'PlnNorm':Normal,'Org':NC}
    #finish and return the slice dictionary
    Slicedict['IssueSlices'] = Issuelist
    return Slicedict

####    -----------------------     ####
#     NON-PlANAR SLICING FUNCTIONS     #
####    -----------------------     ####
def define_control_points(gradient):
    """
    Given the gradient of a spline, returns the indexes of stationary points 
    (points where one of the components of the gradient (x,y,z) is zero).

    Parameters
    ----------
    gradient : numpy.ndarray
        Array of gradient vectors from a parent spline. Can be calculated with
        the pyvista method compute_derivative() (shape M x 3). 
    
    Returns
    -------
    ControlPts : numpy.ndarray    
        Indices of stationary points of the parent spline object (shape M x 3).
    """
    #STEP1: assess the sign of all elements of the gradient 
    GradSigns = np.sign(gradient)
    #STEP2: calculate where sign changes happen
    SignChange =  ((np.roll(GradSigns, 1,axis=0) - GradSigns) != 0).astype(int)
    #STEP3: adjust and account for any funny business at the first point or points with a gradient element=0
    WhereChange = np.vstack((np.where(SignChange == 1)[0], np.where(SignChange == 1)[1] )).T    #indexes where sign changes
    SignZero = np.vstack((np.where(GradSigns == 0)[0], np.where(GradSigns == 0)[1] )).T         #indexes where sign is '0'
    #STEP4: Check if a sign change was detected at any of the zero values, and if so adjust to remove a neighbouring control points
    PointMatch = (WhereChange[:, None] == SignZero).all(-1).any(-1) #see if any of the rows match between arrays
    issue = np.where(PointMatch==True)[0]
    #STEP4a: loop through and adjust the list of stationary points based on what occured
    remove = []
    for p in issue:
        ind = WhereChange[p]    #get the index in the gradient array the match happened at
        #check the signs of the two neighbouring points
        if ind[0]+1 == len(GradSigns):
            pass #if something is detected for the last slice it doesn<t matter because it and the first slice are controls by default
        elif GradSigns[ind[0]-1,ind[1]] * GradSigns[(ind[0]+1),ind[1]] == -1:
            remove.append((p+1))  #a switch of signs occured, make a note to remove n+1's record in WhereChange
        elif GradSigns[ind[0]-1,ind[1]] * GradSigns[(ind[0]+1),ind[1]] == 1:
            remove.append(p)    #touched 0 but never crossed, make note to remove both points n and n+1
            remove.append((p+1))  #this is effectively saying we never had a sign change
        else:
            print('multiple points with gradient of 0 beginning at index',ind,'and ending at index',np.argmax(GradSigns[:ind[0],ind[1]]!=0 ))
            print('examine your data and determine the appropriate course of action')   #I dont expect this to happen much but I<m not sure if theres a one case fits all method
    WhereChange = np.delete(WhereChange,remove,axis=0)
    #STEP5: Return a list of all the indexes of all the points we'll use as control points, adding the first and last slices if not already included
    ControlPts = list(np.unique(WhereChange[:,0]))
    if ControlPts[0]!=0:
        ControlPts.insert(0,0)  #add first point if not included
    if ControlPts[-1]!=len(GradSigns):
        ControlPts.append(len(GradSigns)-1) #add last point if not included
    return ControlPts

def witztum_correction(CollisionSets,SampPoints,Slices,SurfaceObject,CenterLine):
    """
    Corrects overlapping slices of the ROI mesh using the approach of Witztum
    et al (2016) 

    Parameters
    ----------
    CollisionSets : list
        A list of sets of adjacent slices flagged as colliding with other slices.
    SampPoints : numpy.ndarray or list     
        Array of the indices of points along the centerline where sampling took
        place, as well as the indices of any additional control slices.
    Slices : dict 
        A python dictionary of the slice data of all non-colliding slices.
        Each slice's data is stored under a key corresponding to that slice's
        index along the centerline (eg. 345) and contains the following:
        - “PlnNorm”: The unit normal vector of the plane used to create the slice.
        - “Org”: The point from which ray casting to define the slice originated 
        from.
        -”Slice”: A pyvista PolyData object of the slice itself, formatted as a 
        disk. Contains the points around the circumference of the slice used to 
        create a DSM, plus the point “Org” to facilitate the creation of a 
        triangulated mesh. “Org” is always the first point in the mesh’s point 
        data array.

    SurfaceObject : pyvista.PolyData
        The surface meshh of the ROI being sliced.
    centerline : pyvista.PolyData     
        A spline of a path through the ROI sampling is to take place along.

    Returns
    -------         
    Slices : dict
        An updated dictionary of the slice data, including corrected
        previously colliding ones. 
    """
    #STEP1: Begin a loop going through all the sets of neighbouring collisions
    for colset in CollisionSets:
        print(colset,'- - - - -')
        #STEP2: define the slices that sandwich the set of colliding slices (we call them crusts here)
        bcrust,tcrust = max(0,colset[0]-1), min(len(SampPoints)-1,colset[-1]+1) #the max min checks ensure we don't go outside the range of possible slices
        #STEP3: Retrive points on both sandwicing slices taken from the same angles and use to draw vectors from one to the other
        sampres = math.floor((len(Slices[0]['Slice'].points)-1)/3)*np.array([0,1,2])+1 #gives 3 evenely spaced point indices (add one to ignore point 0 which is the centerpoint)
        bLocks = Slices[SampPoints[bcrust]]['Slice'].points[sampres]
        tLocks = Slices[SampPoints[tcrust]]['Slice'].points[sampres]
        vects = tLocks - bLocks
        #STEP4: set variables needed to define the corrected slices
        nfix = tcrust-bcrust-1  #how many slices we need to fix
        fixus = np.arange(bcrust+1,tcrust)  #define indices of the slices to be fixed
        #STEP5: loop through the slices that need fixing and define their new plane normals and centroids
        for n in range(nfix):
            slicedict = {}
            nLocks = bLocks + vects*((n+1)/(nfix+1))     #eg. get points 1/3 of the way along the paths from bottom to top crusts
            norm = np.cross((nLocks[2]-nLocks[0]),(nLocks[1]-nLocks[0]))  #normal of plane is crossproduct of two vectors in the plane     
            #calculate an approximate centroid in the plane for the new slice (will be improved on during uniangular sampling)
            newcent = nLocks.mean(axis=0)
            #save the plane normal andcentroidto the Slices dictionary
            slicedict['PlnNorm'] = norm/np.linalg.norm(norm)
            #sample the mesh and add the points to the dictionary too (keeping the centroid+points format)
            spoints, fails,NC = sample_uniangular(SurfaceObject,slicedict['PlnNorm'],newcent,len(Slices[0]['Slice'].points)-1,fixfailures=True)
            slicepoints = helpers.make_slice_mesh(NC, spoints)
            slicedict['Slice'] = slicepoints
            slicedict['Org'] = NC
            Slices[SampPoints[fixus[n]]] = slicedict
    return Slices

def fix_control_slice(CollisionSets,SampPoints,Slices,SurfaceObject):
    """
    Corrects a control slice that intersects another using a modified Witztum
    approach.
    
    Parameters
    ----------
    CollisionSets : list
        A list of sets of adjacent slices flagged as colliding with other slices.
    SampPoints : numpy.ndarray or list     
        Array of the indices of points along the centerline where sampling took 
        place, as well as the indices of any additional control slices.
    Slices : dict 
        A python dictionary of the slice data of all control slices. Each slice's
        data is stored under a key corresponding to that slice's index along the 
        centerline (eg. 345) and contains the following:
        - “PlnNorm”: The unit normal vector of the plane used to create the slice.
        - “Org”: The point from which ray casting to define the slice originated 
        from.
        -”Slice”: A pyvista PolyData object of the slice itself, formatted as a 
        disk. Contains the points around the circumference of the slice used to 
        create a DSM, plus the point “Org” to facilitate the creation of a 
        triangulated mesh. “Org” is always the first point in the mesh’s point 
        data array.
    SurfaceObject : pyvista.PolyData
        The surface meshh of the ROI being sliced.
    
    Returns
    -------
    Slices : dict     
        Updated version of the Slices dictionary with overlaping slices fixed.
    """
    #STEP 1: For each set of problematic slices, find the sandwiching slices
    for colset in CollisionSets:
        print(colset,'- - - - -')
        bcrust,tcrust = max(0,colset[0]-1), min(len(SampPoints)-1,colset[-1]+1) #the max min checks ensure we don't go outside the range of possible slices
        #STEP2: Get points on both sandwiching slices at common angles and use to draw lines between them
        sampres = math.floor((len(Slices[0]['Slice'].points)-1)/3)*np.array([0,1,2])+1 #gives 3 evenely spaced point indices (add one to ignore point 0 which is the centerpoint)
        bLocks = Slices[SampPoints[bcrust]]['Slice'].points[sampres] #bottom slice
        tLocks = Slices[SampPoints[tcrust]]['Slice'].points[sampres] #top slice
        vects = tLocks - bLocks
        #STEP3: create a line between the sandwiching slices' centroids we'll use to determine where to define the new slice plane(s)
        centerpath = Slices[SampPoints[tcrust]]['Org']-Slices[SampPoints[bcrust]]['Org']
        fixus = np.arange(bcrust+1,tcrust) #makes array of slices we need to fix
        for prob in fixus:
            slicedict = {}
            #STEP4: determine how far along the centerpath the point closest to the original slice's centroid is
            orgcentroid = Slices[SampPoints[prob]]['Org']
            deltaP = orgcentroid - Slices[SampPoints[bcrust]]['Org']
            t = np.dot(centerpath,deltaP)/np.dot(centerpath,centerpath) #calculates point along the centerpath vector the original centroid projects onto
            #STEP5: use this distance t to define points along the outerlines to make a new slicing plane
            nLocks = bLocks + vects*t
            norm = np.cross((nLocks[2]-nLocks[0]),(nLocks[1]-nLocks[0]))#((nLocks[1]-nLocks[0]),(nLocks[2]-nLocks[0]))
            #save the plane normal andcentroidto the Slices dictionary
            slicedict['PlnNorm'] = norm/np.linalg.norm(norm)
            #sample the mesh and add the points to the dictionary too (keeping the centroid+points format)
            spoints, fails, NC = sample_uniangular(SurfaceObject,slicedict['PlnNorm'],orgcentroid,len(Slices[0]['Slice'].points)-1,fixfailures=True)
            slicedict['Org'] = NC #update the point used as the centroid for ray tracing
            slicepoints = helpers.make_slice_mesh(NC, spoints)#(orgcentroid, spoints)
            slicedict['Slice'] = slicepoints
            Slices[SampPoints[prob]] = slicedict
    return Slices

def check_control_slices(Slices,SurfaceObject,zeroind=0):
    """
    Checks all pairs of control slices for possible intersections.
    
    Parameters
    ----------
    Slices : dict 
        A python dictionary of the slice data of all control slices.
        Each slice's data is stored under a key corresponding to that slice's
        index along the centerline (eg. 345) and contains the following:
        - “PlnNorm”: The unit normal vector of the plane used to create the slice.
        - “Org”: The point from which ray casting to define the slice originated 
        from.
        -”Slice”: A pyvista PolyData object of the slice itself, formatted as a 
        disk. Contains the points around the circumference of the slice used to 
        create a DSM, plus the point “Org” to facilitate the creation of a 
        triangulated mesh. “Org” is always the first point in the mesh’s point 
        data array.
    SurfaceObject : pyvista.PolyData
        The surface meshh of the ROI being sliced.
    zeroind : int, optional  
        The component (0=X,1=Y,2=Z) to set to 0 when solving for the line.
        By default, set X=0.

    Returns
    -------
    Slices : dict
        Updated version of the Slices dictionary with overlapping sliced corrected.
    """
    #STEP1: Define the list of control slices and go through all possible pairings
    keylist = list(Slices.keys())
    problemslices = []
    todelete =  []
    for i in range(len(keylist)):
        slice_i = Slices[keylist[i]]['Slice']
        for j in range(i+1,len(keylist)):
            print('checking',i,j,'---------')
            log = (i,j)
            slice_j = Slices[keylist[j]]['Slice']
            #STEP2: check if the normals are the same, if so we knwo they're parlell and wont collide
            if np.array_equal(Slices[keylist[j]]['PlnNorm'],Slices[keylist[i]]['PlnNorm'])==True:
                continue   
            #STEP3: define the intersection line between the planes of the two slcies
            v,p = helpers.find_plane_intersection(Slices[keylist[j]]['PlnNorm'],slice_j.points[0],Slices[keylist[i]]['PlnNorm'],slice_i.points[0],zeroind)
            maxvdim = np.argmax(abs(v))
            p = helpers.select_intersection_point(v,p,slice_i) #adjust the point to be as close to slice_i as possible
            ppdist = pv.PolyData(p).compute_implicit_distance(slice_j ) #calcualte the distance from p to slice j
            #STEP4:If a ray trace collides for both slices or ppdist is under a micron, they overlap
            j1,j2 = slice_j.ray_trace(p,p+v*100) , slice_j.ray_trace(p,p-v*100)
            i1,i2 = slice_i.ray_trace(p,p+v*100) , slice_i.ray_trace(p,p-v*100)
            if  (( len(i1[0]) + len(i2[0]) )>0 and ( len(j1[0]) + len(j2[0]) )>0) or abs(ppdist['implicit_distance'][0]) < 0.001:
                print('collision between 2 control slices',keylist[i],keylist[j])
                #STEP5: check the distance between the slices and their circularity
                dist = np.linalg.norm(slice_i.points[0]-slice_j.points[0])
                areas = np.array([slice_i.area,slice_j.area])
                perims = np.array([slice_perimeter(slice_i),slice_perimeter(slice_j)])
                roundness = (areas*4*np.pi) / (perims**2)
                print('circularity:',roundness)
                if dist <= 3:
                    print('Overlap due to close proximity (<3mm), removing (less circular) one...')
                    if (0 in log):
                        takeind = log.index(0)  #if it includes the first slice flag the other to be fixed
                        todelete.append(log[takeind-1])                 
                    elif ((len(keylist)-1) in log):
                        takeind = log.index(len(keylist)-1) #if it includes the last slice flag the other to be fixed
                        todelete.append(log[takeind-1])
                    else:
                        todelete.append(log[np.argmin(roundness)])
                else:
                    print('Overlap due to crooked (less circular) slice, will adjust its alignment...')
                    if (0 in log):
                        takeind = log.index(0) #if it includes the first slice flag the other to be fixed
                        problemslices.append(log[takeind-1])                    
                    elif ((len(keylist)-1) in log):
                        takeind = log.index(len(keylist)-1) #if it includes the last slice flag the other to be fixed
                        problemslices.append(log[takeind-1])
                    else:
                        problemslices.append(log[np.argmin(roundness)])
    #STEP 6: correct the crooked slices
    print('flagged for removal:',todelete,'flagged for correction:',problemslices)
    #remove duplicates and slices to be delete from the correction code
    problemslices = np.array(list(set(problemslices) - set(todelete)))
    todelete = np.array(list(set(todelete)))
    #Delete the slices that are redundant and adjust the list of slices to correct and the total slice list to reflect this
    for todel in todelete:
        del Slices[ keylist[todel] ]
        keylist.remove(keylist[todel])
        problemslices[problemslices > todel] -= 1
    overlaps =[]            #group colliding slices into sets of continuous collisions
    print('Removed crowding slices, adjusting remaining...')
    for k,g in groupby(enumerate(problemslices),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        overlaps.append((group[0],group[-1])) #the resulting set information will be used to guide the correction code
    print('list of overlaps',overlaps)
    Slices = fix_control_slice(overlaps,keylist,Slices,SurfaceObject)
    return Slices

def sample_nonplanar_slices(SurfaceObject,CenterLine,SamplingPoints,SampPerSlice,zeroind=0,tolerence=0.001):
    """
    Defines slices of an ROI mesh that follow the curve of a path through the mesh.

    Parameters
    ----------
    SurfaceObject : pyvista.PolyData
        The surface mesh of the ROI.
    CenterLine : pyvista.PolyData
        The spline of a path traversing the length of the ROI.
    SamplingPoints : 
        Array of the indices of points along the centerline where sampling 
        should occur. 
    SampPerSlice : int 
        Number of points to define around each slice.
    zeroind : int, optional
        The component (0=X,1=Y,2=Z) to set to 0 when solving for slice 
        intersection lines.
        By default, set X=0.
    tolerence : float, optional
        The closest distance two slices can come within one another before they
        are flagged as colliding. Defaults to 0.001 (one micrometer, assuming
        the position units are in millimeters).

    Returns
    -------      
    Slices : 
        A dictionary housing all the data for each slice. Each slice's data is 
        stored under a key corresponding to that slice's index along the 
        centerline (eg. 345) and contains the following:
        - “PlnNorm”: The unit normal vector of the plane used to create the slice.
        - “Org”: The point from which ray casting to define the slice originated 
        from.
        -”Slice”: A pyvista PolyData object of the slice itself, formatted as a 
        disk. Contains the points around the circumference of the slice used to 
        create a DSM, plus the point “Org” to facilitate the creation of a 
        triangulated mesh. “Org” is always the first point in the mesh’s point 
        data array.
    """
    #STEP1: start by calculating the tangents and the control points
    CenterLine = CenterLine.compute_derivative()
    Tangents = get_spline_tangents(CenterLine)
    ContPts = np.array(define_control_points(CenterLine['gradient']) )
    #STEP2: Define slices for the control points
    Slices = {}
    for cpt in ContPts:
        spoints, fails, NC = sample_uniangular(SurfaceObject,Tangents[cpt],CenterLine.points[cpt],SampPerSlice,fixfailures=True)
        slicemesh = helpers.make_slice_mesh(NC, spoints)#(CenterLine.points[cpt], spoints)
        Slices[cpt] = {'Slice':slicemesh,'PlnNorm':Tangents[cpt],'Org':NC}
    Slices = check_control_slices(Slices,SurfaceObject,zeroind)
    ContPts = np.array(list(Slices.keys()))
    #STEP3: begin looping through and establishing non-colliding slices
    CollisionsToFix = []
    LastSlice = None
    for pnt in SamplingPoints:
        print(pnt,'--------')
        overlap = False
        if pnt in ContPts:      #already in slice dictionary, can continue on
            LastSlice = None    #indicate next slice just needs to check 2 control slices, and not also last neighbor
            continue
        #STEP3a: Define the slices we'll check for collisions against
        ControlChecks = [ContPts[np.argmax(ContPts>pnt)-1], ContPts[np.argmax(ContPts>pnt)] ] 
        if LastSlice != None:
            ControlChecks.append(LastSlice) #check collisions with the control slices and the last slice
        #STEP3b: Check for colisions with other slices
        for CC in ControlChecks:
            #get intersection line properties
            v,p = helpers.find_plane_intersection(Tangents[pnt],CenterLine.points[pnt],Slices[CC]['PlnNorm'],Slices[CC]['Org'],zeroind)
            maxvdim = np.argmax(abs(v)) #adjust p so that it's centered at the x coordinate of the control slice
            p = helpers.select_intersection_point(v,p,Slices[CC]['Slice'])
            ppdist = pv.PolyData(p).compute_implicit_distance(Slices[CC]['Slice'] ) #compute the distance between p and the control slice's mesh
            #run a ray trace in two directions along the line to check for collisions with the reference slice
            Fcol,Rcol = Slices[CC]['Slice'].ray_trace(p,p+v*100) , Slices[CC]['Slice'].ray_trace(p,p-v*100)
            if  ( len(Fcol[0]) + len(Rcol[0]) ) > 0 or abs(ppdist['implicit_distance'][0]) < tolerence:
                #if we found a ray trace collision or the point-plane distance is under a micron do another ray tracce
                print('Possible collision detected. Confirming now...')
                #using the point slice ray tracing will occur from, check if the mesh blocks all the detected collsions before they reach the check slice
                basis = helpers.orthonormal_basis(Tangents[pnt])
                icent,openTF = helpers.adjust_centroid(CenterLine.points[pnt],basis,SurfaceObject)
                #check implemented due to very rare chance some failed ray traces yield a centroid outside the mesh
                ps,pind = SurfaceObject.ray_trace(icent,CenterLine.points[pnt])
                if len(ps)!=0:  
                    overlap=True
                    break
                for oPnt in Fcol[0]:
                    pp,pind = SurfaceObject.ray_trace(icent,oPnt) #see if collision happens before the slice
                    if len(pp) == 0 or np.linalg.norm(p-pp[0])<tolerence:
                        overlap=True
                        break
                for oPnt in Rcol[0]:
                    pp,pind = SurfaceObject.ray_trace(icent,oPnt) #see if collision happens before the slice
                    if len(pp) == 0 or np.linalg.norm(p-pp[0])<tolerence:
                        overlap=True
                        break
                pp,pind = SurfaceObject.ray_trace(icent,p) 
                if len(pp) == 0 or np.linalg.norm(p-pp[0])<tolerence or openTF==True:
                    overlap=True
                    break
            if overlap==True:
                break
        #STEP4: if no collisions detected, add to the Slices list. Otherwise, flag for adjustment
        if overlap == True:
            print('Overlap confirmed. Will be adjusted')
            CollisionsToFix.append(pnt)
            continue #skip on to the next
        else:
            spoints, fails, NC = sample_uniangular(SurfaceObject,Tangents[pnt],CenterLine.points[pnt],SampPerSlice,fixfailures=True)
            slicemesh= helpers.make_slice_mesh(NC, spoints)#(CenterLine.points[pnt], spoints)
            Slices[pnt] = {'Slice':slicemesh,'PlnNorm':Tangents[pnt],'Org':NC}
            LastSlice = pnt
    ##STEP5: Go back and adjust colliding slices
    allslices = np.sort(np.array(list(set(SamplingPoints).union(set(ContPts))) ) )#make a list of all slicees we have data for
    indices = np.where(np.in1d(allslices, np.array(CollisionsToFix)))[0] #get the indices where the colliding slices happened
    overlaps =[]            #group colliding slices into sets of continuous collisions
    for k,g in groupby(enumerate(indices),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        overlaps.append((group[0],group[-1])) #the resulting set information will be used to guide the correction code
    print('list of overlaps',overlaps)
    Slices = witztum_correction(overlaps,allslices,Slices,SurfaceObject,CenterLine)
    return Slices

####    -----------------------     ####
#   CODE TO SAMPLE DOSE AT GIVEN POINTS    #
####    -----------------------     ####
def sample_dose(samples,dosegrid,x,y,z):
    """
    Determines the dose at each point in a numpy array of coordinate points.

    Parameters
    ----------
    samples : numpy.ndarray
        A numpy array of point coordinates (xyz). The function accepts 1D, 2D,
        and 3D arrays, provided the following formatting is followed:
        - 1D array: Must be of shape (3,) and represent a single point (xyz).
        - 2D array: Must be of shape (M,3) and represent a list of points.
        - 3D array: Must be of shape (M,N,3), representing a grid of points
        that dose data is to be filled in for, where M is number of rows and 
        N is number of columns.
    dosegrid : numpy.ndarray
        The 3D dose matrix (MxNxP) of a radiotherapy plan, in the units specified
        by the RT Dose Dicom file. 
    x,y,z : numpy.ndarray
        One dimensional arrays giving the positional centers of the voxels of the
        dosegrid input. Automatically calculated by get_RT_Dose_data().

    Returns
    -------  
    output : numpy.ndarray
        A numpy array of dose information, organized based on the shape of the
        input sample array:
        - 1D input: Array of shape (1,), giving the dose of a single point.
        - 2D input: Array of shape (M,), giving a list of doses at each point in 
        the input array.
        - 3D input: Array of shape (M,N), giving the doses at each point in the
        input array organized into a grid.

    Notes
    -------  
    If calculating dose a set of points that share the same dose grid, it is
    faster to calculate the dose to all points at once rather than repeatedly
    running the function, as the interpolation grid only needs to be created 
    once.
    """
    #STEP1: define the interpolation function we will use to sample the dose grid
        #please note dicome dose grids in np. array format have indexes (Z,Y,X) and not (X,Y,Z) order
    dose_interp = RegularGridInterpolator((z,y,x),dosegrid,fill_value=None) #if point outside the grid, dont extrapolate
    #STEP2: Assess the shape of the input and raise exception if not correct
    shape = np.shape(samples)
    if shape[-1] != 3 or len(shape) > 3:
        print('Shape of the input array:',shape)
        raise Exception('Input samples not an accepted shape (accpeted shapes: (3,),(M,3), and (M,N,3) )')
    output = []
    #STEP3: loop through the sampling list and take the dose at each point
    if shape == (3,):
        if np.isnan(samples[0]):  #if coordinate missing for some reason, leave a hole in the map
            dose = np.nan
        else:
            dose = dose_interp(samples[::-1]) #use interpolation function to find dose at the point
        output.append(dose)
    #List case
    elif len(shape)==2:
        for point in samples:
            if np.isnan(point[0]):  #if coordinate missing for some reason, leave a hole in the map
                row.append(np.nan)
                continue
            dose = dose_interp(point[::-1]) #use interpolation function to find dose at the point
            output.append(dose)
    #Grid case
    elif len(shape)==3:
        for plane in samples:
            row = []
            for point in plane:
                if np.isnan(point[0]):  #if coordinate missing for some reason, leave a hole in the map
                    row.append(np.nan)
                    continue
                dose = dose_interp(point[::-1]) #use interpolation function to find dose at the point
                row.append(dose)
            output.append(row)
    #STEP4: convert to numpy array and return
    output = np.squeeze(np.array(output,dtype=np.float))
    return output

def sample_dsm(SliceDict,SamplingPoints,dosegrid,x,y,z):
    """
    Calculates a dose-surface map using information from a dictionary of slice 
    data.

    Parameters
    ----------
    SliceDict : dict
        A dictionary of point data for a set of slices of an ROI mesh. Either the
        output of the method sample_planar_slices or sample_nonplanar_slices, or
        else a custom dictionary formatted the same way.
    SamplingPoints : numpy.ndarray
        The list of points along a path through an ROI mesh used to create
        SliceDict. Corresponds to the ordered list of keys of the SliceDict 
        dictionary.
    dosegrid : numpy.ndarray
        The 3D dose matrix (MxNxP) of a radiotherapy plan, in the units specified
        by the RT Dose Dicom file. 
    x,y,z : numpy.ndarray
        One dimensional arrays giving the positional centers of the voxels of the
        dosegrid input. Automatically calculated by get_RT_Dose_data().

    Returns
    -------  
    DSM : numpy.ndarray
        The dose-surface map for the slices and dosegrid provided (MxN)

    """
    #STEP1: Convert the Slice mesh data to lists of coordiates (1 list per slice)
    dp_list = []    #list where the coordinates of all points on the DSM will be stored
    for j in SamplingPoints:
        rowpoints = SliceDict[j]['Slice'].points[1:]  #We exclude point 0 as it is the slice centroid used to construct the slice's mesh
        dp_list.append(rowpoints)
    dp_list = np.array(dp_list)

    #STEP2: Calculate dose at each point through regular grid interpolation
    DSM = sample_dose(dp_list,dosegrid,x,y,z)
    DSM = np.squeeze(DSM) #remove extra dimensions if they exist
    return DSM