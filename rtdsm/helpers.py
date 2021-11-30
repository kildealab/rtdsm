import numpy as np
import pyvista as pv
import math

def find_nearest(array,value):
    """
    Returns the index of an element in a sorted array closest in value to the 
    requested value.

    Parameters
    ----------
    array : numpy.ndarray
        A sorted 1D numpy array.
    value : float
        The value to find a closest match to in the array.

    Returns
    -------
    idx : int
        Index of array element closest to value.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def PolygonArea(x,y):
    """
    Uses Gauss's shoelace algorithm to estimate the area of a polygon.

    Parameters
    ----------
    x, y : list or numpy.ndarray
        X and Y coordinates of points comprising the polygon

    Returns
    ----------
    area : float
        Area of the polygon.
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def find_plane_intersection(n1,p1,n2,p2,zeroind=0):
    """
    Calculates the equation of the line of intersection between two planes in 
    3D space.

    Parameters
    ----------
    n1,n2 :  numpy.ndarray 
        The unit normal vectors of the two intersecting planes (1 x 3).
    p1,p2 : numpy.ndarray        
        Points located in planes 1 and 2 (required for plane equation, 1 x 3).
    zeroind : int, optional  
        The component (0=X,1=Y,2=Z) to set to 0 when solving for the line.
        By default, sets X=0.
    
    Returns
    ----------
    v : numpy.ndarray
        The unit vector of the intersection line.
    p : numpy.ndarray
        A point located on the intersection line.

    """
    #STEP1: compute cross product of the plane normals, which is the vector of the intersection line
    v = np.cross(n1,n2)
    v = v/np.linalg.norm(v) #convert to unit vector
    #STEP2: solve a system of plane equations to find a point on the line
    a = np.array([ np.roll(n1,-(zeroind+1))[:-1], np.roll(n2,-(zeroind+1))[:-1]  ]) #removed x,y, or z by effectively setting to 0
    b = np.array([ np.dot(n1,p1), np.dot(n2,p2) ])
    ans = np.linalg.solve(a,b)
    p = np.roll(np.array([ans[0],ans[1],0]),(zeroind+1) )   #convert back to coordinate point
    #STEP 3:return the vector and the point of the intersection line
    return v,p

def make_slice_mesh(centroid,points):
    """
    Creates a pyvista surface mesh of a slice through the ROI mesh using points 
    given.

    Parameters
    ----------
    centroid : numpy.ndarray      
        The coordinate (x,y,z) of the slice's centroid (1 x 3). 
    points : numpy.ndarray       
        An array of point coordinates on the surface of the ROI representing the
        slice. Typically this information comes from the output of 
        sample_uniangular(), (M x 3).

    Returns
    ----------
    slicemesh : pyvista.PolyData 
        Triangulated mesh of the given slice. Each face is composed of the 
        centroid and two points on the outer ring (similar apperance to a 
        spoked wheel).

    Notes
    ----------
    The point data used to create the mesh is accessible as slicemesh.points. The
    first point of this array is always the centroid of the slice.
    
    See Also
    ----------
    sample_uniangular : Returns n equiangular points on the surface of the ROI mesh.

    """
    #STEP1: Create a numpy array of all points
    slicepoints = np.zeros((len(points)+1,3))
    slicepoints[0],slicepoints[1:] = centroid,points
    #STEP2: Create a numpy array of pyvista readable faces (each row = [npoints ind1    ind2... ...indn])
    faces = np.array( [np.full(len(points),3), np.arange(1,len(points)+1), np.zeros(len(points)), np.roll(np.arange(1,len(points)+1),-1) ],dtype=np.uint8).T
    #STEP3: define a pyvista mesh for the slice
    slicemesh = pv.PolyData(slicepoints,faces=faces)
    return slicemesh

def orthonormal_basis(normal):
    """
    Defines a set of 3 basis vectors (i,j,k) for a specified plane normal. By
    design, k is the plane normal and j is restricted to the y-z plane.

    Parameters
    ----------
    normal : numpy.ndarray    
        The normal vector of a plane.
    
    Returns
    ----------
    i,j,k : numpy.ndarray
          The basis vectors for the given plane.
    """
    #STEP1: Define j such that it is in the yz plane, using the property dot(j,k)=0 and setting j1 = 0
    j = np.array([0,normal[2],-normal[1]])
    #STEP2: take the cross product of normal and j to get i
    i = np.cross(j,normal)
    i,j,k = i/np.linalg.norm(i), j/np.linalg.norm(j), normal/np.linalg.norm(normal) #make unit vecotrs
    return i,j,k

def adjust_centroid(centroid,basis,SurfaceObject):
    """
    Approximates the centroid of a proposed slice by calculating the center of 
    mass of 8 equiangular points on the slice surface.

    Parameters
    ----------
    centroid : numpy.ndarray
        Coordinate of a point on a slice that may or may not be the centroid,
        but is currently considered as it.
    basis : tuple or list
        Set of basis vectors for the slice plane. Typically the output of 
        orthonormal_basis().
    SurfaceObject : pyvista.PolyData
        The surface mesh of the ROI.
    
    Returns
    ----------
    newcent : numpy.ndarray          
        Updated centroid coordinate. 
    Outbounds : bool
        Flag indicating if the centroid calculation found one or more angles
        at which a ray casting vector could exit the SurfaceObject without
        encoutering a collision. If True, this either indicates the slicing
        plane needs to be adjusted to avoid creating a slice that exits the
        surface mesh at one of the ends, or that the mesh contains holes.

    Notes
    ----------
    Adjusting the sampling point for a slice to be approximately at the
    centroid is important in order to have evenly spaced points around 
    the surface of the slice to form the DSM. 

    """
    #STEP1: define 8 vectors spaced 45 degrees from one another in the sampling plane
    sampangs = np.linspace(0,2*math.pi,9)[:-1]
    rayvects = basis[0]*np.ones((len(sampangs),3))*np.sin(sampangs[:, np.newaxis]) + basis[1]*np.ones((len(sampangs),3))*np.cos(sampangs[:, np.newaxis])
    rayvects = 200*rayvects #rays will travel 20 cm in search of the mesh surface
    #STEP2: get points on the surface of the contour mesh
    samppoints = np.full_like(rayvects,np.nan)
    bounds = np.reshape(SurfaceObject.bounds,(3,2)).T #used to check 
    Outbounds = False
    for i in range(len(rayvects)):
        inter,ind = SurfaceObject.ray_trace(centroid,centroid+rayvects[i],first_point=True) #give start and end of ray, and only return first intersection
        if len(inter) == 0:
            #if no hit with the mesh detected, check if the ray leaves the bounding box of the mesh
            boundcheck = bounds < centroid+rayvects[i]
            if np.any(boundcheck[0,:])==False or np.any(boundcheck[1,:])==True:
                Outbounds = True
            continue
        samppoints[i,:] = inter
    if Outbounds == True:
        print('Warning: Proposed slice is open towards one of the ROI bounds. Check mesh quality if this slice is not flagged for correction.')
    #STEP3: Calculate the centroid of the 8 points
    newcent = samppoints[~np.isnan(samppoints).any(axis=1)].mean(axis=0)
    return newcent, Outbounds

def slice_perimeter(mesh):
    """
    Calculates the perimeter of a slice mesh created with make_slice_mesh().
    
    Parameters
    ----------
    centroid : pyvista.PolyData   
        Mesh object of a slice of the ROI.
    
    Returns
    ----------
    Dtot : float     
        The perimeter of the mesh.
    """
    points = mesh.points[1:] #get the list of all the points on the perimeter of the mesh
    Dtot = 0
    #loop through the list of points and add up the distances between pairs of neighbours
    for i in range(len(points)-1):
        di = np.linalg.norm(points[i+1] - points[i])
        Dtot = Dtot + di
    return Dtot

def select_intersection_point(v,p,mesh):
    """
    Finds a point on a line within the bounds of a slice mesh that is the
    closest to the slice mesh itself.

    Parameters
    ----------
    v : numpy.ndarray
        Vector of the line. Used with p to define line equation.
    p : numpy.ndarray
        A point on the line. Used with v to define line equation.
    mesh : pyvista.PolyData   
        Mesh of the slice of interest (usually output of make_slice_mesh).
    
    Returns
    -------
    point : numpy.ndarray
        The point on the intersection line closest to the slice mesh
    """
    #STEP1: set up lambda functions and bounds of the mesh
    newT = lambda val,ind: (val - p[ind])/v[ind]    #find t given two points along a line
    newP = lambda t : p + v*t           #equation of a point on a line
    bounds = mesh.bounds
    maxvdim = np.argmax(abs(v)) #identifies if the x,y, or z dimension has the highest vector velocity component
    usebounds = bounds[2*maxvdim:2*maxvdim+2]   #selects the bounds correspoinding to the largest velocity component to be the ones used
    #STEP2: check 10 points within the bounds and see if any fall on the slice (otherwise just give the closest)
    closest = (0,10000) #(T, distance)
    for i in np.linspace(usebounds[0],usebounds[1],10):
        nT = newT(i,maxvdim)
        ppdist = pv.PolyData(newP(nT)).compute_implicit_distance(mesh)
        if abs(ppdist['implicit_distance'][0])<=closest[1]:
            closest = (nT,abs(ppdist['implicit_distance'][0])) 
    point = p+closest[0]*v
    return point   #return the closest point to the slice mesh
