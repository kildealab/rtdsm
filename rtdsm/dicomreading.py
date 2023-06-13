import numpy as np
import pydicom
import os
from rtdsm.helpers import PolygonArea

def contourdata_from_dicom(ROI_Name, RS_filepath):
    """
    Extracts the raw contour vertex data of a specified ROI from a RS-DICOM file.

    Parameters
    ----------
    ROI_Name : str
        The name of the ROI (region of interest) to retrive contour data for.
    RS_filepath : str
        The path to the RS-DICOM file.

    Returns
    -------
    contour : numpy.ndarray
        An array of the raw contour vertex data of the ROI. See Notes for 
        details.

    See Also
    --------
    get_pointcloud : Returns formatted contour point data and slice 
    centroids.

    Notes
    ------
    Contour data is returned in the same format it is stored in the Dicom. This 
    is in the form of n 1D arrays, where n is the number of closed polygonal 
    slices that comprise the contour. The points that comprise each slice are
    given in the 1D arrays as a sequence of (x,y,z) triplets in the Patient-
    Based Coordinate System.

    If the ROI name given does not exist in the Dicom file, the function
    returns the list of ROIs in the file and raises an exception.
    """
    #STEP1: Open the Dicom file and check if the requested ROI is present
    rs = pydicom.read_file(RS_filepath)
    ROI_index,ROIlist = None,[]
    for index, item in enumerate(rs.StructureSetROISequence):
        ROIlist.append(item.ROIName)
        if item.ROIName == ROI_Name:
            ROI_index = index
            break
    if ROI_index == None:
        raise Exception('An ROI with the name specified does not exist in the Dicom file. The available structures are:\n',ROIlist)
    #STEP2: Get all contour points from RS file (organized as: [[x0-N, y0-N, z0-N][x0-N, y0-N, z0-N]] )
    contour = []
    for item in rs.ROIContourSequence[ROI_index].ContourSequence:
        contour.append(item.ContourData)
    return np.array(contour)

def get_pointcloud(ROI_Name, RS_filepath, excludeMultiPolygons=True):
    """
    Creates a formatted array of an ROI's vertex data in addition to an array of
    slice centroids using a Dicom RTStruct file as input.

    Parameters
    ----------
    ROI_Name : str
        The name of the ROI (region of interest) to retrive contour data for.
    RS_filepath : str
        The path to the RS-DICOM file.
    excludeMultiPolygons : bool, optional
        Flag to indicate if multiple closed polygonal regions on the same slice
        of the reference image are kept or not (primarily used to exclude points
        that define holes in the center of the main contour polygon). Defaults
        to True. If True, only the points comprising the largest contour on a
        slice are added to the output array.

    Returns
    -------
    contour : numpy.ndarray
        Array of the ROI's vertex data, formatted into a list of coordinate 
        points (M x 3).
    CoMList : numpy.ndarray
        Array of the centroid coordinates for each slice of the ROI (M x 3).

    Notes
    -------
    Because of how the function get_cubemarch_surface() handles pointcloud data,
    the user must explictly specify how to handle multiple closed polygons on the
    same axial slice of the image used for contouring. Failure to do this will 
    result in them being combined into one continous polygon. By default, only
    the largest polygons are kept per slice. If inclusion of multiple polygons
    is specified to be allowed by the user, a coordinate of np.nan values will
    be added to the output arrays to differentiate the additional polygons from
    the rest of the structure.
    """
    #Function updated April 2023 by HP to better account for the following situations:
        #- Cases where the contours are not drawn on axial image slices
        #- Cases where DICOM pointcloud data is not provided in ascending Z order
        #   (as has been observed for contours provided by MIMvista)
    #STEP 1: get the raw mesh data
    mesh = contourdata_from_dicom(ROI_Name, RS_filepath)
    #STEP2: prep output data arrays
    xdata, ydata, zdata = [], [], []
    Ztracking = [[],[],[]]     #used to check for duplicate Z (axial) indices
    CoMList = []
    extrax, extray, extraz = [], [], [] #holds point data for extra polygons per slice
    extraCOM = []
    nonAxialSlices = False  #flag to indicate if structure slices are not aligned with image slices
    #STEP3: begin adding data
    for plane in mesh:
        #get points for point cloud
        xvals = (plane[0::3])
        yvals = (plane[1::3])
        zvals = (plane[2::3])
        #STEP3A: Check if structure slices are aligned with axial imaging plane
            #If not aligned, rtdsm will report how unaligned the slices are and
            #will align the slice at its average Z location
        if len(list(set(zvals))) > 1 and nonAxialSlices==False:
            print('WARNING: Slice is angled away from axial image plane')
            zmax = min(zvals)
            zmin = max(zvals)
            indmin = zvals.index(zmin)
            indmax = zvals.index(zmax)
            p1,p2 = np.array([xvals[indmin],yvals[indmin],zmin]),np.array([xvals[indmax],yvals[indmax],zmax])
            h = np.linalg.norm(p1-p2)
            o = zmax - zmin
            angle = math.degrees(math.asin(o/h))
            print('angle:',angle,'degrees')
            nonAxialSlices = True
        if nonAxialSlices == True:
            #use the average Z position instead of the actual Z values
            zval = round(sum(zvals)/len(zvals),2)
            zvals = [zval] * len(zvals)
        else:
            zval = plane[2]
        #STEP3B: Get the slice's polygon and calculate its area and CoM
        points = np.array([plane[0::3],plane[1::3]]).T
        area, COM = rtdsm.PolygonArea(points[:,0],points[:,1]), list(np.append(points.mean(axis=0),[zval]))
        #STEP3C: Check for other slices at the same Z location. If another slice
            #does exist, the largest will be kept and the smaller will be saved
            #to a secondary array that is added at the end of the pointcloud array
            #after a buffer of NaN data
            #NOTE: this is done to accomadate the method used to create the
            #the surface mesh of the contour
        if zval in Ztracking[0] and area > Ztracking[1][Ztracking[0].index(zval)]:
            #The prexisting slice is smaller. Replace with the new one
            extraCOM.append(CoMList[Ztracking[0].index(zval)]) #swap and update the COMs
            CoMList[Ztracking[0].index(zval)] = COM
            Ztracking[1][Ztracking[0].index(zval)] = area #update the area in the tracking
            Ztracking[2][Ztracking[0].index(zval)] += 1
            #move the smaller polygon points to the extra arrays
            oldindex = zdata.index(zvals[0])
            if Ztracking[2][Ztracking[0].index(zval)] > 2:
                extrax.append(np.nan)   #done to seperate two polygons on the same slice
                extray.append(np.nan)
                extraz.append(np.nan)
            extrax.extend(xdata[oldindex:])
            extray.extend(ydata[oldindex:])
            extraz.extend(zdata[oldindex:])
            #replace the old pointcloud ones with new
            xdata, ydata, zdata = xdata[:oldindex], ydata[:oldindex], zdata[:oldindex]
            xdata.extend(xvals)
            ydata.extend(yvals)
            zdata.extend(zvals)
        elif zval in Ztracking[0]:
            #The prexisting slice is larger. Add the new one to the extra array
            Ztracking[2][Ztracking[0].index(zval)] += 1
            extraCOM.append(COM)
            if Ztracking[2][Ztracking[0].index(zval)] > 2:
                extrax.append(np.nan)   #done to seperate two polygons on the same slice
                extray.append(np.nan)
                extraz.append(np.nan)
            extrax.extend(xvals)
            extray.extend(yvals)
            extraz.extend(zvals)
        else:
            #No other polygon exists on the slice, proceed normally
            Ztracking[0].append(zval)
            Ztracking[1].append(area)
            Ztracking[2].append(1)
            CoMList.append(COM)
            xdata.extend(xvals)
            ydata.extend(yvals)
            zdata.extend(zvals)
    #STEP 4: Final cleanup of the data
    #STEP4A: If excludeMultiPolygons FALSE, append extra arrays to the normal ones
    if excludeMultiPolygons == False and len(extraz)>0:
        if zdata[-1] == extraz[0]:
            xdata.append(np.nan)    #add a row of nan data if needed to prevent
            ydata.append(np.nan)    #point data for two polygons on the same slice
            zdata.append(np.nan)    #being confused as a single polygon
        xdata.extend(extrax)
        ydata.extend(extray)
        zdata.extend(extraz)
        CoMList.extend(extraCOM)
    #STEP4B: Ensure CoM data is presented in ascending Z position order
    CoMList = np.asarray(CoMList)
    if (CoMList[0,2] > CoMList[-1,2]):
        CoMList = np.flipud(CoMList)
        extraCOM = np.flipud(extraCOM)
        #the pointcloud data does not need to be switched as get_cubemarch_surface()
        #will do it automatically 
    #STEP5: Return the pointcloud and CoM data
    contour = np.array([xdata,ydata,zdata]).T
    return contour, CoMList

def get_doseinfo(RD_filepath):
    """
    Extracts the dose grid and axes of that dose grid from a DICOM RD file.

    Parameters
    ----------
    RD_folderpath : str
        The path to the DICOM RT dose file.

    Returns
    -------
    dosegrid : numpy.ndarray
        A 3D array of RT dose values in the units specified in the file.
    xpos,ypos,zpos : numpy.ndarray
        1D arrays specifying the locations of voxel centers in each dimension 
        (x,y,z). Required to interpolate dose to a given point.

    Notes
    -------
    The indices of the output dosegrid m,n,p are defined such that m = Z, n = Y,
    and p = X ( M[m,n,p] => M[z,y,z] ). Other functions in the package that use
    the dosegrid as input take this into account.
    """
    ds = pydicom.read_file(RD_filepath)
    #STEP1: get the dose grid
    dosegrid= ds.pixel_array*ds.DoseGridScaling #dose grid converted to proper dose units
    #STEP2: make positional arrays that will be used to lookup dose values when creating DSMs
        #each array monotonically increases from the top corner of the grid (imgpospat) 
        #with steps equal to the grid resolution (pixelspacing)
    xpos = np.arange(ds.Columns)*ds.PixelSpacing[0]+ds.ImagePositionPatient[0]
    ypos = np.arange(ds.Rows)*ds.PixelSpacing[1]+ds.ImagePositionPatient[1]
    zpos = np.asarray(ds.GridFrameOffsetVector) + ds.ImagePositionPatient[2]
    return dosegrid, xpos,ypos,zpos

def summarize_dicoms(folderpath):
    """
    Summarizes key data contained in RT Dose, RT Plan, RT Structure Set, or CT 
    Image DICOM files within a given folder that may be relevant to DSM creation.

    Parameters
    ----------
    folderpath : str
        The path to a directory containing the dicom files from a single
        radiotherapy treatment plan (or other clinical scenario).

    Returns
    -------
    Dict : dict 
        Dictionary of all data extracted from files found in the folder:
        - 'CT_res': The resolution of the CT file.
        - 'CT_kvp': The tube voltage used to take the CT image.
        - 'CT_mA': The tube current used to take the CT image.
        - 'CT_orient': The patient orientation used in the CT image.
        - 'RS_ROIs': The list of ROIs included in the RT Structure file.
        - 'RP_name': The name of the RT Plan.
        - 'RP_intent': The clinical intent of the RT Plan.
        - 'RP_presc': The prescription dose (in Gy) of the RT Plan.
        - 'RP_nfrac': The planned number of fractions of the RT Plan.
        - 'RD_res': The resolution (in mm) of the dose matrix of the RT Dose file.
        - 'RD_units': The dose units of the dose matrix of the RT Dose file.
        - 'RD_doseType': The type of dose calculated in the RT Dose file.
        - 'RD_sumType': The type of dose summation of the RT Dose file.

    Notes
    -------
    The variables returned are just a small sampling of the data contained within
    DICOM files. To extract more data from Dicom files read up on the pydicom
    package and consult the DICOM Standard.

    """
    #STEP1: Identify a file of each type in the folder (None if none of the type)
    Allfiles = os.listdir(folderpath)
    CT_file = next((s for s in Allfiles if 'CT.' in s), None)
    RS_file = next((s for s in Allfiles if 'RS.' in s), None)
    RP_file = next((s for s in Allfiles if 'RP.' in s), None)       
    RD_file = next((s for s in Allfiles if 'RD.' in s), None)
    MR_file = next((s for s in Allfiles if 'MR.' in s), None)   #TODO: Test and add display for MR files

    Dict = {}
    #STEP2: Retrieve CT information
    if CT_file != None:
        ds = pydicom.read_file(folderpath+CT_file)
        imgres = list(ds.PixelSpacing)
        imgres.append(ds.SliceThickness)
        imgres = [float(i) for i in imgres]
        kvp = float(ds.KVP)
        mA = float(ds.XRayTubeCurrent)
        ptort = str(ds.PatientPosition)
        print('=== CT Image Details =======\nVoxel Resolution:',imgres,'mm\nKVP:',kvp,'mA:',
            mA,'\nPatient Positioning:',ptort)
        Dict['CT_res'],Dict['CT_kvp'],Dict['CT_mA'],Dict['CT_orient'] = imgres,kvp,mA,ptort
    if RS_file != None:
        ds = pydicom.read_file(folderpath+RS_file)
        ROIlist = []
        for index, item in enumerate(ds.StructureSetROISequence):
            ROIlist.append(str(item.ROIName))
        print('=== RT Structure Details =======\nROIs Included:',ROIlist)
        Dict['RS_ROIs'] = ROIlist
    if RP_file != None:
        ds = pydicom.read_file(folderpath+RP_file) 
        plnname = str(ds.RTPlanLabel)
        plnintent = str(ds.PlanIntent)
        prscdose = float(ds.DoseReferenceSequence[0].TargetPrescriptionDose)
        nfrac = float(ds.FractionGroupSequence[0].NumberOfFractionsPlanned)
        print('=== RT Plan Details =======\nPlan Name:',plnname,'\nPlan Intent:',plnintent,
            '\nPrescription Dose:',prscdose,'Gy in',nfrac,'fractions')
        Dict['RP_name'],Dict['RP_intent'],Dict['RP_presc'],Dict['RP_nfrac'] = plnname,plnintent,prscdose,nfrac
    if RD_file != None:
        ds = pydicom.read_file(folderpath+RD_file) 
        doseres = list(ds.PixelSpacing)
        doseres = [float(i) for i in doseres]
        gf = ds.GridFrameOffsetVector[0:2]
        doseres.append(float(gf[1])-float(gf[0]))
        doseunits = str(ds.DoseUnits)
        dosetype = str(ds.DoseType)
        dosesum = str(ds.DoseSummationType)
        print('=== RT Dose Details =======\nDose Grid Resolution:',doseres,'mm\nDose Units:',doseunits,
            '\nDose Summation Type:',dosetype,dosesum)
        Dict['RD_res'],Dict['RD_units'],Dict['RD_doseType'],Dict['RD_sumType'] = doseres,doseunits,dosetype,dosesum
    ## NEED TO GET AN MR FILE AND ADD AN MR SECTION TOO

    return Dict
