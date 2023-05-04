# rtdsm
**rtdsm** provides a set of python functions that facilitate the calculation of dose-surface maps (DSMs) from radiotherapy treatment planning data. This package empowers users to create both common styles of dose-surface maps (axial parallel-planar and nonplanar centerpath tangential) with a high degree of customization as well as extract common DSM spatial features for use in dose-outcome analysis.

## Table of Contents
* [Motivation](#motivation)
* [Features](#features)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Examples and Documentation](#examples-and-documentation)
* [License](#license)
* [Citing rtdsm](#citing-rtdsm)
* [References](#references)

## Motivation
Radiotherapy research typically uses dose-volume histograms (DVHs), which condense 3D dose distributions within a region of interest (ROI) into a 2D representation, as their main source of dosimetric data for dose-outcome studies. However, DVHs lack spatial information, making it difficult to identify specific sub regions or dose-spread patterns in organs that can be indicative of higher risk of adverse events. Dose-surface maps (DSMs) preserve this spatial information and make it possible to identify spatio-dosimetric features on the surface of any ROI.

DSMs are most commonly used for hollow, tubular organs (like the esophagus and rectum) to characterize the dose to the actual tissue wall and ignore dose to the filling contents. They can be thought of as tubes painted with dose information that have been cut open to allow them to lay flat as a 2D dose map. Simplistically, all that is required to create a DSM is some representation of the organ (like a 3D contour) and a dose field with which to paint it. In reality, the process includes several steps where the organ is (1) segmented into slices, (2) dose is sampled at N points around each slice, and (3) the slices are cut open to produce the 2D map.

Different research groups have developed different approaches for each of these steps, with the largest methodological variation being how slices are defined. The simpler method, popularized by Buettner *et al* and referred to as planar slicing in the package, creates slices of the ROI on the same axial planes that house the slices of the CT/MR image the contour was drawn on. While most common due to its simplicity, it ignores any curvature of the organ and may not produce appropriate representations of the surface dose for organs with high curvature. The more complex method accounts for the organâ€™s curvature and uses sampling planes perpendicular to the central path of the organ and is heretofore referred to as nonplanar slicing in the package. Because the nonplanar approach does not use parallel slicing planes additional checks are required to detect and adjust slices that overlap with one another that can be difficult to implement. For this reason only a few groups have developed (private) code for this method, such as Hoogeman *et al* and Witztum *et al*.

**rtdsm** facilitates the calculation of DSMs with both types of slicing methods to empower researchers to pursue either option, regardless of their programming ability.

## Features
- DICOM-RT importing
- DSM generation with either planar or nonplanar slicing approach 
- Built-in methods to calculate common DSM spatial dose metrics
- Implementation of the popular multiple comparisons permutation test to identify statistically significant subregions (as of May-2023)
- Support for DSM accumulation and EQD2 conversions
- Highly customizable to individual sampling requirements 
- Robust example library
- Open source

## Dependencies
- Python >= 3.6
- Numpy >= 1.19.5
- Pyvista >= 0.31.3
- Pydicom >= 2.1.2
- Scipy >= 1.5.4
- Skimage >= 0.17.2
- Cv2  >= 4.2.0.34	*Only needed to calculate ellipse-based spatial dose metrics

Note: This package was developed using Python 3.6.1 on Linux (Ubuntu 12.04.5) with the package versions listed above.

## Installation
**rtdsm** can be installed by pulling the latest version from GitHub. Please note that your Python installation should be 3.6 or later.

## Examples and Documentation
Several examples are included in the package to demonstrate how to perform the most common tasks associated with creating DSMs, as well as to provide context for the methodology used. Example files are also given for easy start-up.

| Example| Description | 
| --- | --- | 
| [`make_planar_dsm.py`](examples/make_planar_dsm.py) | Quickstart example demonstrating the creation of a planar DSM. | 
| [`make_nonplanar_dsm.py`](examples/make_nonplanar_dsm.py) | Quickstart example demonstrating the creation of a nonplanar DSM. |
| [`load_structure_from_dicom.py`](examples/load_structure_from_dicom.py) | Read in structure data from an RTStructure DICOM file. |
| [`make_surface_mesh.py`](examples/make_surface_mesh.py) | Create a surface mesh from RTStructure DICOMdata. |
| [`load_dose_from_dicom.py`](examples/load_dose_from_dicom.py) | Read in dose information from an RTDose DICOM file. |
| [`dose_sampling.py`](examples/dose_sampling.py) | Sample dose from an RT dosegrid. |
| [`set_sampling_points.py`](examples/set_sampling_points.py) | Identify points spaced along a spline for use during nonplanar slicing. |
| [`planar_slicing.py`](examples/planar_slicing.py) | Use the planar slicing method to create slices of a surface mesh. |
| [`nonplanar_slicing.py`](examples/nonplanar_slicing.py) | Use the nonplanar slicing method to create slices of a surface mesh. |
| [`make_dsh.py`](examples/make_dsh.py) | Make a dose-surface histogram (DSH) using a DSM. |
| [`make_clustermask.py`](examples/make_clustermask.py) | Create a cluster mask of a DSM to use to calculate spatial dose metrics. |
| [`extract_clusterfeatures.py`](examples/extract_clusterfeatures.py) |  Calculate cluster-based spatial dose metrics for a DSM. |
| [`extract_ellipsefeatures.py`](examples/extract_ellipsefeatures.py) | Calculate ellipse-based spatial dose metrics for a DSM. |
| [`convert_dsm.py`](examples/convert_dsm.py) | Convert a DSM to a different fractionation scheme (such as 3Gy/frac to 2Gy/frac). |
| [`combining_dsms.py`](examples/combining_dsms.py) | Add, subtract, or average multiple DSMs with each other. |
| [`run_mcp_test.py`](examples/run_mcp_test.py) | Perform a multiple comparisons permutation test to compare two groups of DSMs. |

Full documentation is a work in progress and will be available on readthedocs when complete. Code is already documented and commented, just not formatted as HTML.

## License
This project is provided under the GNU GLPv3 license to preserve open-source access to any derivative works. See the LICENSE file for more information.

## Citing rtdsm
If you publish any work using this package, please make sure you acknowledge us by citing the following paper:
Patrick HM, Kildea J. Technical note: rtdsm-An open-source software for radiotherapy dose-surface map generation and analysis. Med Phys. 2022 Nov;49(11):7327-7335. doi: 10.1002/mp.15900. Epub 2022 Aug 8. PMID: 35912447.

## References
- Buettner F, Gulliford SL, Webb S, Partridge M. Using dose-surface maps to predict radiation-induced rectal bleeding: a neural network approach. Phys Med Biol. 2009 Sep 7;54(17):5139-53. doi: 10.1088/0031-9155/54/17/005. Epub 2009 Aug 6. PMID: 19661568.
- Hoogeman MS, van Herk M, de Bois J, Muller-Timmermans P, Koper PC, Lebesque JV. Quantification of local rectal wall displacements by virtual rectum unfolding. Radiother Oncol. 2004 Jan;70(1):21-30. doi: 10.1016/j.radonc.2003.11.015. PMID: 15036848.
- Witztum A, George B, Warren S, Partridge M, Hawkins MA. Unwrapping 3D complex hollow organs for spatial dose surface analysis. Med Phys. 2016 Nov;43(11):6009. doi: 10.1118/1.4964790. PMID: 27806596.
- Chen C, Witte M, Heemsbergen W, van Herk M. Multiple comparisons permutation test for image based data mining in radiotherapy. Radiat Oncol. 2013 Dec 23;8:293. doi: 10.1186/1748-717X-8-293. PMID: 24365155.
