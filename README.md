# Description


The `code` folder contains functionalities to detect the trees present in a terrestrial 3D point cloud from a forest plot, and compute individual tree parameters: tree height, tree location, diameters along the stem (including DBH), and stem axis. These are based on an updated version of the algorithm proposed by (Cabo et al., 2018) which is mainly based on rules, although DBSCAN clusterization is used in some stages.


The functionalities may be divided in four main steps:

0. Height-normalization of the point cloud (pre-requisite). 
1. Identification of stems among user-provided stripe.
2. Tree individualization based on point-to-stems distances.
3. Robust computation of stems diameter at different section heights.


# Algorithm


The algorithm is an updated version of the one presented in (Cabo et al., 2018). It requires a height-normalized point cloud as input, as in Figure 1. 


![height-normalization](https://user-images.githubusercontent.com/125979752/220949152-e0fcf390-dcfa-46ca-aec6-d78ff87b5528.jpg)


_Figure 1. The algorithm requires height-normalized point clouds. A) Original point cloud. B) Height-normalized point cloud. From (Cabo et al. 2018)._

## 1. Identification of stems among user-provided horizontal stripe


In this first step, the user selects a stripe, defined this as a subset of the original cloud delimited by a lower height ( $Z_{h(low)}$ ) and an upper height ( $Z_{h(high)}$ ), which will narrow down a region where it is expected to only encounter stems. The points within the stripe will be voxelated and their verticality will be computed, based on fixed radius neighbourhoods. Then, they will be filtered based on their verticality value. After this, the remaining points will be clustered using the DBSCAN algorithm (Ester et al., 1996). These procedures will be repeated iteratively a user-defined number of times. At this stage, the potential stems are referred as ‘voxel groups’. Figure 2 illustrates this step of the algorithm.


![stripe_and_groups](https://user-images.githubusercontent.com/125979752/220949312-b4021e97-fc43-4d9c-a106-cec3bf494e21.jpg)


_Figure 2. Stripe on the height-normalized point cloud, and candidate voxel groups. Points in the stripe in red, and voxel groups in random colours. From (Cabo et al. 2018)._


## 2. Tree individualization based on point-to-stems distances


Once the voxel groups have been computed and properly peeled-off, they are isolated and enumerated, and then, their axes are identified using PCA (PCA1 direction). These axes will be henceforth considered as stem axes. This allows to group points based on their distance to those axes, thus assigning each point to a tree. This is illustrated in Figure 3. 


![individualized_trees](https://user-images.githubusercontent.com/125979752/220949614-01e7a7f2-1868-4939-bc6e-70371de1eac3.jpg)


_Figure 3. Isolated trees. Note that ground and understory points are assigned as well to the closest axis. From (Cabo et al. 2018)._


During this step of the algorithm the tree height is computed as well. For this, and, for each tree, the points that are under a certain distance to the stem axis are selected, voxelated again using a higher resolution and clustered with DBSCAN algorithm. From the points that belong to the main cluster (the one that englobes the tree), the highest point is selected, and its height is considered as the tree height. This allows to exclude from the search of the highest point those that could belong to other trees or any noise that happened to be above the tree whilst being scanned. Figure 4 illustrates this.


![tree_height](https://user-images.githubusercontent.com/125979752/220949657-5a2220dd-822a-4adc-8ce7-458c78c0a12f.jpg)


_Figure 4. Total tree height (TH) computation. Note that it avoids isolated point clusters that may not belong to the tree. From (Cabo et al. 2018)._


## 3. Computation of stem diameter at different section heights


In this final step a set of heights is defined, which will then be used to measure the stem diameter at different sections around the tree axes. To do so, a slice of points will be selected at every section, and those will be fit a circle by least squares minimization. This procedure is similar as the one proposed in (Prendes et al., 2021)


To ensure robustness, the goodness of fit is checked. What follows is a brief list of all the __tests__ that are performed:

* Number of points inside the circle. This is checked via fitting an __inner circle__
* Percentage of __occupied sectors__
* Size of fitted circle (if it is __radius is too small/big__)
* __Vertical deviation from tree axis__ ('outlier probability’)


First, a complementary, inner circle is fitted as well, which will be used to check how points are distributed inside the first circle: they are expected to be outside the inner circle, as the scanning should only scan the surface of the stems. Second, the section is divided in several sectors to check if there are points within them (so they are occupied). If there are not enough occupied sectors, the section fails the test, as it is safe to assume it has an abnormal, non-desirable structure. After this, it is checked whether the diameter of the fitted circle is within some boundaries, to discard anomalies. Finally, the vertical deviation from the tree axis is computed for every section and it is used to check possible bad fittings: highly deviated sections are labelled as possible outliers. 


On top of all goodness of fit tests, there is a last layer or robustness while computing the diameters. If the first fit is not appropriate, another circle will be fitted to substitute it using only points from the largest cluster in the slice of points, and the goodness of fit will be tested again. Figure 5 illustrates an example of some fitted circles after all tests and their respective axes.


![sections_and_axes](https://user-images.githubusercontent.com/125979752/220949703-2c6423f4-dd62-4b0c-8feb-3d202a85ccfb.jpg)


_Figure 5. Fitted circles in 6 stems, at sections ranging from 0.3 to a maximum of 25.0 meters, one every 0.2 meters. Blue circles passed all quality tests, while red circles mean the fitting may be unreliable. This may be due to partial scans, non-expected diameter measurements, non-reasonable distribution of points within the section or a high value of tilting. Computed axes are represented at the right._


During this step, besides computing all the diameters at the selected heights, the DBH will be approximated as well (even if BH was not included as one of the selected heights). For this, the section closest to 1.3 m will be used as a proxy, and the DBH will only be computed if there is coherence between that section and the ones around. 


Tree location [(x, y) coordinates] is obtained at this step too, either derived from the proxy section (to BH) when appropriate; that is, when it passes all goodness of fit tests and it is coherent, or from the tree axis when not.


# Examples

## Height-normalization

Almost all functions in the module expect a height-normalized point cloud to work as intended. If your point cloud is not height-normalized, you can do it in a simple way using some of the module functions.

```Python

import laspy
import numpy as np
import dendromatic as dm

### Reading the point cloud ###
entr = laspy.read(filename_las)
coords = np.vstack((entr.x, entr.y, entr.z)).transpose()

### Normalizing the point cloud ###
cloth_nodes = dm.generate_dtm(clean_points)
z0_values = dm.normalize_heights(coords, dtm)

coords = np.append(coords, np.expand_dims(z0_values, axis = 1), 1) # adding the normalized heights to the point cloud

```
If the point cloud is noisy, you might want to denoise it first before generating the DTM:

```Python

clean_points = dm.clean_ground(coords)

```

## Identifying stems from a stripe

Simply provide a stripe (from a height-normalized point cloud) as follows to iteratively 'peel off' the stems:

```Python

lower_limit = 0.5
upper_limit = 2.5
stripe = coords[(coords[:, 3] > lower_limit) & (coords[:, 3] < upper_limit), 0:4]

stripe_stems = dm.verticality_clustering(stripe, n_iter = 2)       

```

## Individualizing trees

Once the stems have been identified in the stripe, they can be used to individualize the trees in the point cloud:

```Python 

assigned_cloud, tree_vector, tree_heights = dm.individualize_trees(coords, stripe_stems)     

```

## Computing sections along the stems

`compute_sections()` can be used to compute sections along the stems of the individualized trees:

```Python

# Preprocessing: reducing the point cloud size by keeping only the points that are closer than some radius (expected_R) to the tree axes 
# and those that are whithin the lowest section (min_h) and the uppest section (max_h) to be computed.
min_h = 0.5 
max_h = 25
section_width = 0.02

xyz0_coords = assigned_cloud[(assigned_cloud[:, 5] < expected_R) & (assigned_cloud[:, 3] > min_h) & (assigned_cloud[:,3] < max_h + section_width), :]
stems = dm.verticality_clustering(xyz0_coords, n_iter = 2)[:, 0:6]

# Computing the sections

section_len = 0.2
sections = np.arange(min_h, max_h, section_len) # Range of uniformly spaced values within the specified interval 

X_c, Y_c, R, check_circle, second_time, sector_perct, n_points_in = dm.compute_sections(stems, sections)

```

## Tilt detection 

`tilt_detection()` computes an 'outlier probability' for each section based on its tilting relative to neighbour sections and the relative to the tree axis:

```Python

outlier_prob = dm.tilt_detection(X_c, Y_c, R, sections, w_1 = 3, w_2 = 1)

```

For further examples and more thorough explanations, please check `example.py` script in `/examples` folder.


# Dependencies


The script imports several Python libraries and functions. They can be found listed below:

__libraries__. These are libraries that are imported directly:

* os
* sys
* laspy
* numpy
* timeit

__functions__. These are libraries from which only specific functions are imported:

* copy (imports deepcopy)
* jakteristics (imports compute_features)
* scipy (imports cluster.hierarchy, optimize and spatial.distance_matrix)
* sklearn (imports cluster.DBSCAN and decomposition.PCA)
* tkinter (imports filedialog)


# Inputs


The main script takes a .LAS file containing the ground-based 3D point cloud as input. The point cloud must be height normalized. Normalized heights can be contained in the Z coordinate of the point cloud or in an additional field in the .LAS file. If so, the name of that field is used as an input parameter. The parameters may be considered as inputs as well.


# Outputs


After all computations are complete, the following files are output:


__LAS files__:


* LAS file containing the original point cloud and a scalar field that contains tree IDs.

* LAS file containing trunk axes coordinates.

* LAS file containing circles (sections) coordinates.

* LAS file containing the trunks obtained from the stripe during step 1.

* LAS file containing the tree locators coordinates.


__Text files__:


* Text file containing tree height, tree location and DBH of every tree as tabular data.

* Text file containing the (x) coordinate of the center of every section of every tree as tabular data.

* Text file containing the (y) coordinate of the center of every section of every tree as tabular data.

* Text file containing the radius of every section of every tree as tabular data.

* Text file containing the 'outlier probability' of every section of every tree as tabular data.

* Text file containing the sector occupancy of every section of every tree as tabular data.

* Text file containing the 'check' status of every section of every tree as tabular data.


# References


Cabo, C., Ordóñez, C., López-Sánchez, C. A., & Armesto, J. (2018). Automatic dendrometry: Tree detection, tree height and diameter estimation using terrestrial laser scanning. International Journal of Applied Earth Observation and Geoinformation, 69, 164–174. https://doi.org/10.1016/j.jag.2018.01.011


Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. www.aaai.org


Prendes, C., Cabo, C., Ordoñez, C., Majada, J., & Canga, E. (2021). An algorithm for the automatic parametrization of wood volume equations from Terrestrial Laser Scanning point clouds: application in Pinus pinaster. GIScience and Remote Sensing, 58(7), 1130–1150. https://doi.org/10.1080/15481603.2021.1972712 
