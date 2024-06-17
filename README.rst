.. image:: https://github.com/3DFin/dendromatics/blob/main/docs/_static/dendromatics_logo.png
  :width: 300
  :align: center

Dendromatics
============

|docs| |pypi| |tests|

.. |docs| image:: https://readthedocs.org/projects/dendromatics/badge/?version=latest
    :alt: Documentation Status
    :target: https://dendromatics.readthedocs.io/en/latest/?badge=latest

.. |pypi| image:: https://badge.fury.io/py/dendromatics.svg
    :alt: PyPI status
    :target: https://badge.fury.io/py/dendromatics

.. |tests| image:: https://github.com/3DFin/dendromatics/actions/workflows/test.yml/badge.svg
    :alt: Tests status
    :target:  https://github.com/3DFin/dendromatics/actions


Description
===========

The *src* folder contains functionalities to detect the trees present in a terrestrial 3D point cloud from a forest plot, and compute individual tree parameters: tree height, tree location, diameters along the stem (including DBH), and stem axis. These are based on an updated version of the algorithm proposed by (Cabo et al., 2018).

The functionalities may be divided in four main steps:

0. Height-normalization of the point cloud. 
1. Identification of stems among user-provided stripe.
2. Tree individualization based on point-to-stems distances.
3. Robust computation of stems diameter at different section heights.

Although individual, somewhat independent functions are provided, they are designed to be used in a script that calls one after the other following the algorithm structure. An example script can be found in `example` folder.


Examples
========


Height-normalization
--------------------

Almost all functions in the module expect a height-normalized point cloud to work as intended. If your point cloud is not height-normalized, you can do it in a simple way using some of the module functions. I'ts based on 'Cloth simulation Filter' (Zhang et al., 2016).

.. code-block:: python
    
    import laspy
    import numpy as np
    import dendromatics as dm

    # Reading the point cloud
    filename_las = 'example_data.las' # your .las file
    entr = laspy.read(filename_las)
    coords = np.vstack((entr.x, entr.y, entr.z)).transpose()
    
    # Normalizing the point cloud
    dtm = dm.generate_dtm(clean_points)
    z0_values = dm.normalize_heights(coords, dtm)

    # adding the normalized heights to the point cloud
    coords = np.append(coords, np.expand_dims(z0_values, axis = 1), 1) 

If the point cloud is noisy, you might want to denoise it first before generating the DTM.

.. code-block:: python

    clean_points = dm.clean_ground(coords)


Identifying stems from a stripe
-------------------------------

Simply provide a stripe (from a height-normalized point cloud) as follows to iteratively 'peel off' the stems.

.. code-block:: python

    # Defining the stripe
    lower_limit = 0.5
    upper_limit = 2.5
    stripe = coords[(coords[:, 3] > lower_limit) & (coords[:, 3] < upper_limit), 0:4]

    stripe_stems = dm.verticality_clustering(stripe, n_iter = 2)  


Individualizing trees
---------------------

Once the stems have been identified in the stripe, they can be used to individualize the trees in the point cloud.

.. code-block:: python
   
    assigned_cloud, tree_vector, tree_heights = dm.individualize_trees(coords, stripe_stems)     


Computing sections along the stems
----------------------------------

compute_sections() can be used to compute sections along the stems of the individualized trees.

.. code-block:: python

    # Preprocessing: reducing the point cloud size by keeping only the points that are closer than some radius (expected_R) to the tree axes 
    # and those that are whithin the lowest section (min_h) and the uppest section (max_h) to be computed.
    expected_R = 0.5
    min_h = 0.5 
    max_h = 25
    section_width = 0.02
    xyz0_coords = assigned_cloud[(assigned_cloud[:, 5] < expected_R) & (assigned_cloud[:, 3] > min_h) & (assigned_cloud[:,3] < max_h + section_width), :]
    
    stems = dm.verticality_clustering(xyz0_coords, n_iter = 2)[:, 0:6]
    
    # Computing the sections
    section_len = 0.2
    sections = np.arange(min_h, max_h, section_len) # Range of uniformly spaced values within the specified interval 
    X_c, Y_c, R, check_circle, second_time, sector_perct, n_points_in = dm.compute_sections(stems, sections)


Tilt detection 
--------------

tilt_detection() computes an 'outlier probability' for each section based on its tilting relative to neighbour sections and relative to the tree axis.

.. code-block:: python
    
    outlier_prob = dm.tilt_detection(X_c, Y_c, R, sections)


For further examples and more thorough explanations, please check *example.py* script in */examples* folder.


References
==========

Cabo, C., Ordóñez, C., López-Sánchez, C. A., & Armesto, J. (2018). Automatic dendrometry: Tree detection, tree height and diameter estimation using terrestrial laser scanning. International Journal of Applied Earth Observation and Geoinformation, 69, 164–174. https://doi.org/10.1016/j.jag.2018.01.011


Prendes, C., Cabo, C., Ordoñez, C., Majada, J., & Canga, E. (2021). An algorithm for the automatic parametrization of wood volume equations from Terrestrial Laser Scanning point clouds: application in Pinus pinaster. GIScience and Remote Sensing, 58(7), 1130–1150. https://doi.org/10.1080/15481603.2021.1972712 


Zhang, W., Qi, J., Wan, P., Wang, H., Xie, D., Wang, X., & Yan, G. (2016). An easy-to-use airborne LiDAR data filtering method based on cloth simulation. Remote Sensing, 8(6). https://doi.org/10.3390/rs8060501


Install
=======

*dendromatics* is available on `PyPI <https://pypi.org/project/dendromatics/>`_ and the full documentation can be consulted on `ReadTheDocs.io <https://dendromatics.readthedocs.io/en/latest/>`_

.. code-block:: console
    
    python -m pip install dendromatics

The list of dependencies is available in the *pyproject.toml* file.

*dendromatics* relies on `hatch <https://github.com/pypa/hatch>` (version > 1.12)

Depending on your version of Python and your OS, you might also need a C/C++ compiler to compile some of the mandatory dependencies (CSF and jakteristics). 
But in any case you would not have to run the compiler by yourself, the build system will manage dependencies and compilation for you. 

.. code-block:: console
    
    python -m pip install hatch

You can run tests to ensure it works on your computer.

.. code-block:: console
    
    hatch test -c

It is also possible to build doc locally.

.. code-block:: console
   
    hatch run docs:build
    hatch run docs:serve

and then go to `http://localhost:8000 <http://localhost:8000>`_ to browse it.
