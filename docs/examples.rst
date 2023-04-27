Examples
========


Height-normalization
--------------------

Almost all functions in the module expect a height-normalized point cloud to work as intended. If your point cloud is not height-normalized, you can do it in a simple way using some of the module functions::
    
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

If the point cloud is noisy, you might want to denoise it first before generating the DTM::
    
    clean_points = dm.clean_ground(coords)


Identifying stems from a stripe
-------------------------------

Simply provide a stripe (from a height-normalized point cloud) as follows to iteratively 'peel off' the stems::
    
    # Defining the stripe
    lower_limit = 0.5
    upper_limit = 2.5
    stripe = coords[(coords[:, 3] > lower_limit) & (coords[:, 3] < upper_limit), 0:4]

    stripe_stems = dm.verticality_clustering(stripe, n_iter = 2)  


Individualizing trees
---------------------

Once the stems have been identified in the stripe, they can be used to individualize the trees in the point cloud::
    
    assigned_cloud, tree_vector, tree_heights = dm.individualize_trees(coords, stripe_stems)     


Computing sections along the stems
----------------------------------

compute_sections() can be used to compute sections along the stems of the individualized trees::
    
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

tilt_detection() computes an 'outlier probability' for each section based on its tilting relative to neighbour sections and the relative to the tree axis::
    
    outlier_prob = dm.tilt_detection(X_c, Y_c, R, sections)


For further examples and more thorough explanations, please check *example.py* script in */examples* folder in the official GitHub repository:

https://github.com/3DFin/dendromatics




