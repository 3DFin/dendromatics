### Example script ###

import laspy
import numpy as np
import dendromatics as dm

### Reading the point cloud ###
entr = laspy.read(filename_las)
coords = np.vstack((entr.x, entr.y, entr.z)).transpose()
#---------------------------------------------------------#

### Normalizing the point cloud ###
clean_points = dm.clean_ground(coords)
cloth_nodes = dm.generate_dtm(clean_points)

z0_values = dm.normalize_heights(coords, dtm)
coords = np.append(coords, np.expand_dims(z0_values, axis = 1), 1)

# OPTIONAL: Reducing noise from the dtm and saving it #
dtm = clean_cloth(cloth_nodes)    

las_dtm_points = laspy.create(point_format = 2, file_version='1.2')
las_dtm_points.x = dtm[:, 0]
las_dtm_points.y = dtm[:, 1]
las_dtm_points.z = dtm[:, 2]
las_dtm_points.write(filename_las + "_dtm_points.las")
#---------------------------------------------------------#

### Extracting the stripe and peeling the stems ###
upper_limit = 2.5
lower_limit = 0.5

stripe = coords[(coords[:, 3] > 0.5) & (coords[:, 3] < 2.5), 0:4]
stripe_stems = dm.verticality_clustering(stripe, float(config['advanced']['verticality_scale_stripe']), float(config['advanced']['verticality_thresh_stripe']), float(config['advanced']['epsilon_stripe']), int(config['advanced']['number_of_points']), int(config['basic']['number_of_iterations']), float(config['advanced']['res_xy_stripe']), float(config['advanced']['res_z_stripe']), n_digits)       
#---------------------------------------------------------#

### Individualizing the trees ###
assigned_cloud, tree_vector, tree_heights = dm.individualize_trees(coords, stripe_stems, float(config['advanced']['res_xy']), float(config['advanced']['res_z']), float(config['advanced']['maximum_d']), float(config['advanced']['height_range']), int(config['advanced']['minimum_points']), float(config['intermediate']['distance_to_axis']), float(config['advanced']['maximum_dev']), filename_las, float(config['advanced']['res_heights']), tree_id_field = -1)     

# Saving the assigned cloud, the stems and the tree heights

# Assigned cloud
entr.add_extra_dim(laspy.ExtraBytesParams(name="dist_axes", type=np.float64))
entr.add_extra_dim(laspy.ExtraBytesParams(name="tree_ID", type=np.int32))
entr.dist_axes = assigned_cloud[:, 5]
entr.tree_ID = assigned_cloud[:, 4]
entr.write(filename_las[:-4]+"_tree_ID_dist_axes.las")

# Stems
las_stripe = laspy.create(point_format = 2, file_version='1.2')
las_stripe.x = stripe_stems[:, 0]
las_stripe.y = stripe_stems[:, 1]
las_stripe.z = stripe_stems[:, 2]

las_stripe.add_extra_dim(laspy.ExtraBytesParams(name = "tree_ID", type = np.int32))
las_stripe.tree_ID = stripe_stems[:, -1]
las_stripe.write(filename_las[:-4]+"_stripe.las")

# Tree heights
las_tree_heights = laspy.create(point_format = 2, file_version='1.2')
las_tree_heights.x = tree_heights[:, 0] # x
las_tree_heights.y = tree_heights[:, 1] # y
las_tree_heights.z = tree_heights[:, 2] # z
las_tree_heights.add_extra_dim(laspy.ExtraBytesParams(name = "z0", type = np.int32))
las_tree_heights.z0 = tree_heights[:, 3] # z0
las_tree_heights.add_extra_dim(laspy.ExtraBytesParams(name = "deviated", type = np.int32))
las_tree_heights.deviated = tree_heights[:, 4] # vertical deviation binary indicator
las_tree_heights.write(filename_las[: -4] + "_tree_heights.las")
#---------------------------------------------------------#

### Extracting and curating stems ###
min_h = 0.5 
max_h = 25
section_width = 0.02

xyz0_coords = assigned_cloud[(assigned_cloud[:, 5] < expected_R) & (assigned_cloud[:, 3] > min_h) & (assigned_cloud[:,3] < max_h + section_width),:]

stems = dm.verticality_clustering(xyz0_coords, vert_scale_stems, vert_threshold_stems, eps_stems, n_points_stems, n_iter_stems, resolution_xy_stripe, resolution_z_stripe, n_digits)[:, 0:6]
#---------------------------------------------------------#

### Computing sections ###
section_len = 0.2
sections = np.arange(min_h, max_h, section_len) # Range of uniformly spaced values within the specified interval 

X_c, Y_c, R, check_circle, second_time, sector_perct, n_points_in = dm.compute_sections(stems, sections)
#---------------------------------------------------------#

### Tilt detection ###
np.seterr(divide='ignore', invalid='ignore')
outliers = tilt_detection(X_c, Y_c, R, sections, w_1 = 3, w_2 = 1)
np.seterr(divide='warn', invalid='warn')
#---------------------------------------------------------#

### Drawing sections and axes ###
dm.draw_circles(X_c, Y_c, R, sections, check_circle, sector_perct, n_points_in, tree_vector, outliers, R_min, R_max, threshold, n_sectors, min_n_sectors, filename_las, circa_points)
dm.draw_axes(tree_vector, line_downstep, line_upstep, stripe_lower_limit, stripe_upper_limit, point_interval, filename_las)
#---------------------------------------------------------#

### Computing DBH and tree locators
dbh_values, tree_locations = dm.tree_locator(sections, X_c, Y_c, tree_vector, sector_perct, R, n_points_in, threshold, outliers, filename_las, X_field = 0, Y_field = 1, Z_field = 2)

las_tree_locations = laspy.create(point_format = 2, file_version = '1.2')
las_tree_locations.x = tree_locations[:, 0]
las_tree_locations.y = tree_locations[:, 1]
las_tree_locations.z = tree_locations[:, 2]

las_tree_locations.write(filename_las[:-4] + "_tree_locator.las")
#---------------------------------------------------------#