### Example script ###

import os

import laspy
import numpy as np

import dendromatics as dm

### Reading the point cloud ###
os.chdir("C:/path/data")  # your path to data
filename_las = "example_data.las"  # your .las file
entr = laspy.read(filename_las)
coords = np.vstack((entr.x, entr.y, entr.z)).transpose()
# coords = np.vstack((entr.x, entr.y, entr.z, entr[field_name_z0])).transpose() for already height-normalized point clouds
# ---------------------------------------------------------#

### Normalizing the point cloud ###
clean_points = dm.clean_ground(coords)
cloth_nodes = dm.generate_dtm(clean_points)

z0_values = dm.normalize_heights(coords, cloth_nodes)
coords = np.append(coords, np.expand_dims(z0_values, axis=1), 1)

# OPTIONAL: Reducing noise from the dtm and saving it #
dtm = dm.clean_cloth(cloth_nodes)

las_dtm_points = laspy.create(point_format=2, file_version="1.2")
las_dtm_points.x = dtm[:, 0]
las_dtm_points.y = dtm[:, 1]
las_dtm_points.z = dtm[:, 2]
las_dtm_points.write(filename_las[:, -4] + "_dtm_points.las")
# ---------------------------------------------------------#

### Extracting the stripe and peeling the stems ###
upper_limit = 2.5
lower_limit = 0.5

stripe = coords[(coords[:, 3] > lower_limit) & (coords[:, 3] < upper_limit)]
stripe_stems = dm.verticality_clustering(stripe)
# ---------------------------------------------------------#

### Individualizing the trees ###
assigned_cloud, tree_vector, tree_heights = dm.individualize_trees(coords, stripe_stems)

# Saving the assigned cloud, the stems and the tree heights

# Assigned cloud
entr.add_extra_dim(laspy.ExtraBytesParams(name="dist_axes", type=np.float64))
entr.add_extra_dim(laspy.ExtraBytesParams(name="tree_ID", type=np.int32))
entr.dist_axes = assigned_cloud[:, 5]
entr.tree_ID = assigned_cloud[:, 4]
entr.write(filename_las[:, -4] + "_tree_ID_dist_axes.las")

# Stems
las_stripe = laspy.create(point_format=2, file_version="1.2")
las_stripe.x = stripe_stems[:, 0]
las_stripe.y = stripe_stems[:, 1]
las_stripe.z = stripe_stems[:, 2]

las_stripe.add_extra_dim(laspy.ExtraBytesParams(name="tree_ID", type=np.int32))
las_stripe.tree_ID = stripe_stems[:, -1]
las_stripe.write(filename_las[:, -4] + "_stripe.las")

# Tree heights
las_tree_heights = laspy.create(point_format=2, file_version="1.2")
las_tree_heights.x = tree_heights[:, 0]  # x
las_tree_heights.y = tree_heights[:, 1]  # y
las_tree_heights.z = tree_heights[:, 2]  # z
las_tree_heights.add_extra_dim(laspy.ExtraBytesParams(name="z0", type=np.int32))
las_tree_heights.z0 = tree_heights[:, 3]  # z0
las_tree_heights.add_extra_dim(laspy.ExtraBytesParams(name="deviated", type=np.int32))
las_tree_heights.deviated = tree_heights[:, 4]  # vertical deviation binary indicator
las_tree_heights.write(filename_las[:, -4] + "_tree_heights.las")
# ---------------------------------------------------------#

### Extracting and curating stems ###

expected_R = 0.5
min_h = 0.5
max_h = 25
section_width = 0.02

xyz0_coords = assigned_cloud[
    (assigned_cloud[:, 5] < expected_R)
    & (assigned_cloud[:, 3] > min_h)
    & (assigned_cloud[:, 3] < max_h + section_width),
    :,
]

stems = dm.verticality_clustering(xyz0_coords)[:, 0:6]
# ---------------------------------------------------------#

### Computing sections ###
section_len = 0.2
sections = np.arange(min_h, max_h, section_len)  # Range of uniformly spaced values within the specified interval

X_c, Y_c, R, check_circle, second_time, sector_perct, n_points_in = dm.compute_sections(stems, sections)
# ---------------------------------------------------------#

### Tilt detection ###
outliers = dm.tilt_detection(X_c, Y_c, R, sections)
# ---------------------------------------------------------#

### Drawing sections and axes ###
dm.draw_circles(
    X_c,
    Y_c,
    R,
    sections,
    check_circle,
    sector_perct,
    n_points_in,
    tree_vector,
    outliers,
    filename_las,
)
dm.draw_axes(tree_vector, filename_las)
# ---------------------------------------------------------#

### Computing DBH and tree locators
dbh_values, tree_locations = dm.tree_locator(sections, X_c, Y_c, tree_vector, sector_perct, R, outliers, n_points_in)

las_tree_locations = laspy.create(point_format=2, file_version="1.2")
las_tree_locations.x = tree_locations[:, 0]
las_tree_locations.y = tree_locations[:, 1]
las_tree_locations.z = tree_locations[:, 2]

las_tree_locations.write(filename_las[:, -4] + "_tree_locator.las")
# ---------------------------------------------------------#
