import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from .voxel.voxel import *

# -----------------------------------------------------------------------------
# compute_axes
# -----------------------------------------------------------------------------


def compute_axes(
    voxelated_cloud,
    clust_stripe,
    stripe_lower_limit,
    stripe_upper_limit,
    h_range,
    min_points,
    d_max,
    X_field,
    Y_field,
    Z_field,
    Z0_field,
    tree_id_field,
    progress_hook=None,
):
    """Function used inside individualize_trees during tree individualization
    process. It identifies tree axes. It expects a voxelated version of the
    point cloud and a filtered (based on the verticality clustering process)
    stripe as input, so that it only contains (hopefully) stems. Those stems
    are isolated and enumerated, and then, their axes are identified using PCA
    (PCA1 direction). This allows to group points based on their distance to
    those axes, thus assigning each point to a tree.

    It requires a height-normalized cloud in order to function properly.

    Parameters
    ----------
    voxelated_cloud : numpy.ndarray
        The voxelated point cloud. It is expected to have X, Y, Z and Z0 fields.
    clust_stripe : numpy.ndarray
        The point cloud containing the clusterized stripe (the stems) from
        verticality_clustering.
    stripe_lower_limit : float
        Lower (vertical) limit of the stripe (units is meters). Defaults to 0.7.
    stripe_upper_limit : float
        Upper (vertical) limit of the stripe (units is meters). Defaults to 3.5.
    h_range : float
        Only stems where points extend vertically throughout a range as tall as
        defined by h_range are considered.
    min_points : int
        Minimum number of points in a cluster for it to be considered as a
        potential stem
    d_max : float
        Points that are closer than d_max to an axis are assigned to that axis.
    X_field : int
        Index at which (x) coordinates are stored.
    Y_field : int
        Index at which (y) coordinates are stored.
    Z_field : int
        Index at which (z) coordinates are stored.
    Z0_field : int
        Index at which (z0) coordinates are stored.
    tree_id_field : int
        Index at which cluster ID is stored.
    progress_hook : callable, optional
        A hook that take two int, the first is the current number of iteration
        and the second is the targetted number iteration. Defaults to None.

    Returns
    -------
    detected_trees : numpy.ndarray
        Matrix with as many rows as trees, containing a description of each
        individualized tree. It stores the following values: tree ID, PCA1 X
        value, PCA1 Y value, PCA1 Z value, stem centroid X value, stem centroid
        Y value, stem centroid Z value, height difference of stem centroid
        (z - z0), axis vertical deviation.
    dist_to_axis : numpy.ndarray
        Matrix containing the distance from each point to the closest axis.
    tree_id_vector : numpy.ndarray
        Vector containing the tree IDs.
    """
    # Empty vectors that will store final outputs: - distance from each point to
    # closest axis - ID of the corresponding tree (the tree that the point belongs to).
    dist_to_axis = (
        np.zeros((np.size(voxelated_cloud, 0))) + 100000
    )  # distance to the closest axis
    tree_id_vector = (
        np.zeros((np.size(voxelated_cloud, 0))) + 100000
    )  # tree ID of closest axis

    # Set of all possible trees (trunks at this stage) and number of points associated to each:
    unique_values, n = np.unique(clust_stripe[:, tree_id_field], return_counts=True)

    # Filtering of possible trees that do not contain enough points to be considered.
    filt_unique_values = unique_values[n > min_points]

    # Final number of trees
    n_values = np.size(filt_unique_values)

    # Empty array to be filled with several descriptors of the trees. In the following order:
    # tree ID | PCA1 X value | PCA1 Y value | PCA1 Z value | trunk centroid X value
    # | trunk centroid Y value | trunk centroid Z value | height difference |
    # It has as many rows as trees are.

    detected_trees = np.zeros((np.size(filt_unique_values, 0), 9))

    # Index used to display progress information.
    # we use this index because filt_unique_values contains tree_id that
    # could be non contiguous
    id_progress = 0
    # Index used to keep track of valid trees and index the detected_trees array.
    id_valid = 0

    # Height range (actual value, not the %) that points should extend throughout
    h_range_value = (stripe_upper_limit - stripe_lower_limit) * h_range

    # First loop: It goes over each tree (still stems) except for the first entry,
    # which maps to noise (this entry is generated by DBSCAN during clustering).

    if progress_hook is not None:
        progress_hook(0, n_values)
    for i in filt_unique_values:
        # Isolation of stems: stem_i only contains points associated to 1 tree.
        stem_i = clust_stripe[clust_stripe[:, tree_id_field] == i][
            :, [X_field, Y_field, Z_field]
        ]

        # Z and Z0 mean heights of points in a given tree
        z_z0 = np.average(
            clust_stripe[clust_stripe[:, tree_id_field] == i][:, [Z_field, Z0_field]],
            axis=0,
        )

        # Difference between Z and Z0 mean heights
        diff_z_z0 = z_z0[0] - z_z0[1]

        # Second loop: only stems where points extend vertically throughout its
        # whole range are considered.
        if np.ptp(stem_i[:, Z_field]) > (h_range_value):
            # PCA and centroid computation.
            pca_out = PCA(n_components=3)
            pca_out.fit(stem_i)
            centroid = np.mean(stem_i, 0)

            # Values are stored in tree vector
            detected_trees[id_valid, 0] = i  # tree ID
            detected_trees[id_valid, 1:4] = pca_out.components_[
                0, :
            ]  # PCA1 X value | PCA1 Y value | PCA1 Z value
            detected_trees[id_valid, 4:7] = (
                centroid  # stem centroid X value | stem centroid Y value | stem centroid Z value
            )
            detected_trees[id_valid, 7] = diff_z_z0  # Height difference
            detected_trees[id_valid, 8] = np.abs(
                np.arctan(
                    np.sqrt(
                        detected_trees[id_valid, 1] ** 2
                        + detected_trees[id_valid, 2] ** 2
                    )
                    / detected_trees[id_valid, 3]
                )
                * 180
                / np.pi
            )

            # Coordinate transformation from original to PCA. Done for EVERY
            # point of the original cloud from the PCA of a SINGLE stem.
            cloud_pca_coords = pca_out.transform(
                voxelated_cloud[:, [X_field, Y_field, Z_field]]
            )

            # Distance from every point in the new coordinate system to the axes.
            # It is directly computed from the quadratic component of PC2 and PC3
            axis_dist = np.hypot(cloud_pca_coords[:, 1], cloud_pca_coords[:, 2])

            # Points that are closer than d_max to an axis are assigned to that axis.
            # Also, if a point is closer to an axis than it was to previous axes, accounting for points
            # that were previously assigned to some other axis in previous iterations, it is assigned
            # to the new, closer axis. Distance to the axis is stored as well.
            valid_points = (axis_dist < d_max) & ((axis_dist - dist_to_axis) < 0)
            tree_id_vector[valid_points] = i
            dist_to_axis[valid_points] = axis_dist[valid_points]

            # Progress hook
            id_valid = id_valid + 1
            id_progress = id_progress + 1
            if progress_hook is not None:
                progress_hook(id_progress, n_values)
        else:
            # we keep track of the progres even if the tree is not valid
            id_progress = id_progress + 1
            if progress_hook is not None:
                progress_hook(id_progress, n_values)
    # This deletes the trailing rows that only contains zeros
    detected_trees = detected_trees[~np.all(detected_trees == 0, axis=1)]
    return (detected_trees, dist_to_axis, tree_id_vector)


# -----------------------------------------------------------------------------
# compute_heights
# -----------------------------------------------------------------------------


def compute_heights(
    voxelated_cloud,
    detected_trees,
    dist_to_axis,
    tree_id_vector,
    d,
    max_dev,
    resolution_heights,
    n_digits,
    X_field,
    Y_field,
    Z_field,
):
    """Function used inside individualize_trees during tree individualization
    process. It measures tree heights. The function creates a large-resolution
    voxel cloud to and filters voxels containing few points. This has the
    purpose to discard any outlier point that might be over the trees, to then
    identify the highest point within the remaining voxels.

    It requires a height-normalized cloud in order to function properly.

    Parameters
    ----------
    voxelated_cloud : numpy.ndarray
        The voxelated point cloud. It is expected to have X, Y, Z and Z0 fields.
    detected_trees : numpy.ndarray
        See compute_axes.
    dist_to_axis : numpy.ndarray
        See compute_axes.
    tree_id_vector : numpynd.array
        See compute_axes.
    d : float
        Points within this distance from tree axis will be considered as
        potential points to define tree height.
    max_dev : float
        Maximum degree of vertical deviation of a tree axis to consider its
        tree height measurement as valid.
    resolution_heights : float
        (x, y, z) voxel resolution.
    n_digits : int
        Number of digits dedicated to each coordinate ((x), (y) or (z)) during
        the generation of each point code.
    X_field : int
        Index at which (x) coordinates are stored.
    Y_field : int
        Index at which (y) coordinates are stored.
    Z_field : int
        Index at which (z) coordinates are stored.

    Returns
    -------
    tree_heights: numpy.ndarray
        Matrix containing (x, y, z) coordinates of each tree's highest point,
        as well as its normalized height and a binary field stating if the
        axis was deviated (1) or if it was not (0).
    """

    # The cloud is re-voxelated to a larger resolution to then be clusterized.
    # Small clusters containing 1-2 voxels will be discarded to eliminate
    # outliers points that could interfere in height measurement.
    large_voxels_cloud, large_vox_to_cloud_ind, _ = voxelate(
        voxelated_cloud,
        resolution_heights,
        resolution_heights,
        n_digits,
        X_field,
        Y_field,
        Z_field,
        with_n_points=False,
    )

    # eps for DBSCAN
    eps_heights = resolution_heights * 1.9

    # Large-resolution voxelated cloud is clusterized
    clustering = DBSCAN(eps=eps_heights, min_samples=2, n_jobs=-1).fit(
        large_voxels_cloud
    )

    # Cluster labels are attached to the fine-resolution voxelated cloud
    voxelated_cloud = np.append(
        voxelated_cloud,
        np.expand_dims(clustering.labels_[large_vox_to_cloud_ind], axis=1),
        axis=1,
    )

    # Tree IDS are attached to the fine-resolution voxelated cloud too
    voxelated_cloud = np.append(
        voxelated_cloud, np.expand_dims(tree_id_vector, axis=1), axis=1
    )

    # Eliminating all points too far away from axes
    voxelated_cloud = voxelated_cloud[dist_to_axis < d, :]

    # Set of all cluster labels and their cardinality: cluster_id = {1,...,K},
    # K = 'number of clusters'.
    cluster_id, K = np.unique(clustering.labels_, return_counts=True)

    # Filtering of labels associated only to clusters that contain a minimum number of points.
    large_clusters = cluster_id[K > 3]

    # Discarding points that do not belong to any cluster
    large_clusters = large_clusters[large_clusters != -1]

    # Eliminating all points that belong to clusters with less than 2 points (large voxels)
    voxelated_cloud = voxelated_cloud[np.isin(voxelated_cloud[:, -2], large_clusters)]

    # Only pick the actual trees:
    n_trees = detected_trees.shape[0]
    tree_heights = np.zeros((n_trees, 5))

    for i in range(n_trees):  # Last row of tree_vector
        # Be aware this finds the highest voxel (fine-resolution), not the highest point.
        valid_id = detected_trees[i, 0]
        single_tree = voxelated_cloud[
            voxelated_cloud[:, -1] == valid_id, 0:3
        ]  # Just the (x, y, z) coordinates
        which_z_max = np.argmax(single_tree[:, 2])  # The highest (z) value
        highest_point = single_tree[which_z_max, :]  # The highest point
        tree_heights[i, 0:3] = highest_point
        tree_heights[i, 3] = highest_point[2] - detected_trees[i, 7]  # (z0)

        # If tree is deviated from vertical, 1, else, 0.
        if detected_trees[i, -1] > [max_dev]:
            tree_heights[i, -1] = 0
        else:
            tree_heights[i, -1] = 1

    return tree_heights


# -----------------------------------------------------------------------------
# individualize_trees
# -----------------------------------------------------------------------------


def individualize_trees(
    cloud,
    clust_stripe,
    resolution_z=0.035,
    resolution_xy=0.035,
    stripe_lower_limit=0.7,
    stripe_upper_limit=3.5,
    h_range=1,
    d_max=1.5,
    min_points=20,
    d=15,
    max_dev=25,
    resolution_heights=0.3,
    n_digits=5,
    X_field=0,
    Y_field=1,
    Z_field=2,
    Z0_field=3,
    tree_id_field=-1,
    progress_hook=None,
):
    """This function expects a filtered (based on the clustering process)
    stripe as input, so that it only contains (hopefully) stems. Those stems
    are voxelated and enumerated, and then, their axes are identified using PCA
    (PCA1 direction). This allows to group points based on their distance to
    those axes, thus assigning each point to a tree. This last step is applied
    to the WHOLE original cloud. It also measures tree heights.

    It requires a height-normalized cloud in order to function properly.

    Parameters
    ----------
    cloud : numpy.ndarray
        The point cloud. It is expected to have X, Y, Z and Z0 fields.
    clust_stripe : numpy.ndarray.
        The point cloud containing the clusterized stripe from
        verticality_clustering. It is expected to have X, Y, Z0 and cluster ID
        fields.
    resolution_z : float
        (x, y) voxel resolution in meters. Defaults to 0.035.
    resolution_xy : float
        (z) voxel resolution in meters. Defaults to 0.035.
    stripe_lower_limit : float
        Lower (vertical) limit of the stripe (units is meters). Defaults to 0.7.
    stripe_upper_limit : float
        Upper (vertical) limit of the stripe (units is meters). Defaults to 3.5.
    h_range : float
        Only stems where points extend vertically throughout a range as tall as
        defined by h_range are considered (units is meters). Defaults to 1.2.
    d_max : float
        Points that are closer than d_max to an axis are assigned to that axis
        (units is meters). Defaults to 1.5.
    min_points : int
        Minimum number of points in a cluster for it to be considered as a
        potential stem. Defaults to 20.
    d : float
        Points within this distance from tree axis will be considered as
        potential points to define tree height (units is meters).
        Defaults to 15.
    max_dev : float
        Maximum degree of vertical deviation of a tree axis to consider its
        tree height measurement as valid (units is sexagesimal degrees).
        Defaults to 25.
    n_digits : int
        Number of digits dedicated to each coordinate ((x), (y) or (z))
        during the generation of each point code. Defaults to 5.
    resolution_heights : float
        (x, y, z) voxel resolution in meters used during height computation.
        Defaults to 0.3
    X_field : int
        Index at which (x) coordinate is stored. Defaults to 0.
    Y_field : int
        Index at which (y) coordinate is stored. Defaults to 1.
    Z_field : int
        Index at which (z) coordinate is stored. Defaults to 2.
    Z0_field : int
        Index at which (z0) coordinate is stored. Defaults to 3.
    tree_id_field : int
        Index at which cluster ID is stored. Defaults to -1.
    progress_hook : callable, optional
        A hook that take two int, the first is the current number of iteration
        and the second is the targetted number iteration. Defaults to None.

    Returns
    -------
    assigned_cloud : numpy.ndarray
        Point cloud containing individualized trees. It consists of
        (x), (y), (z) and (z0) coordinates, a 5th column containing tree ID
        that each point belongs to and a 6th column containing point distance
        to closest axis.
    detected_trees : numpy.ndarray
        Matrix with as many rows as trees, containing a description of each
        individualized tree. It stores tree ID, PCA1 X value, PCA1 Y value,
        PCA1 Z value, stem centroid X value, stem centroid Y value, stem
        centroid Z value, height difference of stem centroid (z - z0), axis
        vertical deviation.
    tree_heights : numpy.ndarray
        Matrix containing the heights of individualized trees. It consists of
        (x), (y), (z) and (z0) coordinates of the highest point of the tree and
        a 5th column containing a binary indicator: 0 - tree was too deviated
        from vertical, and height may not be accurate, or 1 - tree was not too
        deviated from vertical, thus height may be trusted.
    """

    # Whole original cloud voxelization
    voxelated_cloud, vox_to_cloud_ind, _ = voxelate(
        cloud,
        resolution_z,
        resolution_xy,
        n_digits,
        X_field,
        Y_field,
        Z_field,
        with_n_points=False,
    )

    # Call to compute_axes
    detected_trees, dist_to_axis, tree_id_vector = compute_axes(
        voxelated_cloud,
        clust_stripe,
        stripe_lower_limit,
        stripe_upper_limit,
        h_range,
        min_points,
        d_max,
        X_field,
        Y_field,
        Z_field,
        Z0_field,
        tree_id_field,
        progress_hook,
    )

    # Call to compute_heights
    tree_heights = compute_heights(
        voxelated_cloud,
        detected_trees,
        dist_to_axis,
        tree_id_vector,
        d,
        max_dev,
        resolution_heights,
        n_digits,
        X_field,
        Y_field,
        Z_field,
    )

    # Two new fields are added to the original cloud: - tree ID (id of closest axis)
    # - distance to that axis
    assigned_cloud = np.append(
        cloud, tree_id_vector[vox_to_cloud_ind, np.newaxis], axis=1
    )
    assigned_cloud = np.append(
        assigned_cloud, dist_to_axis[vox_to_cloud_ind, np.newaxis], axis=1
    )

    # Output: - Assigned cloud (X, Y, Z, Z0, tree_id, dist_to_axis) - tree vector
    return assigned_cloud, detected_trees, tree_heights
