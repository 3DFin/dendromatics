#### IMPORTS ####
import timeit

import numpy as np

# -----------------------------------------------------------------------------------------------------------------------------------
# voxelate
# ----------------------------------------------------------------------------------------------------------------------------------------


def voxelate(
    cloud,
    resolution_xy,
    resolution_z,
    n_digits=5,
    X_field=0,
    Y_field=1,
    Z_field=2,
    with_n_points=True,
    verbose=True,
):
    """Function used to voxelate point clouds. It allows to use a different
    resolution for (z), but (x, y) will always share the same resolution. It
    also allows to revert the process, by creating a unique code for each point
    in the point cloud, thus voxelated cloud can be seamlessly reverted to the
    original point cloud.

    Parameters
    ----------
    cloud : numpy.ndarray
        The point cloud to be voxelated. It is expected to have X, Y, Z fields.
        3D or higher array containing data with `float` type.
    resolution_xy : float
        (x, y) voxel resolution.
    resolution_z : float
        (z) voxel resolution.
    n_digits : int
        Number of digits dedicated to each coordinate ((x), (y) or (z)) during
        the generation of each point code. Defaults to 5.
    X_field : int
        Index at which (x) coordinate is stored. Defaults to 0.
    Y_field : int
        Index at which (y) coordinate is stored. Defaults to 1.
    Z_field : int
        Index at which (z) coordinate is stored. Defaults to 2.
    with_n_points : boolean
        If True, output voxelated cloud will have a field including the number
        of points that each voxel contains. Defaults to True.

    Returns
    -------
    voxelated_cloud : numpy.ndarray
        The voxelated cloud. It consists of 3 columns, each with (x), (y) and
        (z) coordinates, and an optional 4th column having the number of points
        included in each voxel if with_n_points = True.
    vox_to_cloud_ind : numpy.ndarray
        Vector containing the indexes to revert to the original point cloud
        from the voxelated cloud.
    cloud_to_vox_ind : numpy.ndarray
        Vector containing the indexes to directly go from the original point
        cloud to the voxelated cloud.
    """

    t = timeit.default_timer()

    # The coordinate minima
    cloud_min = np.min(cloud[:, [X_field, Y_field, Z_field]], axis=0)

    # Substraction of the coordinates
    cloud[:, X_field] = cloud[:, X_field] - cloud_min[0]
    cloud[:, Y_field] = cloud[:, Y_field] - cloud_min[1]
    cloud[:, Z_field] = cloud[:, Z_field] - cloud_min[2]

    if verbose:
        elapsed = timeit.default_timer() - t
        print("      -Voxelization")
        print(
            "        ",
            "Voxel resolution:",
            "{:.2f}".format(resolution_xy),
            "x",
            "{:.2f}".format(resolution_xy),
            "x",
            "{:.2f}".format(resolution_z),
            "m",
        )
        print("        ", "%.2f" % elapsed, "s: scaling and translating")

    # Generation of 'pixel code'. It provides each point with an unique identifier.
    code = (
        np.floor(cloud[:, Z_field] / resolution_z) * 10 ** (n_digits * 2)
        + np.floor(cloud[:, Y_field] / resolution_xy) * 10**n_digits
        + np.floor(cloud[:, X_field] / resolution_xy)
    )

    if not verbose:
        elapsed = timeit.default_timer() - t
        print("        ", "%.2f" % elapsed, "s: encoding")

    # Vector that contains the ordered code. It will be used to sort the code to
    # then sort the cloud.
    vox_order_ind = np.argsort(code)

    if not verbose:
        elapsed = timeit.default_timer() - t
        print("        ", "%.2f" % elapsed, "s: 1st sorting")

    # Vector that contains the indexes of said code. It will be used to restore
    # the order of points within the original cloud.
    vox_order_ind_inverse = np.argsort(vox_order_ind)

    # Sorted code.
    code = code[vox_order_ind]

    if not verbose:
        elapsed = timeit.default_timer() - t
        print("        ", "%.2f" % elapsed, "s: 2nd sorting")

    # Unique values of said 'pixel code':
    # unique_code: Unique values. They contain codified coordinates of which will
    #   later be the voxel centroids.
    # vox_first_point_id. Index corresponding to the point of each voxel that
    #   corresponds to the first point, among those in the same voxel, in the original cloud.
    # inverse_id: Indexes that allow to revert the voxelization.
    # vox_points: Number of points in each voxel
    unique_code, vox_first_point_id, inverse_id, vox_points = np.unique(
        code, return_index=True, return_inverse=True, return_counts=True
    )

    if not verbose:
        elapsed = timeit.default_timer() - t
        print("        ", "%.2f" % elapsed, "s: extracting uniques values")

    # Indexes that directly associate each voxel to its corresponding points in
    # the original cloud (unordered)
    vox_to_cloud_ind = inverse_id[vox_order_ind_inverse]

    # Indexes that directly associate each point in the original, unordered cloud
    # to its corresponding voxel
    cloud_to_vox_ind = vox_order_ind[vox_first_point_id]

    # Empty array to be filled with voxel coordinates
    voxelated_cloud = np.zeros((np.size(unique_code, 0), 3))

    # Each coordinate 'pixel code'. They will then be transformed into coordinates
    z_code = np.floor(unique_code / 10 ** (n_digits * 2))
    y_code = np.floor((unique_code - z_code * 10 ** (n_digits * 2)) / 10**n_digits)
    x_code = unique_code - z_code * 10 ** (n_digits * 2) - y_code * 10**n_digits

    voxelated_cloud[:, 0] = x_code
    voxelated_cloud[:, 1] = y_code
    voxelated_cloud[:, 2] = z_code

    if not verbose:
        elapsed = timeit.default_timer() - t
        print("        ", "%.2f" % elapsed, "s: decomposing code")

    # Transformation of x, z, y codes into X, Y, X voxel coordinates, by scaling,
    # translating and centering.
    voxelated_cloud[:, 2] = voxelated_cloud[:, 2] * resolution_z + cloud_min[2] + resolution_z / 2
    voxelated_cloud[:, 1] = voxelated_cloud[:, 1] * resolution_xy + cloud_min[1] + resolution_xy / 2
    voxelated_cloud[:, 0] = voxelated_cloud[:, 0] * resolution_xy + cloud_min[0] + resolution_xy / 2

    # Boolean parameter that includes or not a 4th column with the number of
    # points in each voxel
    if with_n_points is True:
        voxelated_cloud = np.append(voxelated_cloud, vox_points[:, np.newaxis], axis=1)

    if not verbose:
        elapsed = timeit.default_timer() - t
        print("        ", "%.2f" % elapsed, "s: rescaling and translating back")
        print(
            "        ",
            "{:.2f}".format(vox_to_cloud_ind.shape[0] / 1000000),
            "million points ->",
            "{:.2f}".format(cloud_to_vox_ind.shape[0] / 1000000),
            "million voxels",
        )
        print(
            "        ",
            "Voxels account for",
            "{:.2f}".format(cloud_to_vox_ind.shape[0] * 100 / vox_to_cloud_ind.shape[0]),
            "% of original points",
        )

    cloud[:, X_field] = cloud[:, X_field] + cloud_min[0]
    cloud[:, Y_field] = cloud[:, Y_field] + cloud_min[1]
    cloud[:, Z_field] = cloud[:, Z_field] + cloud_min[2]

    return voxelated_cloud, vox_to_cloud_ind, cloud_to_vox_ind


# monkey patch voxelization if possible
try:
    print("Using dendromatic with optimized C++ voxelization")
    import dendroptimized

    voxelate = dendroptimized.voxelize
except ImportError:
    pass
