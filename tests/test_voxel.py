import numpy as np

import dendromatics as dm


def test_voxelate_with_n_points_true():
    # Right triangle of 3 points per side; 6 points total
    cloud = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 3.0, 3.0],
            [0.0, 0.0, 6.0],
        ]
    )

    res_xy, res_z = 4.0, 4.0

    # Expects right triangle of 2 points per side; 3 points total (some points collapsed into voxels)
    # Also expects a 4th column indicating the number of points within each voxel (4 should collapse into 1st voxel)
    expect = np.array(
        [
            [2.0, 2.0, 2.0, 4],
            [2.0, 6.0, 2.0, 1],
            [2.0, 2.0, 6.0, 1],
        ]
    )

    output, _, _ = dm.voxelate(cloud, res_xy, res_z, with_n_points=True)

    np.testing.assert_array_almost_equal(output, expect)


def test_voxelate_with_n_points_false():
    # Right triangle of 3 points per side; 6 points total
    cloud = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 3.0, 3.0],
            [0.0, 0.0, 6.0],
        ]
    )

    res_xy, res_z = 4.0, 4.0

    # Expects right triangle of 2 points per side; 3 points total (some points collapsed into voxels)
    # Also expects a 4th column indicating the number of points within each voxel (4 should collapse into 1st voxel)
    expect = np.array(
        [
            [2.0, 2.0, 2.0],
            [2.0, 6.0, 2.0],
            [2.0, 2.0, 6.0],
        ]
    )

    output, _, _ = dm.voxelate(cloud, res_xy, res_z, with_n_points=False)

    np.testing.assert_array_almost_equal(output, expect)


def test_voxelate_low_res():
    # Right triangle of 3 points per side; 6 points total
    cloud = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 3.0, 3.0],
            [0.0, 0.0, 6.0],
        ]
    )

    # Resolution is lower than distance between some points
    res_xy, res_z = 1.5, 1.5

    # Expects right triangle of 3 points per side; 6 points total (no points collapsed into voxels)
    # Also expects a 4th column indicating the number of points within each voxel (1 point per voxel)
    expect = np.array(
        [
            [0.75, 0.75, 0.75, 1],
            [0.75, 3.75, 0.75, 1],
            [0.75, 6.75, 0.75, 1],
            [0.75, 0.75, 3.75, 1],
            [0.75, 3.75, 3.75, 1],
            [0.75, 0.75, 6.75, 1],
        ]
    )

    output, _, _ = dm.voxelate(cloud, res_xy, res_z)

    np.testing.assert_array_almost_equal(output, expect)


def test_voxelate_high_res():
    # Right triangle of 3 points per side; 6 points total
    cloud = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 3.0, 3.0],
            [0.0, 0.0, 6.0],
        ]
    )

    # Resolution is higher than distance between any 2 points
    res_xy, res_z = 7.0, 7.0

    # Expects just 1 voxel; (all points collapsed into it)
    # Also expects a 4th column indicating the number of points within each voxel (all 6 should collapse into 1 voxel)
    expect = np.array([[3.5, 3.5, 3.5, 6]])

    output, _, _ = dm.voxelate(cloud, res_xy, res_z)

    np.testing.assert_array_almost_equal(output, expect)


def test_voxelate_edge_res():
    # Right triangle of 3 points per side; 6 points total
    cloud = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 3.0, 3.0],
            [0.0, 0.0, 6.0],
        ]
    )

    # Edge resolution: resolution is equal than distance between closest points
    res_xy, res_z = 3.0, 3.0

    # Expects right triangle of 3 points per side; 6 points total (no points collapsed into voxels)
    # Also expects a 4th column indicating the number of points within each voxel (1 point per voxel)
    expect = np.array(
        [
            [1.5, 1.5, 1.5, 1],
            [1.5, 4.5, 1.5, 1],
            [1.5, 7.5, 1.5, 1],
            [1.5, 1.5, 4.5, 1],
            [1.5, 4.5, 4.5, 1],
            [1.5, 1.5, 7.5, 1],
        ]
    )

    output, _, _ = dm.voxelate(cloud, res_xy, res_z)

    np.testing.assert_array_almost_equal(output, expect)
