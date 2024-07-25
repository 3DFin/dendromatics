import numpy as np

import dendromatics as dm


def test_clean_ground():
    # Regular grid of points expanding from (0, 0) to (1, 1)
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)

    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # adding a constant Z field (z = 0) to all points in the grid
    z = np.expand_dims(np.zeros(121), axis=1)

    cloud = np.append(xy, z, axis=1)

    # Adding noise/outlier: point outside the grid, at z = 2
    z_outlier = np.array([[0.5, 0.5, 2]])
    noisy_cloud = np.append(cloud, z_outlier, axis=0)

    # Expect is the regular grid
    expect = np.append(xy, z, axis=1)

    # Output should be the noisy cloud minus the outlier point --> the regular grid
    output = dm.clean_ground(noisy_cloud)

    np.testing.assert_array_almost_equal(output, expect)


def test_clean_cloth_with_noise():
    # Regular grid of points expanding from (0, 0) to (1, 1)
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)

    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # adding a constant Z field (z = 0) to all points in the grid
    z = np.expand_dims(np.zeros(121), axis=1)

    cloud = np.append(xy, z, axis=1)

    # Adding noise/outlier: point outside the grid, at z = 2
    z_outlier = np.array([[0.5, 0.5, 2]])

    noisy_cloud = np.append(cloud, z_outlier, axis=0)

    # Expect is the regular grid
    expect = np.append(xy, z, axis=1)

    # Output should be the noisy cloud minus the outlier point --> the regular grid
    output = dm.clean_cloth(noisy_cloud)

    np.testing.assert_array_almost_equal(output, expect)


def test_clean_cloth_already_is_clean():
    # Regular grid of points expanding from (0, 0) to (1, 1)
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)

    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # adding a constant Z field (z = 0) to all points in the grid
    z = np.expand_dims(np.zeros(121), axis=1)

    cloud = np.append(xy, z, axis=1)

    # Expect is the regular grid
    expect = np.append(xy, z, axis=1)

    # Output should be the regular grid --> no points removed
    output = dm.clean_cloth(cloud)

    np.testing.assert_array_almost_equal(output, expect)


def test_clean_cloth_small_cloth():
    # Regular grid of points expanding from (0, 0) to (1, 1) (ONLY 9 points)
    x = np.linspace(0.0, 1.0, 3)
    y = np.linspace(0.0, 1.0, 3)

    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # adding a constant Z field (z = 0) to all points in the grid
    z = np.expand_dims(np.zeros(9), axis=1)

    cloud = np.append(xy, z, axis=1)

    # function should not work as it performs a 15-nearest-neighbours search.
    np.testing.assert_raises(ValueError, dm.clean_cloth, cloud)


def test_clean_cloth_barely_big_cloth():
    # Regular grid of points expanding from (0, 0) to (1, 1) (16 points)
    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, 1.0, 4)

    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # adding a constant Z field (z = 0) to all points in the grid
    z = np.expand_dims(np.zeros(16), axis=1)

    cloud = np.append(xy, z, axis=1)

    # one points is removed, so there are 15 points
    cloud = cloud[0 : (cloud.shape[0] - 1)]

    np.testing.assert_warns(Warning, dm.clean_cloth, cloud)


def test_normalize_heights():
    # DTM:
    # Regular grid of points expanding from (0, 0) to (1, 1)
    dtm_x = np.linspace(0.0, 1.0, 11)
    dtm_y = np.linspace(0.0, 1.0, 11)
    dtm_xy = np.array(np.meshgrid(dtm_x, dtm_y)).T.reshape(-1, 2)

    # adding a constant Z field (z = 1) to all points in the grid
    dtm_z = np.expand_dims(np.zeros(121), axis=1) + 1
    dtm = np.append(dtm_xy, dtm_z, axis=1)

    # Cloud:
    cloud_x = np.expand_dims(np.linspace(0.0, 1.0, 11), axis=1)
    cloud_y = np.expand_dims(np.zeros(11), axis=1)

    cloud_xy = np.append(cloud_x, cloud_y, axis=1)

    cloud_z = np.expand_dims(np.linspace(1.0, 2.0, 11), axis=1)

    # Diagonal line from (0, 0, 1) to (1, 0, 2)
    cloud = np.append(cloud_xy, cloud_z, axis=1)

    # the normalized heights should be cloud_z - 1 for every point
    expect = np.linspace(0.0, 1.0, 11)
    output = dm.normalize_heights(cloud, dtm)

    np.testing.assert_array_almost_equal(output, expect)
