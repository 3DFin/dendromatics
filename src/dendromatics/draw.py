import laspy
import numpy as np

# -----------------------------------------------------------------------------
# draw_circles
# -----------------------------------------------------------------------------


def generate_circles_cloud(
    X_c,
    Y_c,
    R,
    sections,
    check_circle,
    sector_perct,
    n_points_in,
    tree_vector,
    outliers,
    R_min=0.03,
    R_max=0.5,
    threshold=5,
    n_sectors=16,
    min_n_sectors=9,
    circa_points=200,
):
    """This function generates points that comprise the circles computed by
    fit_circle_check function, so sections can be visualized. The circles
    points cloud along with their associated meta data are returned as a Matrix
    (numpy.ndarray)

    Parameters
    ----------
    X_c : numpy.ndarray
        Matrix containing (x) coordinates of the center of the sections.
    Y_c : numpy.ndarray
        Matrix containing (y) coordinates of the center of the sections.
    R : numpy.ndarray
        Vector containing section radia.
    sections : numpy.ndarray
        Vector containing section heights (normalized heights).
    section_perct : numpy.ndarray
        Matrix containing the percentage of occupied sectors.
    n_points_in : numpy.ndarray
        Matrix containing the number of points in the inner circumferences.
    tree_vector : numpy.ndarray
        detected_trees output from individualize_trees.
    outliers : numpy.ndarray
        Vector containing the 'outlier probability' of each section.
    R_min : float
        Refer to fit_circle_check in 'sections' module. Defaults to 0.03.
    R_max : float
        Refer to fit_circle_check in 'sections' module. Defaults to 0.5.
    threshold : float
        Refer to fit_circle_check in 'sections' module. Defaults to 5.
    n_sectors : int
        Refer to fit_circle_check in 'sections' module. Defaults to 16.
    min_n_sectors: int
        Refer to fit_circle_check in sections module. Defaults to 9.
    circa_points : int
        Number of points used to draw each circle. Defaults to 200.

    Returns
    -------
    coords : numpy.ndarray
        Matrix containing the circles coordinates and their associated meta data
    """

    # Empty vector to be filled. It has as many elements as the vector containing
    # the center of the circles for a given tree.
    tree_section = X_c.shape

    # Empty array that will contain the information about each section, to then
    # be used to complete the .LAS file data.
    section_c_xyz = np.zeros([tree_section[0] * tree_section[1], 9])

    # Auxiliary index indicating which section is in use.
    section = 0

    # Double for loop to iterate through each combination of coordinates
    for i in range(tree_section[0]):
        for j in range(tree_section[1]):
            # If distance is within range (R_min, R_max), then proceed.
            if R[i, j] >= R_min and R[i, j] <= R_max:
                # Filling the array with the appropriate data
                section_c_xyz[section, :] = [
                    X_c[i, j],
                    Y_c[i, j],
                    sections[j] + tree_vector[i, 7],
                    R[i, j],
                    check_circle[i, j],
                    sector_perct[i, j],
                    n_points_in[i, j],
                    sections[j],
                    outliers[i, j],
                ]

                section = section + 1

    # Just the centers of each filled section
    centers = section_c_xyz[:section, :]

    # Number of centers
    n = centers.shape[0]

    # Empty vector to be filled with the coordinates of each circle.
    coords = np.zeros((circa_points * n, 11))

    # User-create function to tranform polar coordinates to cartesian coordinates.
    def polar_to_cart(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    # For loop to iterate over each circle and compute their (x, y) coordinates.
    # (z) coordinates are already given by the user.
    for i in range(n):
        start = i * circa_points
        end = (i + 1) * circa_points
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / circa_points)
        radius = centers[i, 3]
        (x, y) = polar_to_cart(angles, radius)

        coords[start:end, 0] = x + centers[i, 0]  # X
        coords[start:end, 1] = y + centers[i, 1]  # Y
        coords[start:end, 2] = centers[i, 2]  # check
        coords[start:end, 3] = centers[i, 4]  # Z0
        coords[start:end, 4] = i  # Tree ID
        coords[start:end, 5] = centers[i, 5]  # sector occupancy
        coords[start:end, 6] = centers[i, 6]  # points in inner circle
        coords[start:end, 7] = centers[i, 7]  # Z0
        coords[start:end, 8] = centers[i, 3] * 2  # Diameter
        coords[start:end, 9] = centers[i, 8]  # outlier probability

        if (
            (centers[i, 5] < min_n_sectors / n_sectors * 100)
            | (centers[i, 6] > threshold)
            | (centers[i, 8] > 0.3)
            | (centers[i, 3] < R_min)
            | (centers[i, 3] > R_max)
        ):  # only happens when which_dbh == 0 # which_valid_points should be used here
            coords[start:end, 10] = 1  # does not pass quality checks
        else:
            coords[start:end, 10] = 0  # passes quality checks
    return coords


def draw_circles(
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
    R_min=0.03,
    R_max=0.5,
    threshold=5,
    n_sectors=16,
    min_n_sectors=9,
    circa_points=200,
):
    """This function generates points that comprise the circles computed by
    fit_circle_check function, so sections can be visualized. The circles are
    then saved in a LAS file, along some descriptive fields. Each circle
    corresponds on a one-to-one basis to the sections described by the user.

    Parameters
    ----------
    X_c : numpy.ndarray
        Matrix containing (x) coordinates of the center of the sections.
    Y_c : numpy.ndarray
        Matrix containing (y) coordinates of the center of the sections.
    R : numpy.ndarray
        Vector containing section radia.
    sections : numpy.ndarray
        Vector containing section heights (normalized heights).
    section_perct : numpy.ndarray
        Matrix containing the percentage of occupied sectors.
    n_points_in : numpy.ndarray
        Matrix containing the number of points in the inner circumferences.
    tree_vector : numpy.ndarray
        detected_trees output from individualize_trees.
    outliers : numpy.ndarray
        Vector containing the 'outlier probability' of each section.
    filename_las : char
        File name for the output file.
    R_min : float
        Refer to fit_circle_check in 'sections' module. Defaults to 0.03.
    R_max : float
        Refer to fit_circle_check in 'sections' module. Defaults to 0.5.
    threshold : float
        Refer to fit_circle_check in 'sections' module. Defaults to 5.
    n_sectors : int
        Refer to fit_circle_check in 'sections' module. Defaults to 16.
    min_n_sectors: int
        Refer to fit_circle_check in sections module. Defaults to 9.
    circa_points : int
        Number of points used to draw each circle. Defaults to 200.
    """
    coords = generate_circles_cloud(
        X_c,
        Y_c,
        R,
        sections,
        check_circle,
        sector_perct,
        n_points_in,
        tree_vector,
        outliers,
        R_min,
        R_max,
        threshold,
        n_sectors,
        min_n_sectors,
        circa_points,
    )

    # LAS file containing circle coordinates.
    las_circ = laspy.create(point_format=2, file_version="1.2")
    las_circ.x = coords[:, 0]
    las_circ.y = coords[:, 1]
    las_circ.z = coords[:, 2]

    # All extra fields.

    # las_circ.add_extra_dim(laspy.ExtraBytesParams(name = "check", type = np.int32))
    # las_circ.check = coords[:, 3]

    las_circ.add_extra_dim(laspy.ExtraBytesParams(name="tree_ID", type=np.int32))
    las_circ.tree_ID = coords[:, 4]

    las_circ.add_extra_dim(laspy.ExtraBytesParams(name="sector_occupancy_percent", type=np.float64))
    las_circ.sector_occupancy_percent = coords[:, 5]

    las_circ.add_extra_dim(laspy.ExtraBytesParams(name="pts_inner_circle", type=np.int32))
    las_circ.pts_inner_circle = coords[:, 6]

    las_circ.add_extra_dim(laspy.ExtraBytesParams(name="Z0", type=np.float64))
    las_circ.Z0 = coords[:, 7]

    las_circ.add_extra_dim(laspy.ExtraBytesParams(name="Diameter", type=np.float64))
    las_circ.Diameter = coords[:, 8]

    las_circ.add_extra_dim(laspy.ExtraBytesParams(name="outlier_prob", type=np.float64))
    las_circ.outlier_prob = coords[:, 9]

    las_circ.add_extra_dim(laspy.ExtraBytesParams(name="quality", type=np.int32))
    las_circ.quality = coords[:, 10]

    las_circ.write(filename_las)


# -----------------------------------------------------------------------------
# draw_axes
# -----------------------------------------------------------------------------


def generate_axis_cloud(
    tree_vector,
    line_downstep=0.5,
    line_upstep=10.0,
    stripe_lower_limit=0.5,
    stripe_upper_limit=2.5,
    point_interval=0.01,
):
    """This function generates points that comprise the axes computed by
    individualize_trees, so that they can be visualized. It output two
    numpy.ndarray that describes the point cloud of the axis
    and their associated tilt.

    Parameters
    ----------
    tree_vector : numpy.ndarray
        detected_trees output from individualize_trees.
    filename_las : char
        File name for the output file
    line_downstep : float
        From the stripe centroid, how much (downwards direction) will the drawn
        axes extend (units is meters). Defaults to 0.5.
    line_upstep : float
        From the stripe centroid, how much (upwards direction) will the drawn
        axes extend (units is meters). Defaults to 10.0.
    stripe_lower_limit : float
        Lower (vertical) limit of the stripe (units is meters). Defaults to 0.7.
    stripe_upper_limit : float
        Upper (vertical) limit of the stripe (units is meters). Defaults to 3.5.
    point_interval : float
        Step value used to draw points (unit is meters). Defaults to 0.01.

    Returns
    --------
    axes_point : numpy.ndarray
        Matrix that describes the point cloud of the axes
    tilt : numpy.ndarray
        Matrix that describes the tilt of each axes
    """
    stripe_centroid = (stripe_lower_limit + stripe_upper_limit) / 2.0
    mean_descend = stripe_centroid + line_downstep
    mean_rise = line_upstep - stripe_centroid

    up_iter = int(np.floor(mean_rise / point_interval))
    down_iter = int(mean_descend / point_interval)

    axes_points = np.zeros((tree_vector.shape[0] * (up_iter + down_iter), 3))
    tilt = np.zeros(tree_vector.shape[0] * (up_iter + down_iter))

    ind = 0
    for i in range(tree_vector.shape[0]):
        if np.sum(np.exp2(tree_vector[i, 1:4])) > 0:
            vector = -tree_vector[i, 1:4] if tree_vector[i, 3] < 0 else tree_vector[i, 1:4]
            next_ind = ind + up_iter + down_iter
            axes_points[ind:next_ind] = (
                np.column_stack(
                    (
                        np.arange(-down_iter, up_iter),
                        np.arange(-down_iter, up_iter),
                        np.arange(-down_iter, up_iter),
                    )
                )
                * vector
                * point_interval
                + tree_vector[i, 4:7]
            )
            tilt[ind:next_ind] = tree_vector[i, 8]
            ind = next_ind

    axes_points = axes_points[:ind]
    return axes_points, tilt


def draw_axes(
    tree_vector,
    filename_las,
    line_downstep=0.5,
    line_upstep=10.0,
    stripe_lower_limit=0.5,
    stripe_upper_limit=2.5,
    point_interval=0.01,
):
    """This function generates points that comprise the axes computed by
    individualize_trees, so that they can be visualized. The axes are then
    saved in a LAS file, along some descriptive fields. Each axis corresponds
    on a one-to-one basis to the individualized trees.

    Parameters
    ----------
    tree_vector : numpy.ndarray
        detected_trees output from individualize_trees.
    filename_las : char
        File name for the output file
    line_downstep : float
        From the stripe centroid, how much (downwards direction) will the drawn
        axes extend (units is meters). Defaults to 0.5.
    line_upstep : float
        From the stripe centroid, how much (upwards direction) will the drawn
        axes extend (units is meters). Defaults to 10.0.
    stripe_lower_limit : float
        Lower (vertical) limit of the stripe (units is meters). Defaults to 0.5.
    stripe_upper_limit : float
        Upper (vertical) limit of the stripe (units is meters). Defaults to 2.5.
    point_interval : float
        Step value used to draw points (unit is meters). Defaults to 0.01..
    """
    axes_points, tilt = generate_axis_cloud(
        tree_vector,
        line_downstep,
        line_upstep,
        stripe_lower_limit,
        stripe_upper_limit,
        point_interval,
    )

    las_axes = laspy.create(point_format=2, file_version="1.2")
    las_axes.x = axes_points[:, 0]
    las_axes.y = axes_points[:, 1]
    las_axes.z = axes_points[:, 2]
    las_axes.add_extra_dim(laspy.ExtraBytesParams(name="tilting_degree", type=np.float64))
    las_axes.tilting_degree = tilt

    las_axes.write(filename_las)
