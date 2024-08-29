import numpy as np
from scipy import optimize as opt
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance_matrix

# -----------------------------------------------------------------------------
# point_clustering
# -----------------------------------------------------------------------------


def point_clustering(X, Y, max_dist):
    """This function clusters points by distance and finds the largest
    cluster. It is to be used inside fit_circle_check().

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.
    max_dist : float
        Max separation among the points to be considered as members of the same
        cluster.

    Returns
    -------
    X_g : numpy.ndarray
        Vector containing the (x) coordinates of the largest cluster.
    Y_g : numpy.ndarray
        Vector containing the (y) coordinates of the largest cluster.
    """

    # Stacks 1D arrays ([X], [Y]) into a 2D array ([X, Y])
    xy_stack = np.column_stack((X, Y))

    # sch.fclusterdata outputs a vector that contains cluster ID of each point
    # (which cluster does each point belong to)
    clust_id = sch.fclusterdata(xy_stack, max_dist, criterion="distance", metric="euclidean")

    # Set of all clusters
    clust_id_unique = np.unique(clust_id)

    # For loop that iterates over each cluster ID, sums its elements and finds
    # the largest
    n_max = 0
    for c in clust_id_unique:
        # How many elements are in each cluster
        n = np.sum(clust_id == c)

        # Update largest cluster and its cardinality
        if n > n_max:
            n_max = n
            largest_cluster = c

    # X, Y coordinates of points that belong to the largest cluster
    X_g = xy_stack[clust_id == largest_cluster, 0]
    Y_g = xy_stack[clust_id == largest_cluster, 1]

    # Output: those X, Y coordinates
    return X_g, Y_g


# -------------------------------------------------------------------------------------------------------------------------------------------------------
# fit_circle
# -------------------------------------------------------------------------------------------------------------------------------------------------------


def fit_circle(X, Y):
    """This function fits points within a tree section into a circle by
    least squares minimization. It is to be used inside fit_circle_check().

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.

    Returns
    -------
    circle_c : numpy.ndarray
        Matrix containing the (x, y) coordinates of the circle center.
    mean_radius : numpy.ndarray
        Vector containing the radius of each fitted circle
        (units is meters).
    """

    # Function that computes distance from each 2D point to a single point defined by (X_c, Y_c)
    # It will be used to compute the distance from each point to the circle center.
    def _calc_R(X, Y, X_c, Y_c):
        return np.sqrt((X - X_c) ** 2 + (Y - Y_c) ** 2)

    # Function that computes algebraic distance from each 2D point to some middle circle c
    # It calls calc_R (just defined above) and it is used during the least squares optimization.
    def _f_2(c, X, Y):
        R_i = _calc_R(X, Y, *c)
        return R_i - R_i.mean()

    # Initial barycenter coordinates (middle circle c center)
    X_m = X.mean()
    Y_m = Y.mean()
    barycenter = X_m, Y_m

    # Least square minimization to find the circle that best fits all
    # points within the section. 'ier' is a flag indicating whether the solution
    # was found (ier = 1, 2, 3 or 4) or not (otherwise).
    circle_c, _ = opt.leastsq(_f_2, barycenter, args=(X, Y), maxfev=2000)

    # Its radius
    radius = _calc_R(X, Y, *circle_c)
    mean_radius = radius.mean()

    # Output: - X, Y coordinates of best-fit circle center - its radius
    return circle_c, mean_radius


# -----------------------------------------------------------------------------
# inner_circle
# -----------------------------------------------------------------------------


def inner_circle(X, Y, X_c, Y_c, R, times_R):
    """Function that computes an internal circle inside the one fitted by
    fit_circle. This new circle is used as a validation tool and it gives
    insight on the quality of the 'fit_circle-circle'.

        - If points are closest to the inner circle, then the first fit was not
          appropriate

        - On the contrary, if points are closer to the outer circle, the
          'fit_circle-circle' is appropriate and describes well the stem diameter.

    Instead of directly computing the inner circle, it just takes a proportion
    (less than one) of the original circle radius and its center. Then, it just
    checks how many points are closest to the inner circle than to the original
    circle.

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.
    X_c : numpy.ndarray
    Vector containing (x) coordinates of fitted circles.
    Y_c : numpy.ndarray
        Vector containing (y) coordinates of fitted circles.
    R : numpy.ndarray
        Vector containing the radii of the fitted circles.

    Returns
    -------
    n_points_in : numpy.ndarray
        Vector containing the number of points inside the inner circle of each
        section.
    """

    # Distance from each 2D point to the center.
    distance = np.sqrt((X - X_c) ** 2 + (Y - Y_c) ** 2)

    # Number of points closest to the inner circle, whose radius is
    # proportionate to the outer circle radius by a factor defined by 'times_R'.
    n_points_in = np.sum(distance < R * times_R)

    # Output: Number of points closest to the inner circle.
    return n_points_in


# -------------------------------------------------------------------------------------------------------------------------------------------------------
# sector_occupancy
# -------------------------------------------------------------------------------------------------------------------------------------------------------


def sector_occupancy(X, Y, X_c, Y_c, R, n_sectors, min_n_sectors, width):
    """This function provides quality measurements for the fitting of the
    circle. It divides the section in a number of sectors to check if there are
    points within them (so they are occupied). If there are not enough occupied
    sectors, the section fails the test, as it is safe to assume it has an
    abnormal, non desirable structure.

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.
    X_c : numpy.ndarray
        Vector containing (x) coordinates of fitted circles.
    Y_c : numpy.ndarray
        Vector containing (y) coordinates of fitted circles.
    R : numpy.ndarray
        Vector containing the radii of the fitted circles.
    n_sectors : int
        Number of sectors in which sections will be divided.
    min_n_sectors : int
        Minimum number of occupied sectors in a section for its fitted circle
        to be considered as valid.
    width : float
        Width around the fitted circle to look for points (units is
        meters).

    Returns
    -------
    perct_occupied_sectors : float
        Percentage of occupied sectors in each section.
    enough_occupied_sectors : int
        Binary indicators whether the fitted circle is valid
        or not. 1 - valid, 0 - not valid.
    """

    # Coordinates translation.
    X_red = X - X_c
    Y_red = Y - Y_c

    # Computation of radius and angle necessary to transform cartesian coordinates
    # to polar coordinates.
    radial_coord = np.sqrt(X_red**2 + Y_red**2)  # radial coordinate
    angular_coord = np.arctan2(X_red, Y_red)  # angular coordinate. This function from numpy directly computes it.

    # Points that are close enough to the circle that will be checked.
    points_within = (radial_coord > (R - width)) * (radial_coord < (R + width))

    # Codification of points in each sector. Basically the range of angular coordinates
    # is divided in n_sector pieces and granted an integer number. Then, every
    # point is assigned the integer corresponding to the sector it belongs to.
    norm_angles = np.floor(
        angular_coord[points_within] / (2 * np.pi / n_sectors)
    )  # np.floor only keep the integer part of the division

    # Number of points in each sector.
    n_occupied_sectors = np.size(np.unique(norm_angles))

    # Percentage of occupied sectors.
    perct_occupied_sectors = n_occupied_sectors * 100 / n_sectors

    # If there are enough occupied sectors, then it is a valid section.
    enough_occupied_sectors = 0 if n_occupied_sectors < min_n_sectors else 1  # TODO(RJ): Maybe convert this to boolean

    # Output: percentage of occupied sectors | boolean indicating if it has enough
    # occupied sectors to pass the test.
    return perct_occupied_sectors, enough_occupied_sectors


# -----------------------------------------------------------------------------
# fit_circle_check
# -----------------------------------------------------------------------------


def fit_circle_check(
    X,
    Y,
    review,
    second_time,
    times_R,
    threshold,
    R_min,
    R_max,
    max_dist,
    n_points_section,
    n_sectors,
    min_n_sectors,
    width,
):
    """This function calls fit_circle() to fit points within a section to a
    circle by least squares minimization. These circles will define tree
    sections. It checks the goodness of fit using sector_occupancy and
    inner_circle. If fit is not appropriate, another circle will be fitted
    using only points from the largest cluster inside the first circle.

    Parameters
    ----------
    X : numpy.ndarray
        Vector containing (x) coordinates of points belonging to a tree section.
    Y : numpy.ndarray
        Vector containing (y) coordinates of points belonging to a tree section.
    second_time : numpy.ndarray
        Vector containing integers that indicates whether it is the first time
        a circle is fitted or not (will be modified internally).
    times_R : float
        Ratio of radius between outer circle and inner circle.
    threshold : float
        Minimum number of points in inner circle for a fitted circle to be
        valid.
    R_min : float
        Minimum radius that a fitted circle must have to be valid.
    R_max : float
        Maximum radius that a fitted circle must have to be valid.
    max_dist : float
        Max separation among the points to be considered as members of the same
        cluster.
    n_points_section : int
        Minimum points within a section for its fitted circle to be valid.
    n_sectors : int
        Number of sectors in which sections will be divided.
    min_n_sectors : int
        Minimum number of occupied sectors in a section for its fitted circle
        to be considered as valid.
    width : float
        Width around the fitted circle to look for points (units is millimeters).

    Returns
    -------
    X_gs : numpy.ndarray
        Matrix containing (x) coordinates of largest clusters.
    Y_gs : numpy.ndarray
        Matrix containing (y) coordinates of largest clusters.
    X_c : numpy.ndarray
        Matrix containing (x) coordinates of the center of the best-fit circles.
    Y_c : numpy.ndarray
        Matrix containing (y) coordinates of the center of the best-fit circles.
    R : numpy.ndarray
        Vector containing best-fit circle radii.
    section_perct : numpy.ndarray
        Matrix containing the percentage of occupied sectors.
    n_points_in : numpy.ndarray
        Matrix containing the number of points in the inner circle.
    """

    # If loop that discards sections that do not have enough points (n_points_section)
    if X.size > n_points_section:
        # Call to fit_circle to fit the circle that best fits all points
        # within the section.
        (circle_center, R) = fit_circle(X=X, Y=Y)
        X_c = circle_center[0]  # Column 0 is center X coordinate
        Y_c = circle_center[1]  # Column 1 is center Y coordinate

        # Call to inner_circle to fit an inner circle and to get the number
        # of points closest to it.
        n_points_in = inner_circle(X, Y, X_c, Y_c, R, times_R)

        # Call to sector_occupancy to check if sectors around inner circle are occupied.
        sector_perct, enough_sectors = sector_occupancy(X, Y, X_c, Y_c, R, n_sectors, min_n_sectors, width)

        # If any of the following conditions hold:
        #   - Too many points in inner circle
        #   - Radius of best-fit circle is too small
        #   - Number of occupied sectors is too low
        # Then proceed with countermeasures
        if n_points_in > threshold or R < R_min or R > R_max or enough_sectors == 0:
            # If this is not the second round or, simply, if it is the first round,
            # then proceed
            if second_time == 0:
                # First round implies there is no X_g or Y_g, as points would not
                # have been grouped yet. point_clustering is called.
                X_g, Y_g = point_clustering(X, Y, max_dist)  # X_g or Y_g are the coordinates of the largest cluster.

                # If cluster size is big enough, then proceed. It is done this way to
                # account for cases where, even though the section had enough points,
                # there might not be enough points within the largest cluster.
                if X_g.size > n_points_section:
                    # Call to fit_circle_check (lets call it the 'deep call').
                    # Now it is guaranteed that it is a valid section (has enough
                    # points and largest cluster has enough points as well).
                    (
                        X_c,
                        Y_c,
                        R,
                        review,
                        second_time,
                        sector_perct,
                        n_points_in,
                    ) = fit_circle_check(
                        X_g,
                        Y_g,
                        0,
                        1,
                        times_R,
                        threshold,
                        R_min,
                        R_max,
                        max_dist,
                        n_points_section,
                        n_sectors,
                        min_n_sectors,
                        width,
                    )

                # If cluster size is not big enough, then don't take the section
                # it belongs to into account.
                else:
                    review = 1  # Even if it is not a valid section, lets note it has been checked.
                    X_c = 0
                    Y_c = 0
                    R = 0
                    second_time = 1

            # If this is the second round (whether the first round successfully
            # provided a valid section or not), then proceed.
            else:
                review = 1  # Just stating that if this is the second round, the check has happened.

    # This matches the first loop. If section is not even big enough (does not contain enough points), it is not valid.
    else:
        review = 2
        X_c = 0
        Y_c = 0
        R = 0
        second_time = 2
        sector_perct = 0
        n_points_in = 0

    return X_c, Y_c, R, review, second_time, sector_perct, n_points_in


# -----------------------------------------------------------------------------
# compute_sections
# -----------------------------------------------------------------------------


def compute_sections(
    stems,
    sections,
    section_width=0.02,
    times_R=0.5,
    threshold=5,
    R_min=0.03,
    R_max=0.5,
    max_dist=0.02,
    n_points_section=80,
    n_sectors=16,
    min_n_sectors=9,
    width=2,
    X_field=0,
    Y_field=1,
    Z0_field=3,
    tree_id_field=4,
    progress_hook=None,
):
    """This function calls fit_circle_check() to compute stem diameter at
    given sections.

    Parameters
    ----------
    stems : numpy.ndarray
        Point cloud containing the individualized trees. It is expected to
        have X, Y, Z0 and tree_ID fields.
    sections : numpy.ndarray
        Matrix containing a range of height values at which sections will be
        computed.
    section_width : float
        Points within this distance from any `sections` value will be considered
        as belonging to said section (units is meters). Defaults to 0.02.
    times_R : float
        Refer to fit_circle_check. Defaults to 0.5.
    threshold : float
        Refer to fit_circle_check. Defaults to 5.
    R_min : float
        Refer to fit_circle_check. Defaults to 0.03.
    R_max : float
        Refer to fit_circle_check. Defaults to 0.5.
    max_dist : float
        Refer to fit_circle_check. Defaults to 0.02.
    n_points_section : int
        Refer to fit_circle_check. Defaults to 80.
    n_sectors : int
        Refer to fit_circle_check. Defaults to 16.
    min_n_sectors : int
        Refer to fit_circle_check. Defaults to 9.
    width : float
        Refer to fit_circle_check. Defaults to 2.0.
    X_field : int
        Index at which (x) coordinate is stored. Defaults to 0.
    Y_field : int
        Index at which (y) coordinate is stored. Defaults to 1.
    Z0_field : int
        Index at which (z0) coordinate is stored. Defaults to 3.
    tree_id_field : int
        Index at which cluster ID is stored. Defaults to 4.
    progress_hook : callable, optional
        A hook that take two int, the first is the current number of iteration
        and the second is the targeted number iteration. Defaults to None.

    Returns
    -------
    X_c : numpy.ndarray
        Matrix containing (x) coordinates of the center of the best-fit circles.
    Y_c : numpy.ndarray
        Matrix containing (y) coordinates of the center of the best-fit circles.
    R : numpy.ndarray
        Vector containing best-fit circle radii.
    section_perct : numpy.ndarray
        Matrix containing the percentage of occupied sectors.
    n_points_in : numpy.ndarray
        Matrix containing the number of points in the inner circles.
    """
    trees = np.unique(stems[:, tree_id_field])  # Select the column that contains tree ID
    n_trees = trees.size  # Number of trees
    n_sections = sections.size  # Number of sections

    X_c = np.zeros((n_trees, n_sections))  # Empty array to store X data
    Y_c = np.zeros((n_trees, n_sections))  # Empty array to store Y data
    R = np.zeros((n_trees, n_sections))  # Empty array to store radius data
    check_circle = np.zeros((n_trees, n_sections))  # Empty array to store 'check' data
    second_time = np.zeros((n_trees, n_sections))  # Empty array to store 'second_time' data
    sector_perct = np.zeros((n_trees, n_sections))  # Empty array to store percentage of occupied sectors data
    n_points_in = np.zeros((n_trees, n_sections))  # Empty array to store inner points data

    # Filling previous empty arrays

    # Auxiliary index for first loop
    tree = -1  # Loop will start at -1
    if progress_hook is not None:
        progress_hook(0, n_trees)
    # First loop: iterates over each tree
    for tr in trees:
        # Tree ID is used to iterate over trees
        tree_i = stems[stems[:, tree_id_field] == tr, :]
        tree = tree + 1
        if progress_hook is not None:
            progress_hook(tree + 1, n_trees)
        # Auxiliary index for second loop
        section = 0

        # Second loop: iterates over each section
        for b in sections:
            # Selecting (x, y) coordinates of points within the section
            X = tree_i[
                (tree_i[:, Z0_field] >= b) & (tree_i[:, Z0_field] < b + section_width),
                X_field,
            ]
            Y = tree_i[
                (tree_i[:, Z0_field] >= b) & (tree_i[:, Z0_field] < b + section_width),
                Y_field,
            ]

            # fit_circle_check call. It provides data to fill the empty arrays
            (
                X_c[tree, section],
                Y_c[tree, section],
                R[tree, section],
                check_circle[tree, section],
                second_time[tree, section],
                sector_perct[tree, section],
                n_points_in[tree, section],
            ) = fit_circle_check(
                X,
                Y,
                0,
                0,
                times_R,
                threshold,
                R_min,
                R_max,
                max_dist,
                n_points_section,
                n_sectors,
                min_n_sectors,
                width,
            )

            section = section + 1
    return (X_c, Y_c, R, check_circle, second_time, sector_perct, n_points_in)


# -----------------------------------------------------------------------------
# tilt_detection
# -----------------------------------------------------------------------------


def tilt_detection(X_tree, Y_tree, radius, sections, Z_field=2, w_1=3.0, w_2=1.0):
    """This function finds outlier tilting values among sections within a tree
    and assigns a score to the sections based on those outliers. Two kinds of
    outliers are considered.

        - Absolute outliers are obtained from the sum of the deviations from
          every section center to all axes within a tree (the most tilted sections
          relative to all axes)

        - Relative outliers are obtained from the deviations of other section
          centers from a certain axis, within a tree (the most tilted sections
          relative to a certain axis)

    The 'outlier score' consists on a weighted sum of the absolute tilting value
    and the relative tilting value.

    Parameters
    ----------
    X_tree : numpy.ndarray
        Matrix containing (x) coordinates of the center of the sections.
    Y_tree : numpy.ndarray
        Matrix containing (y) coordinates of the center of the sections.
    radius : numpy.ndarray
        Vector containing section radii.
    sections : numpy.ndarray
        Vector containing the height of the section associated to each section.
    Z_field : int
        Index at which (z) coordinate is stored. Defaults to 2.
    w_1 : float
        Weight of absolute deviation. Defaults to 3.0.
    w_2 : float
        Weight of relative deviation. Defaults to 1.0.

    Returns
    -------
    outlier_prob : numpy.ndarray
        Vector containing the 'outlier probability' of each section.
    """

    # This function simply defines 1st and 3rd quartile of a vector and separates
    # values that are outside the interquartile range defined by these. Those
    # are the candidates to be outliers. This filtering may be done either
    # directly from the interquartile range, or from a certain distance from it,
    # thanks to 'n_range' parameter. Its default value is 1.5.

    def _outlier_vector(vector, lower_q=0.25, upper_q=0.75, n_range=1.5):
        q1, q3 = np.quantile(vector, [lower_q, upper_q])  # First quartile and Third quartile
        iqr = q3 - q1  # Interquartile range

        lower_bound = q1 - iqr * n_range  # Lower bound of filter. If n_range = 0 -> lower_bound = q1
        upper_bound = q3 + iqr * n_range  # Upper bound of filter. If n_range = 0 -> upper_bound = q3

        # return the outlier vector.
        return ((vector < lower_bound) | (vector > upper_bound)).astype(int)

    # Empty matrix that will store the probabilities of a section to be invalid
    outlier_prob = np.zeros_like(X_tree)

    # First loop: iterates over each tree
    for i in range(X_tree.shape[0]):
        # If there is, at least, 1 circle with positive radius in a tree, then
        # proceed (invalid circles are stored with a radius value of 0)
        if np.sum(radius[i, :]) > 0:
            # Filtering sections within a tree that have valid circles (non-zero radius).
            valid_radius = radius[i, :] > 0
            num_valid_sections = np.size(sections[valid_radius])
            # Weights associated to each section. They are computed in a way
            # that the final value of outliers sums up to 1 as maximum.
            abs_outlier_w = w_1 / (num_valid_sections * w_2 + w_1)
            rel_outlier_w = w_2 / (num_valid_sections * w_2 + w_1)

            # Vertical distance matrix among all sections (among their centers)
            # Empty matrix to store heights of each section
            heights = np.zeros((num_valid_sections, Z_field))
            #  Height (Z value) of each section
            heights[:, 0] = np.transpose(sections[valid_radius])
            # Vertical distance matrix
            z_dist_matrix = distance_matrix(heights, heights)

            # Horizontal distance matrix among all sections (among their centers)
            # Store X, Y coordinates of each section
            c_coord = np.column_stack((X_tree[i][valid_radius], Y_tree[i][valid_radius]))
            # Horizontal distance matrix
            xy_dist_matrix = distance_matrix(c_coord, c_coord)

            # Tilting measured from every vertical within a tree: All verticals
            # obtained from the set of sections within a tree. For instance, if
            # there are 10 sections, there are 10 tilting values for each section.
            tilt_matrix = np.degrees(np.arctan(xy_dist_matrix / z_dist_matrix))

            # Summation of tilting values from each center.
            tilt_sum = np.nansum(tilt_matrix, axis=0)

            # Outliers within previous vector (too low / too high tilting values).
            # These are abnormals tilting values from ANY axis.
            outlier_prob[i][valid_radius] = _outlier_vector(tilt_sum) * abs_outlier_w

            # Second loop: iterates over each section (within a single tree).
            for j in range(np.size(sections[valid_radius])):
                # Search for abnormals tilting values from a CERTAIN axis.
                tilt_matrix[j, j] = np.quantile(tilt_matrix[j, ~j], 0.5)
                # Storing those values.
                rel_outlier = _outlier_vector(tilt_matrix[j]) * rel_outlier_w
                # Sum of absolute outlier value and relative outlier values
                outlier_prob[i][valid_radius] += rel_outlier

    return outlier_prob


# -----------------------------------------------------------------------------
# tree_locator
# --------------------------------------------------------------------------


def tree_locator(
    sections,
    X_c,
    Y_c,
    tree_vector,
    sector_perct,
    R,
    outliers,
    n_points_in,
    threshold=5,
    X_field=0,
    Y_field=1,
    Z_field=2,
):
    """This function generates points that locate the individualized trees and
    computes their DBH (diameter at breast height). It uses all the quality
    measurements defined in previous functions to check whether the DBH should
    be computed or not and to check which point should be used as the tree locator.

    The tree locators are then saved in a LAS file. Each tree locator corresponds
    on a one-to-one basis to the individualized trees.

    Parameters
    ----------
    sections : numpy.ndarray
        Vector containing section heights (normalized heights).
    X_c : numpy.ndarray
        Matrix containing (x) coordinates of the center of the sections.
    Y_c : numpy.ndarray
        Matrix containing (y) coordinates of the center of the sections.
    tree_vector : numpy.ndarray
        detected_trees output from individualize_trees.
    sector_perct : numpy.ndarray
        Matrix containing the percentage of occupied sectors.
    R : numpy.ndarray
        Vector containing section radii.
    outliers : numpy.ndarray
        Vector containing the 'outlier probability' of each section.
    n_points_in : numpy.ndarray
        Matrix containing the number of points in the inner circles.
    threshold : float
        Minimum number of points in inner circle for a fitted circle to be valid.
        Defaults to 5.
    X_field : int
        Index at which (x) coordinate is stored. Defaults to 0.
    Y_field : int
        Index at which (y) coordinate is stored. Defaults to 1.
    Z_field : int
        Index at which (z) coordinate is stored. Defaults to 2.

    Returns
    -------
    dbh_values : numpy.ndarray
        Vector containing DBH values.
    tree_locations : numpy.ndarray
        Matrix containing (x, y, z) coordinates of each tree locator.
    """
    DBH = 1.3  # Breast height constant

    # Number of trees
    n_trees = X_c.shape[0]
    # Empty vector to be filled with tree locators
    tree_locations = np.zeros((n_trees, 3))
    # Empty vector to be filled with DBH values.
    dbh_values = np.zeros((n_trees, 1))

    def _axis_location(index):
        """Given an index compute tree location from axis"""
        vector = -tree_vector[index, 1:4] if tree_vector[index, 3] < 0 else tree_vector[index, 1:4]
        dbh_values[index] = 0
        # Compute the height difference between centroid and BH
        diff_height = DBH - tree_vector[index, 6] + tree_vector[index, 7]
        # Compute the distance between centroid and axis point at BH.
        dist_centroid_dbh = diff_height / np.cos(np.radians(tree_vector[index, 8]))
        # Compute coordinates of axis point at BH.
        tree_locations[i, :] = vector * dist_centroid_dbh + tree_vector[i, 4:7]

    def _dbh_location(index, which_dbh):
        """Given an index, compute the tree location from the computed DBH"""
        dbh_values[index] = R[index, which_dbh] * 2
        # Their centers are averaged and we keep that value
        tree_locations[index, X_field] = X_c[index, which_dbh]
        tree_locations[index, Y_field] = Y_c[index, which_dbh]
        # Original height is obtained
        tree_locations[index, Z_field] = tree_vector[index, 7] + DBH

    # This if loop covers the cases where the stripe was defined in a way that
    # it did not include BH and DBH nor tree locator cannot be obtained from a
    # section at or close to BH. If that happens, tree axis is used to locate
    # the tree and DBH is not computed.
    if np.min(sections) > DBH:
        for i in range(n_trees):
            _axis_location(i)
    else:
        d = 1
        which_dbh = np.argmin(np.abs(sections - DBH))  # Which section is closer to BH.

        # get surrounding sections too
        lower_d_section = max(0, which_dbh - d)
        upper_d_section = min(sections.shape[0], which_dbh + d)
        # BH section and its neighbors. From now on, neighborhood
        close_to_dbh = np.arange(lower_d_section, upper_d_section)

        for i in range(n_trees):  # For each tree
            which_valid_R = R[i, close_to_dbh] > 0  # From neighborhood, select only those with non 0 radius
            # From neighborhood, select only those with outlier probability lower than 30 %
            which_valid_out = outliers[i, close_to_dbh] < 0.3
            # only those with sector occupancy higher than 30 %
            which_valid_sector_perct = sector_perct[i, close_to_dbh] > 30.0
            # valid points could be retrieved as well / i.e. only those with enough points in inner circle
            # which_valid_points = (n_points_in[i, close_to_dbh] < threshold)

            # If there are valid sections among the selected
            if np.any(which_valid_R) & np.any(which_valid_out):
                # If first section is BH section and if itself and its only neighbor are valid
                if (
                    (lower_d_section == 0)
                    & (np.all(which_valid_R))
                    & (np.all(which_valid_out))
                    & np.all(which_valid_sector_perct)
                ):  # Only happens when which_dbh == 0 in this case which_valid_points should be used here
                    # If they are coherent: difference among their radii is not larger than 10 % of the largest radius
                    if np.abs(R[i, close_to_dbh[0]] - R[i, close_to_dbh[1]]) < np.max(R[i, close_to_dbh]) * 0.1:
                        _dbh_location(i, which_dbh)
                    # If not all of them are valid, then there is no coherence and the axis location is used
                    else:
                        _axis_location(i)

                # If last section is BH section and if itself and its only neighbor are valid
                elif (upper_d_section == sections.shape[0]) & (np.all(which_valid_R)) & (np.all(which_valid_out)):
                    # if they are coherent; difference among their radii is not larger than 15 % of the largest radius
                    if np.abs(R[i, close_to_dbh[0]] - R[i, close_to_dbh[1]]) < np.max(R[i, close_to_dbh]) * 0.15:
                        # use BH section diameter as DBH
                        _dbh_location(i, which_dbh)

                    # If not all of them are valid, then there is no coherence in
                    # any case, and the axis location is used and DBH is not computed
                    else:
                        _axis_location(i)

                # In any other case, BH section is not first or last section, so it has 2 neighbors
                # 3 possibilities left:
                # A: Not all of three sections are valid: there is no possible coherence
                # B: All of three sections are valid, and there is coherence among the three
                # C: All of three sections are valid, but there is only coherence among neighbors
                # and not BH section or All of three sections are valid, but there is no coherence
                else:
                    # Case A:
                    if not ((np.all(which_valid_R)) & (np.all(which_valid_out)) & np.all(which_valid_sector_perct)):
                        _axis_location(i)
                    # case B&C:
                    else:
                        valid_sections = close_to_dbh  # Valid sections indexes
                        valid_radii = R[i, valid_sections]  # Valid sections radii
                        median_radius = np.median(valid_radii)  # Valid sections median radius
                        # Valid sections absolute deviation from median radius
                        abs_dev = np.abs(valid_radii - median_radius)
                        mad = np.median(abs_dev)  # Median absolute deviation
                        # Only keep sections close to median radius (3 MAD criterion)
                        filtered_sections = valid_sections[abs_dev < 3 * mad]
                        # 3 things can happen here:
                        # There are no deviated sections --> there is coherence among 3 --> case B
                        # There are 2 deviated sections --> only median radius survives filter --> case C
                        # Case B
                        if filtered_sections.shape[0] == close_to_dbh.shape[0]:
                            _dbh_location(i, which_dbh)
                        # Case C
                        else:
                            _axis_location(i)
            # If there is not a single section that either has non 0 radius nor low
            # outlier probability, there is nothing else to do -> axis location is used
            else:
                _axis_location(i)

    return dbh_values, tree_locations
