import math
import warnings

import CSF_3DFin as CSF
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree

from .primitives.clustering import DBSCAN_clustering
from .primitives.voxel import voxelate

# -----------------------------------------------------------------------------
# clean_ground
# -----------------------------------------------------------------------------


def clean_ground(cloud, res_ground=0.15, min_points=2):
    """This function takes a point cloud and denoises it via DBSCAN
    clustering. It first voxelates the point cloud, then it clusters the voxel
    cloud and excludes clusters of size less than the value determined by
    min_points.

    Parameters
    ----------
    cloud : numpy.ndarray
        The point cloud to be denoised. Matrix containing (x, y, z) coordinates
        of the points.
    res_ground : float
        (x, y, z) voxel resolution in meters. Defaults to 0.15.
    min_points : int
        Clusters with size smaller than this value will be regarded as noise
        and thus eliminated from the point cloud. Defaults to 2.

    Returns
    -------
    clust_cloud : numpy.ndarray
        The denoised point cloud. Matrix containing (x, y, z) coordinates of
        the denoised points.
    """

    vox_cloud, vox_to_cloud_ind, _ = voxelate(cloud, res_ground, res_ground, with_n_points=False)
    # Cluster labels are appended to the FILTERED cloud. They map each point to
    # the cluster they belong to, according to the clustering algorithm.
    eps = res_ground * math.sqrt(3) + 1e-6
    cluster_labels = DBSCAN_clustering(vox_cloud, eps=eps, min_samples=min_points)

    cloud_labs = np.append(
        cloud,
        np.expand_dims(cluster_labels[vox_to_cloud_ind], axis=1),
        axis=1,
    )

    # Set of all cluster labels and their cardinality: cluster_id = {1,...,K},
    # K = 'number of points'.
    cluster_id, K = np.unique(cluster_labels, return_counts=True)

    # Filtering of labels associated only to clusters that contain a minimum
    # number of points.
    # ID = -1 is always created by DBSCAN to include points that were not
    # included in any cluster.
    large_clusters = cluster_id[(K > min_points) & (cluster_id != -1)]

    # Removing the points that are not in valid clusters.
    clust_cloud = cloud_labs[np.isin(cloud_labs[:, -1], large_clusters), :3]

    return clust_cloud


# -----------------------------------------------------------------------------
# classify_ground
# -----------------------------------------------------------------------------


def generate_dtm(
    cloud,
    bSloopSmooth=True,
    cloth_resolution=0.5,
    classify_threshold=0.1,
):
    """This function takes a point cloud and generates a Digital Terrain Model
    (DTM) based on its ground. It's based on 'Cloth Simulation Filter' by
    W. Zhang et al., 2016 (http://www.mdpi.com/2072-4292/8/6/501/htm),
    which is implemented in CSF package. This function just implements it in a
    convenient way for this use-case.

    Parameters
    ----------
    cloud : numpy.ndarray
        The point cloud. Matrix containing (x, y, z) coordinates of the points.
    bSloopSmooth : Boolean
        The resulting DTM will be smoothed. Refer to CSF documentation.
        Defaults to True.
    cloth_resolution : float
        The resolution of the cloth grid. Refer to CSF documentation. Defaults
        to 0.5.
    classify_threshold : float
        The height threshold used to classify the point cloud into ground and
        non-ground parts. Refer to CSF documentation. Defaults to 0.1.

    Returns
    -------
    cloth_nodes : numpy.ndarray
        Matrix containing (x, y, z) coordinates of the DTM points.
    """

    ### Cloth simulation filter ###
    csf = CSF.CSF()  # initialize the csf

    ### parameter settings ###
    csf.params.bSloopSmooth = bSloopSmooth
    csf.params.cloth_resolution = cloth_resolution
    # csf.params.rigidness # 1, 2 or 3
    csf.params.classify_threshold = classify_threshold  # default is 0.5 m

    csf.setPointCloud(cloud)  # pass the (x), (y), (z) list to csf

    raw_nodes = csf.do_cloth_export()  # do actual filtering and export cloth
    cloth_nodes = np.reshape(np.array(raw_nodes), (-1, 3))

    return cloth_nodes


# -----------------------------------------------------------------------------
# clean_cloth
# -----------------------------------------------------------------------------


def clean_cloth(dtm_points):
    """This function takes a Digital Terrain Model (DTM) and denoises it. This
    denoising is done via a 2 MADs criterion from the median height value of a
    neighborhood of size 15.

    Parameters
    ----------
    dtm_points : numpy.ndarray
        Matrix containing (x, y, z) coordinates of the DTM points.

    Returns
    -------
    clean_points : numpy.ndarray
        Matrix containing (x, y, z) coordinates of the denoised DTM points.
    """

    if dtm_points.shape[0] < 15:
        raise ValueError("input DTM is too small (less than 15 points). Denoising cannot be done.")
    if dtm_points.shape[0] == 15:
        warnings.warn(
            "input DTM contains exactly 15 points, which is the minimum input size accepted by clean_cloth().",
            stacklevel=2,
        )
    tree = KDTree(dtm_points[:, :2])
    _, indexes = tree.query(dtm_points[:, :2], 15, workers=-1)
    abs_devs = np.abs(dtm_points[:, 2] - np.median(dtm_points[:, 2][indexes], axis=1))
    mads = np.median(abs_devs)
    clean_points = dtm_points[abs_devs <= 2 * mads]

    return clean_points


# -----------------------------------------------------------------------------
# complete_dtm
# -----------------------------------------------------------------------------


def complete_dtm(dtm_points):
    """This function uses scipy.interpolate.griddata to interpolate the missing
    values in a Digital Terrain Model (DTM).

    Parameters
    ----------
    dtm_points : numpy array
        Matrix containing (x, y, z) coordinates of the DTM points.

    Returns
    -------
    completed_dtm : numpy.ndarray
        Matrix containing (x, y, z) coordinates of the completed DTM points.
    """

    # Separate x, y, z coordinates
    x = dtm_points[:, 0]
    y = dtm_points[:, 1]
    z = dtm_points[:, 2]

    # Generate a grid of points based on min x, y values
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate missing values using griddata
    zi = griddata((x, y), z, (xi, yi), method="cubic")

    # Combine interpolated points with existing points
    completed_dtm = np.hstack((xi.reshape(-1, 1), yi.reshape(-1, 1), zi.reshape(-1, 1)))

    # Remove nan values which may arise from interpolation
    completed_dtm = completed_dtm[~np.isnan(completed_dtm).any(axis=1)]

    return completed_dtm


# -----------------------------------------------------------------------------
# normalize_heights
# -----------------------------------------------------------------------------


def normalize_heights(cloud, dtm_points):
    """This function takes a point cloud and a Digital Terrain Model (DTM) and
    normalizes the heights of the first based on the second.

    Parameters
    ----------
    cloud : numpy.ndarray
        Matrix containing (x, y, z) coordinates of the point cloud.
    dtm_points : numpy array
        Matrix containing (x, y, z) coordinates of the DTM points.

    Returns
    -------
    zs_diff_triples : numpy.ndarray
        Vector containing the normalized height values for the cloud points.
    """

    tree = KDTree(dtm_points[:, :2])
    d, idx_pt_mesh = tree.query(cloud[:, :2], 3, workers=-1)
    # Z point cloud - Z dtm (Weighted average, based on distance)
    zs_diff_triples = cloud[:, 2] - np.average(dtm_points[:, 2][idx_pt_mesh], weights=d, axis=1)
    return zs_diff_triples


# -----------------------------------------------------------------------------
# check_normalization()
# -----------------------------------------------------------------------------


def check_normalization_discrepancy(cloud, original_area, res_xy=1.0, z_min=-0.1, z_max=0.15, warning_thresh=0.1):
    """Compare the area of a slice of points from a point cloud to another area and
    return a warning indicator if difference is greater than a certain threshold. The
    percentage of discrepancy between the too area is also returned. Area of the slice
    will be approximated from a voxelated version of it.

    Parameters
    ----------
    cloud : numpy.ndarray
        A 2D numpy array storing the point cloud. It must be a normalized point cloud.
    original_area : float
        Area to compare with.
    res_xy : float
        (x, y) voxel resolution. Defaults to 1.0 m.
    z_min: float
        The minimum Z value that defines the slice. Defaults to -0.10 m.
    z_max: float
        The maximum Z value that defines the slice. Defaults to 0.15 m.
    warning_thresh: float
        Threshold area difference. Defaults to 0.1 (10 % difference in area).

    Returns
    -------
    area_warning : bool
        True if area difference is greater than threshold, False if not.
    difference_percentage:
        The percentage of discrepancy between the original area and the slice area.
    """

    # (z) voxel resolution.
    if z_min > z_max:
        raise ValueError("z_min must be smaller than z_max")

    if z_min == z_max:
        raise ValueError("z_min and z_max must be different")

    # original_area
    if original_area <= 0:
        raise ValueError("Original area to compare with must be positive")

    # warning_threshold
    if not 0 < warning_thresh < 1:
        raise ValueError("warning_thresh must be larger than 0 and smaller than 1")

    # Compute the z resolution as a function of z_max - z_min
    res_z = (z_max - z_min) * 1.01

    # Select a slice of points from the cloud where Z value is within (z_min, z_max)
    ground_slice = cloud[(cloud[:, 2] >= z_min) & (cloud[:, 2] <= z_max)]

    # Voxelate the slice and store only cloud_to_vox_ind output for efficiency
    _, _, voxelated_slice = voxelate(ground_slice, res_xy, res_z, with_n_points=False, silent=False)

    # Area of the voxelated ground slice (n of voxels * area of voxel base)
    slice_area = voxelated_slice.shape[0] * res_xy**2

    # Calculate difference in area that breaks the threshold
    threshold_difference = warning_thresh * original_area

    # Calculate the absolute difference between the two areas
    area_difference = abs(original_area - slice_area)

    # TODO: In very rare occasions, the slice area could be larger than the original
    # area. The function should account for that, and return a different kind of
    # warning for those situations (and its threshold could be different).
    # For instance, if the original area has been computed through a grid of voxels
    # (as this function does to compute slice_area) using a smaller voxel size,
    # this could happen. We haven't tested it yet as we do not have access
    # to any point clouds where this situation happens.

    # Check if the difference is greater than 10 % of the first number
    area_warning = area_difference >= threshold_difference

    return area_warning, area_difference * 100 / original_area


def check_normalization(cloud, original_area, res_xy=1.0, z_min=-0.1, z_max=0.15, warning_thresh=0.1):
    """Compare the area of a slice of points from a point cloud to another area and
    store a warning indicator if difference is greater than a certain threshold. Area
    of the slice will be approximated from a voxelated version of it. This function is
    kept for backward compatibility and call check_normalization_discrepancy under the
    hood.

    Parameters
    ----------
    cloud : numpy.ndarray
        A 2D numpy array storing the point cloud. It must be a normalized point cloud.
    original_area : float
        Area to compare with.
    res_xy : float
        (x, y) voxel resolution. Defaults to 1.0 m.
    z_min: float
        The minimum Z value that defines the slice. Defaults to -0.10 m.
    z_max: float
        The maximum Z value that defines the slice. Defaults to 0.15 m.
    warning_thresh: float
        Threshold area difference. Defaults to 0.1 (10 % difference in area).

    Returns
    -------
    area_warning : bool
        True if area difference is greater than threshold, False if not.
    """
    indicator, _ = check_normalization_discrepancy(cloud, original_area, res_xy, z_min, z_max, warning_thresh)
    return indicator
