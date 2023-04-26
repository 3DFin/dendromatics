#### IMPORTS ####
import CSF
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .voxel.voxel import *

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

    vox_cloud, vox_to_cloud_ind, cloud_to_vox_ind = voxelate(
        cloud, res_ground, res_ground, with_n_points=False
    )
    # Cluster labels are appended to the FILTERED cloud. They map each point to
    # the cluster they belong to, according to the clustering algorithm.
    clustering = DBSCAN(eps=0.3, min_samples=min_points).fit(vox_cloud)

    cloud_labs = np.append(
        cloud, np.expand_dims(clustering.labels_[vox_to_cloud_ind], axis=1), axis=1
    )

    # Set of all cluster labels and their cardinality: cluster_id = {1,...,K},
    # K = 'number of points'.
    cluster_id, K = np.unique(clustering.labels_, return_counts=True)

    # Filtering of labels associated only to clusters that contain a minimum
    # number of points.
    large_clusters = cluster_id[K > min_points]

    # ID = -1 is always created by DBSCAN() to include points that were not
    # included in any cluster.
    large_clusters = large_clusters[large_clusters != -1]

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
    neighbourhood of size 15.

    Parameters
    ----------
    dtm_points : numpy.ndarray
        Matrix containing (x, y, z) coordinates of the DTM points.

    Returns
    -------
    clean_points : numpy.ndarray
        Matrix containing (x, y, z) coordinates of the denoised DTM points.
    """

    tree = cKDTree(dtm_points[:, :2])
    _, indexes = tree.query(dtm_points[:, :2], 15)
    abs_devs = np.abs(dtm_points[:, 2] - np.median(dtm_points[:, 2][indexes], axis=1))
    mads = np.median(abs_devs)
    clean_points = dtm_points[abs_devs < 2 * mads]

    return clean_points


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

    tree = cKDTree(dtm_points[:, :2])
    d, idx_pt_mesh = tree.query(cloud[:, :2], 3)
    # Z point cloud - Z dtm (Weighted average, based on distance)
    zs_diff_triples = cloud[:, 2] - np.average(
        dtm_points[:, 2][idx_pt_mesh], weights=d, axis=1
    )
    return zs_diff_triples
