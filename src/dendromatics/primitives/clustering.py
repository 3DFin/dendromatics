from sklearn.cluster import DBSCAN


def DBSCAN_clustering(xyz, eps, min_samples):
    """
    Cluster 3D dimensional point clouds.

    It's essentially a wrapper around scikit-learn's DBSCAN.
    If `dendroptimized` is installed, an optimized version is used instead.
    However, this optimized version has an additional constraint:
    the point cloud must be regularly sampled, and `eps` must equal
    the sampling resolution of the point cloud x sqrt(3) i.e. the diagonal
    of a voxel (the enclosing ball of the voxel).

    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud to be denoised. Matrix containing (x, y, z) coordinates
        of the points.

    eps : float
        The clustering radius.

    min_samples : int
        The number of neighbors for a point to be considered as a core point.
        This includes the point itself.

    Returns
    -------
    labels : numpy.ndarray
        Cluster labels for each point. Noisy samples are labeled "-1".
    """
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(xyz).labels_


# monkey patch clustering if possible
try:
    import dendroptimized

    DBSCAN_clustering = dendroptimized.connected_components
except ImportError:
    pass
