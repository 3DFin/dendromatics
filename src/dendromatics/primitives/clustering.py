from sklearn.cluster import DBSCAN


def DBSCAN_clustering(xyz, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(xyz).labels_


# monkey patch clustering if possible
try:
    print("Using dendromatic with optimized C++ clustering")
    import dendroptimized

    DBSCAN_clustering = dendroptimized.connected_components
except ImportError:
    pass
