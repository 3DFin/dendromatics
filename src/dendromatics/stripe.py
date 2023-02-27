#### IMPORTS ####
import timeit
import jakteristics as jak
import numpy as np
from sklearn.cluster import DBSCAN

#-----------------------------------------------------------------------------------------------------------------------------------
# verticality_clustering
#-----------------------------------------------------------------------------------------------------------------------------------

def verticality_clustering_iteration(stripe, vert_scale, vert_treshold, eps, n_points, resolution_xy, resolution_z, n_digits):

    '''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    This function is to be used internally by verticality_clustering_iteration.
    The intended use of this function is to accept a stripe as an input, defined this as a subset of 
    the original cloud delimited by a lower height and an upper height, which will narrow down a region 
    where it is expected to only be stems. Then it will voxelate those points and compute the verticality
    via compute_features() from jakteristics. It will filter points based on their verticality value, 
    voxelate again and then cluster the remaining points. Those are expected to belong to stems.

    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    stripe: numpy array. The point cloud containing the stripe. It is expected to have X, Y, Z fields.
    vert_scale: float. Scale to be used during verticality computation to define a neighbourhood around 
    a given point. Verticality will be computed from the structure tensor of said neighbourhood via 
    eigendecomposition.
    vert_threshold: float. Minimum verticality value associated to a point to consider it as part of a stem.
    eps: float. Refer to DBSCAN documentation.
    n_points: int. Minimum number of points in a cluster for it to be considered as a potential stem.
    resolution_xy: float. (x, y) voxel resolution.
    resolution_z: float. (z) voxel resolution.
    n_digits: int. default value: 5. Number of digits dedicated to each coordinate ((x), (y) or (z))
    during the generation of each point code. If the cloud is really large, it can be advisable
    to increase n_digits.

    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    clust_stripe: numpy array. Point cloud containing the points from the stripe that are considered as stems. 
    It consists of 4 columns: (x), (y) and (z) coordinates, and a 4th column containing the cluster ID of the 
    cluster that each point belongs to.
    t1: float. Time spent.
    '''
    
    t = timeit.default_timer()
    print(" -Computing verticality...")
    
    # Call to 'voxelate' function to voxelate the cloud.
    voxelated_stripe, vox_to_stripe_ind, stripe_to_vox_ind = voxelate(stripe, resolution_xy, resolution_z, n_digits,with_n_points = False)
    
    # Computation of verticality values associated to voxels using 'compute_features' function. It needs a vicinity radius, provided by 'vert_scale'.
    vert_values = jak.compute_features(voxelated_stripe, search_radius = vert_scale, feature_names = ["verticality"])

    elapsed = timeit.default_timer() - t
    print("   %.2f" % elapsed,'s')
    t1 = elapsed
    
    # Verticality values are appended to the ORIGINAL cloud, using voxel-to-original-cloud indexes.
    vert_stripe = np.append(stripe, vert_values[vox_to_stripe_ind], axis = 1)
    
    # Filtering of points that were in voxels whose verticality value is under the threshold. Output is a filtered cloud.
    filt_stripe = vert_stripe[vert_stripe[:, -1] > vert_treshold]
    
    t = timeit.default_timer()
    print(" -Clustering...")
    
    # The filtered cloud is voxelated.
    vox_filt_stripe, vox_to_filt_stripe_ind, filt_stripe_to_vox_ind = voxelate(filt_stripe, resolution_xy, resolution_z, n_digits, with_n_points = False)    
    
    # Clusterization of the voxelated cloud obtained from the filtered cloud. 
    # 'eps': The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
    #        This is not a maximum bound on the distances of points within a cluster. 
    # min samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
    #              This includes the point itself.
    clustering = DBSCAN(eps = eps, min_samples = 2).fit(vox_filt_stripe)
    
    elapsed = timeit.default_timer() - t
    print("   %.2f" % elapsed,'s')
    t1 = elapsed + t1
    
    t = timeit.default_timer()
    print(" -Extracting 'candidate' stems...")
    
    # Cluster labels are appended to the FILTERED cloud. They map each point to the cluster they belong to, according to the clustering algorithm.
    vox_filt_lab_stripe = np.append(filt_stripe, np.expand_dims(clustering.labels_[vox_to_filt_stripe_ind], axis = 1), axis = 1)
    
    # Set of all cluster labels and their cardinality: cluster_id = {1,...,K}, K = 'number of clusters'.
    cluster_id, K = np.unique(clustering.labels_, return_counts = True)
    
    # Filtering of labels associated only to clusters that contain a minimum number of points.
    large_clusters = cluster_id[K > n_points]
    
    # ID = -1 is always created by DBSCAN() to include points that were not included in any cluster.
    large_clusters = large_clusters[large_clusters != -1]
    
    # Removing the points that are not in valid clusters.       
    clust_stripe = vox_filt_lab_stripe[np.isin(vox_filt_lab_stripe[:, -1], large_clusters)]
    
    n_clusters = large_clusters.shape[0]
    
    elapsed = timeit.default_timer() - t
    print("   %.2f" % elapsed,'s')
    t1 = elapsed + t1
    print("   %.2f" % t1, 's per iteration')
    print("   ", n_clusters, " clusters")
    return clust_stripe, t1



#-----------------------------------------------------------------------------------------------------------------------------------
# verticality_clustering_iteration
#----------------------------------------------------------------------------------------------------------------------------------------                  

def verticality_clustering(stripe, scale,vert_treshold, eps_dbscan, n_points, n_iter, resolution_xy, resolution_z, n_digits):

    '''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    This function implements a for loop that iteratively calls verticality_clustering_iteration, 
    'peeling off' the stems.


    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    stripe: numpy array. The point cloud containing the stripe. It is expected to have X, Y, Z fields.
    vert_scale: float. Scale to be used during verticality computation to define a neighbourhood around 
    a given point. Verticality will be computed from the structure tensor of said neighbourhood via 
    eigendecomposition.
    vert_threshold: float. Minimum verticality value associated to a point to consider it as part of a stem.
    eps: float. Refer to DBSCAN documentation.
    n_points: int. Minimum number of points in a cluster for it to be considered as a potential stem.
    n_iter: integer. Number of iterations of 'peeling'.
    resolution_xy: float. (x, y) voxel resolution.
    resolution_z: float. (z) voxel resolution.
    n_digits: int. default value: 5. Number of digits dedicated to each coordinate ((x), (y) or (z))
    during the generation of each point code. If the cloud is really large, it can be advisable
    to increase n_digits.


    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    clust_stripe: numpy array. Point cloud containing the points from the stripe that are considered as stems. 
    It consists of 4 columns: (x), (y) and (z) coordinates, and a 4th column containing the cluster ID of the 
    cluster that each point belongs to.
    '''
    
    # This first if loop is just a fix that allows to compute everything ignoring verticality.
    # It should be addressed as it currently computes verticality when n_iter = 0 and that should
    # not happen (although, in practice, n_iter should never be 0).
    # It does not provide wrong results but it slows down the process needlessly.
    if n_iter == 0:
        n_iter = 1
        vert_treshold = 0
    
    # Basically, use verticality_clustering as many times as defined by n_iter
    for i in np.arange(n_iter):
        print("Iteration number",i + 1,"out of", n_iter)
        if i == 0:
            total_t = 0
            aux_stripe = stripe
        else:
            aux_stripe = clust_stripe
        clust_stripe, t = verticality_clustering_iteration(aux_stripe, scale, vert_treshold, eps_dbscan, n_points, resolution_xy, resolution_z, n_digits)
        total_t = total_t + t
    print("Final:")            
    print("%.2f" % total_t, 's in total (whole process)')   
    return clust_stripe