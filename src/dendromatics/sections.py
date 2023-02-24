#### IMPORTS ####
import sys
import laspy
import numpy as np
from scipy import optimize as opt
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance_matrix

#-----------------------------------------------------------------------------------------------------------------------------------
# point_clustering
#----------------------------------------------------------------------------------------------------------------------------------------                  

def point_clustering(X, Y, max_dist):      

    '''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    This function clusters points by distance and finds the largest cluster.
    It will be used during circle fitting stage.

    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    X: numpy array. Vector containing (x) coordinates of points belonging to a tree section. 
    Y: numpy array. Vector containing (y) coordinates of points belonging to a tree section. 
    max_dist: float. Max separation among the points to be considered as members of the same cluster.
    
    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    X_g: numpy array. Vector containing the (x) coordinates of the largest cluster.
    Y_g: numpy array. Vector containing the (y) coordinates of the largest cluster.
    '''
    
    # Stacks 1D arrays ([X], [Y]) into a 2D array ([X, Y])
    xy_stack = np.column_stack((X, Y))
    
    # fclusterdata outputs a vector that contains cluster ID of each point (which cluster does each point belong to)
    clust_id = sch.fclusterdata(xy_stack, max_dist, criterion = 'distance', metric = 'euclidean')
    
    
    # Set of all clusters
    clust_id_unique = np.unique(clust_id)
    
    # For loop that iterates over each cluster ID, sums its elements and finds the largest
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
 


#-------------------------------------------------------------------------------------------------------------------------------------------------------
# fit_circle
#-------------------------------------------------------------------------------------------------------------------------------------------------------

def fit_circle(X, Y):

    '''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    This function fits points within a tree section into a circumference by least squares minimization.
    Its intended inputs are X, Y coordinates of points belonging to a section.

    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    X: numpy array. Vector containing (x) coordinates of points belonging to a tree section. 
    Y: numpy array. Vector containing (y) coordinates of points belonging to a tree section. 
    
    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    circle_c: numpy array. Matrix containing the (x, y) coordinates of the circumference center.
    mean_radius: numpy array. Vector containing the radius of each fitted circumference.
    '''
    
    # Function that computes distance from each 2D point to a single point defined by (X_c, Y_c)
    # It will be used to compute the distance from each point to the circumference center.
    def calc_R(X, Y, X_c, Y_c):
        return np.sqrt((X - X_c) ** 2 + (Y - Y_c) ** 2)

    # Function that computes algebraic distance from each 2D point to some middle circle c
    # It calls calc_R (just defined above) and it is used during the least squares optimization.
    def f_2(c, X, Y):
        R_i = calc_R(X, Y, *c)
        return R_i - R_i.mean()

    # Initial barycenter coordinates (middle circle c center)
    X_m = X.mean()
    Y_m = Y.mean()
    barycenter = X_m, Y_m
  
  # Least square minimization to find the circumference that best fits all points within the section.
    circle_c, ier = opt.leastsq(f_2, barycenter, args = (X, Y)) # ier is a flag indicating whether the solution was found (ier = 1, 2, 3 or 4) or not (otherwise).
    X_c, Y_c = circle_c

  # Its radius 
    radius = calc_R(X, Y, *circle_c)
    mean_radius = radius.mean()
  
  # Output: - X, Y coordinates of best-fit circumference center - its radius
    return (circle_c, mean_radius)



#--------------------------------------------------------------------------------------------------------------------------------------------------------  
# inner_circle      
#--------------------------------------------------------------------------------------------------------------------------------------------------------

def inner_circle(X, Y, X_c, Y_c, R, times_R):

    '''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    Function that computes an internal circumference inside the one fitted by fit_circle (the one that best fits all points within a section by least squares minimization).
    This new circumference is used as a validation tool: it gives insight on the quality of the 'fit_circle-circumference':
      - If points are closest to the inner circumference, then the first fit was not appropiate
      - On the contrary, if points are closer to the outer circumference, the 'fit_circle-circumference' is appropiate and describes well the stem diameter.
    Instead of directly computing the inner circle, it just takes a proportion (less than one) of the original circumference radius and its center.
    After this, it just checks how many points are closest to the inner circle than to the original circumference.

    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    X: numpy array. Vector containing (x) coordinates of points belonging to a tree section. 
    Y: numpy array. Vector containing (y) coordinates of points belonging to a tree section. 
    X_c: numpy array. Vector containing (x) coordinates of fitted circumferences.
    Y_c: numpy array. Vector containing (y) coordinates of fitted circumferences.
    R: numpy array. Vector containing the radia of the fitted circumferences.
    
    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    n_points_in: numpy array. Vector containing the number of points inside the inner circle of each section.
    '''
    
    # Distance from each 2D point to the center. 
    distance = np.sqrt((X - X_c) ** 2 + (Y - Y_c) ** 2) 
    
    # Number of points closest to the inner circumference, whose radius is proportionate to the outer circumference radius by a factor defined by 'times_R'.
    n_points_in = np.sum(distance < R * times_R) 
    
    # Output: Number of points closest to the inner circumference. 
    return n_points_in



#-------------------------------------------------------------------------------------------------------------------------------------------------------
# sector_occupancy
#-------------------------------------------------------------------------------------------------------------------------------------------------------

def sector_occupancy(X, Y, X_c, Y_c, R, n_sectors = 16, min_n_sectors = 9, width = 2.0):

    '''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    This function provides quality measurements for the fitting of the circle.
    It divides the section in a number of sectors to check if there are points within them 
    (so they are occupied). It is divided in 16 sectors by default.
    If there are not enough occupied sectors, the section fails the test, as it is safe to asume it has an anomale, non desirable structure.

    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    X: numpy array. Vector containing (x) coordinates of points belonging to a tree section. 
    Y: numpy array. Vector containing (y) coordinates of points belonging to a tree section. 
    X_c: numpy array. Vector containing (x) coordinates of fitted circumferences.
    Y_c: numpy array. Vector containing (y) coordinates of fitted circumferences.
    R: numpy array. Vector containing the radia of the fitted circumferences.
    n_sectors: int. default value: 16. Number of sectors in which the sections will be divided 
    min_n_sectors: int. default value: 9. Minimum number of occupied sectors in a section for its fitted circumference to be considered as valid.
    width: float. default value: 2.0. Width (cm) around the fitted circumference to look for points.
    
    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    perct_occuped_sectors: numpy array. Vector containing the percentage of occupied sectors in each section.
    enough_occuped_sectors: numpy array. Vector containing binary indicators whether the fitted circle is valid or not:
    1 - valid; 0 - not valid.
    '''
    
    # Coordinates translation.
    X_red = X - X_c
    Y_red = Y - Y_c
    
    # Computation of radius and angle necessary to transform cartesian coordinates to polar coordinates. 
    radial_coord  = np.sqrt(X_red ** 2 + Y_red ** 2) # radial coordinate
    angular_coord = np.arctan2(X_red, Y_red) # angular coordinate. This function from numpy directly computes it. 
    
    # Points that are close enough to the circumference that will be checked.
    points_within = (radial_coord > (R - width / 100)) * (radial_coord < (R + width / 100))
    
    # Codification of points in each sector. Basically the range of angular coordinates is divided in n_sector pieces and granted an integer number.
    # Then, every point is assigned the integer corresponding to the sector it belongs to.
    norm_angles = np.floor(angular_coord[points_within] / (2 * np.pi / n_sectors)) # np.floor se queda solo con la parte entera de la división
    
    # Number of points in each sector. 
    n_occuped_sectors = np.size(np.unique(norm_angles)) 
    
    # Percentage of occupied sectors.
    perct_occuped_sectors = n_occuped_sectors * 100 / n_sectors
    
    # If there are enough occupied sectors, then it is a valid section.
    if n_occuped_sectors < min_n_sectors:
        enough_occuped_sectors = 0
    
    # If there are not, then it is not a valid section.
    else:
        enough_occuped_sectors = 1
    
    # Output: percentage of occuped sectors | boolean indicating if it has enough occuped sectors to pass the test.
    return (perct_occuped_sectors, enough_occuped_sectors)# 0: no pasa; 1: pasa el test



#-------------------------------------------------------------------------------------------------------------------------------------
# fit_circle_check
#----------------------------------------------------------------------------------------------------------------------------------------

def fit_circle_check(X, Y, review, second_time, times_R, threshold, R_min, R_max, max_dist, n_points_section, n_sectors = 16, min_n_sectors = 9, width = 2):
     
    '''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    This function calls fit_circle() to fit points within a section to a circumference by least squares
    minimization. These circumferences will define tree sections. It checks the goodness of fit 
    using the functions defined above. If fit is not appropriate, another circumference 
    will be fitted using only points from the largest cluster inside the first circumference. 
    
    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    X: numpy array. Vector containing (x) coordinates of points belonging to a tree section. 
    Y: numpy array. Vector containing (y) coordinates of points belonging to a tree section. 
    second_time: numpy array. Vector containing integers that indicates whether it is the first
    time a circle is fitted or not (will be modified internally).
    times_R: float. Ratio of radius between outer circumference and inner circumference.
    threshold: float. Minimum number of points in inner circumference for a fitted circumference to be valid.
    R_min: float. Minimum radius that a fitted circumference must have to be valid.
    R_max: float. Maximum radius that a fitted circumference must have to be valid.
    max_dist: float. Refer to point_clustering.
    n_points_section: int. Minimum points within a section for its fitted circumference to be valid.
    n_sectors: int. default value: 16. Number of sectors in which the sections will be divided 
    min_n_sectors: int. default value: 9. Minimum number of occupied sectors in a section for its fitted circumference to be considered as valid.
    width: float. default value: 2.0. Width (cm) around the fitted circumference to look for points.
    
    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    X_gs: numpy array. Matrix containing (x) coordinates of largest clusters.
    Y_gs: numpy array. Matrix containing (y) coordinates of largest clusters.
    X_c: numpy array. Matrix containing (x) coordinates of the center of the best-fit circumferences.
    Y_c: numpy array. Matrix containing (y) coordinates of the center of the best-fit circumferences.
    R: numpy array. Vector containing best-fit circumference radia.
    section_perct: numpy array. Matrix containing the percentage of occupied sectors.
    n_points_in: numpy array. Matrix containing the number of points in the inner circumferences.
    '''
    
    # If loop that discards sections that do not have enough points (n_points_section)
    if X.size > n_points_section:
      
        # Call to fit_circle to fit the circumference that best fits all points within the section. 
        (circle_center, R) = fit_circle(X = X, Y = Y)
        X_c = circle_center[0] # Column 0 is center X coordinate 
        Y_c = circle_center[1] # Column 1 is center Y coordinate 
        
        # Call to inner_circle to fit an inner circumference and to get the number of points closest to it. 
        n_points_in = inner_circle(X, Y, X_c, Y_c, R, times_R)

        # Call to sector_occupancy to check if sectors around inner circumference are occupied.
        (sector_perct, enough_sectors) = sector_occupancy(X, Y, X_c, Y_c, R, n_sectors, min_n_sectors, width)
        
        # If any of the following conditions hold:
        #   - Too many points in inner circle
        #   - Radius of best-fit circle is too small
        #   - Number of occupied sectors is too low
        # Then proceed with countermeasures
        if n_points_in > threshold or R < R_min or R > R_max or enough_sectors == 0:
            
            # If this is not the second round or, simply, if it is the first round, then proceed
            if second_time == 0:
                 
                 # First round implies there is no X_g or Y_g, as points would not have been grouped yet. point_clustering is called.
                (X_g, Y_g) = point_clustering(X, Y, max_dist) #X_g or Y_g are the coordinates of the largest cluster.
                
                # If cluster size is big enough, then proceed. It is done this way to account for cases where, even though the section had enough points,
                # there might not be enough points within the largest cluster.
                if X_g.size > n_points_section: 
    
                     # Call to fit_circle_check (lets call it the 'deep call'). Now it is guaranteed that it is a valid section (has enough points and largest cluster has enough points as well).
                    (X_c, Y_c, R, review, second_time, sector_perct, n_points_in) = fit_circle_check(X_g, Y_g, 0, 1, times_R, threshold, R_min, R_max, max_dist, n_points_section, n_sectors, min_n_sectors, width)
                    
                # If cluster size is not big enough, then don't take the section it belongs to into account. 
                else:
                    review = 1 # Even if it is not a valid section, lets note it has been checked.
                    X_c = 0
                    Y_c = 0
                    R = 0
                    second_time = 1
            
            # If this is the second round (whether the first round succesfully provided a valid section or not), then proceed.        
            else:
                review = 1 # Just stating that if this is the second round, the check has, obviously, happened.
    
    # This matches the first loop. If section is not even big enough (does not contain enough points), it is not valid.
    else:
        review = 2
        X_c = 0
        Y_c = 0
        R = 0
        second_time = 2
        sector_perct = 0
        n_points_in = 0
    
    # Output is basically the one obtained during the 'deep call', if the section was valid.
    # If not, it is basically empty values (actually zeros).
    # X_gs, Y_gs are actually never used again, they are kept just in case they would become useful in a future update.
    # X_c, Y_c are the coordinates of the center of the best-fit circumference an R its radius.
    # section_perct is the percentage of occupied sectors, and n_points_in is the number of points closest to the inner circumference (quality measurements).
    return X_c, Y_c, R, review, second_time, sector_perct, n_points_in
    


#-------------------------------------------------------------------------------------------------------------------------------------------------------
# compute_sections
#-------------------------------------------------------------------------------------------------------------------------------------------------------

'''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    This function calls fit_circle_check() to compute stem diameter at given sections.
    
    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    stems: numpy array. Point cloud containing the individualized trees. It is expected to have X, Y, Z0, tree_ID fields.
    sections: numpy array. Matrix containing a range of height values at which sections will be computed.
    section_width: float. Points within this distance (in meters) from any `sections` value will be considered as belonging to said section.
    times_R: float. Refer to fit_circle_check.
    threshold: float. Refer to fit_circle_check.
    R_min: float. Refer to fit_circle_check.
    R_max: float. Refer to fit_circle_check.
    max_dist: float. Refer to point_clustering.
    n_points_section: int. Refer to fit_circle_check.
    n_sectors: int. default value: 16. Refer to fit_circle_check.
    min_n_sectors: int. default value: 9. Refer to fit_circle_check.
    width: float. default value: 2.0. Refer to fit_circle_check.
    
    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    X_c: numpy array. Matrix containing (x) coordinates of the center of the best-fit circumferences.
    Y_c: numpy array. Matrix containing (y) coordinates of the center of the best-fit circumferences.
    R: numpy array. Vector containing best-fit circumference radia.
    section_perct: numpy array. Matrix containing the percentage of occupied sectors.
    n_points_in: numpy array. Matrix containing the number of points in the inner circumferences.
    '''
    
def compute_sections(stems, sections, section_width = 2, times_R = 0.5, threshold = 5, R_min = 0.03, R_max = 0.5, max_dist = 0.02, n_points_section = 80, n_sectors = 16, min_n_sectors = 9, width = 2):
    
    X_field = 0
    Y_field = 1
    Z0_field = 3
    tree_id_field = 4
    
    trees = np.unique(stems[:, tree_id_field]) # Select the column that contains tree ID
    n_trees = trees.size # Number of trees
    n_sections = sections.size  # Number of sections
        
    X_c            = np.zeros((n_trees, n_sections), dtype = float) # Empty array to store X data 
    Y_c            = np.zeros((n_trees, n_sections), dtype = float) # Empty array to store Y data
    R              = np.zeros((n_trees, n_sections), dtype = float) # Empty array to store radius data
    check_circle   = np.zeros((n_trees, n_sections), dtype = float) # Empty array to store 'check' data
    second_time    = np.zeros((n_trees, n_sections), dtype = float) # Empty array to store 'second_time' data
    sector_perct   = np.zeros((n_trees, n_sections), dtype = float) # Empty array to store percentage of occuped sectors data
    n_points_in    = np.zeros((n_trees, n_sections), dtype = float) # Empty array to store inner points data
    
    # Filling previous empty arrays

    # Auxiliar index for first loop
    tree = -1 # Loop will start at -1

    # First loop: iterates over each tree
    for tr in trees: 
        
        # Tree ID is used to iterate over trees
        tree_i = stems[stems[:, tree_id_field] == tr, :]
        tree = tree + 1 
        
        sys.stdout.write("\r%d%%" % np.float64((trees.shape[0] - tree) * 100 / trees.shape[0]))
        sys.stdout.flush()
        
        # Auxiliar index for second loop
        section = 0 
        
        # Second loop: iterates over each section
        for b in sections: 
            
            # Selecting (x, y) coordinates of points within the section
            X = tree_i[(tree_i[:, Z0_field] >= b) & (tree_i[:, Z0_field] < b + section_width), X_field]
            Y = tree_i[(tree_i[:, Z0_field] >= b) & (tree_i[:, Z0_field] < b + section_width), Y_field]
            
            # fit_circle_check call. It provides data to fill the empty arrays  
            (X_c_fill, Y_c_fill, R_fill, check_circle_fill, second_time_fill, sector_perct_fill, n_points_in_fill) = fit_circle_check(X, Y, 0, 0, times_R, threshold, R_min, R_max, max_dist, n_points_section, n_sectors, min_n_sectors, width)

            # Filling the empty arrays
            X_c[tree, section] = X_c_fill
            Y_c[tree, section] = Y_c_fill
            R[tree, section] = R_fill
            check_circle[tree, section] = check_circle_fill
            second_time[tree, section] = second_time_fill
            sector_perct[tree, section] = sector_perct_fill
            n_points_in[tree, section] = n_points_in_fill
            
            section = section + 1
    
    return(X_c, Y_c, R, check_circle, second_time, sector_perct, n_points_in)





#-------------------------------------------------------------------------------------------------------------------------------------------------------
# tilt_detection
#-------------------------------------------------------------------------------------------------------------------------------------------------------

# relat_peso_outliers_suma_inclinaciones = w_1
# relat_peso_outliers_relativos = w_2


def tilt_detection(X_tree, Y_tree, radius, sections, Z_field = 2, w_1 = 3.0, w_2 = 1.0):

    '''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    This function finds outlier tilting values among sections within a tree and assigns a score to the sections based on those outliers.
    There are two kinds of outliers: absolute and relative outliers.
    Absolute outliers are obtained from the sum of the deviations from every section center to all axes within a tree (the most tilted sections relative to all axes)
    Relative outliers are obtained from the deviations of other section centers from a certain axis, within a tree (the most tilted sections relative to a certain axis)
    The 'outlier score' consists on a weighted sum of the absolute tilting value and the relative tilting value.
    
    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    X_tree: numpy array. Matrix containing (x) coordinates of the center of the sections.
    Y_tree: numpy array. Matrix containing (y) coordinates of the center of the sections. 
    radius: numpy array. Vector containing section radia.
    sections: numpy array. Vector containing the height of the section associated to each section.
    Z_field: int. default value: 2. Index at which (z) coordinate is stored.
    w_1: float. default value: 3.0. Weight of absolute deviation.
    w_2: float. default value: 1.0. Weight of relative deviation.
    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    outlier_prob: numpy array. Vector containing the 'outlier probability' of each section.
    '''
    
    # This function simply defines 1st and 3rd cuartile of a vector and separates values that are outside the interquartilic range
    # defined by these. Those are the candidates to be outliers. This filtering may be done either directly from the interquartilic range,  
    # or from a certain distance from it, thanks to 'n_range' parameter. Its default value is 1.5.
    
    def outlier_vector(vector, lower_q = 0.25, upper_q = 0.75, n_range = 1.5):
        
        q1 = np.quantile(vector, lower_q) # First quartile
        q3 = np.quantile(vector, upper_q) # Third quartile
        iqr = q3 - q1 # Interquartilic range

        lower_bound = q1 - iqr * n_range # Lower bound of filter. If n_range = 0 -> lower_bound = q1
        upper_bound = q3 + iqr * n_range # Upper bound of filter. If n_range = 0 -> upper_bound = q3
        
        # Outlier vector.
        outlier_ind = (vector < lower_bound) | (vector > upper_bound) * 1
        return outlier_ind

    # Empty matrix that will store the probabilities of a section to be invalid
    outlier_prob = np.zeros_like(X_tree)
    
    # First loop: iterates over each tree
    for i in range(X_tree.shape[0]):
        
        # If there is, at least, 1 circle with positive radius in a tree, then proceed (invalid circles are stored with a radius value of 0)
        if np.sum(radius[i, :]) > 0:
            
            # Filtering sections within a tree that have valid circles (non-zero radius).
            valid_radius = radius[i, :] > 0
            
            # Weights associated to each section. They are computed in a way that the final value of outliers sums up to 1 as maximum.  
            abs_outlier_w = w_1 / (np.size(sections[valid_radius]) * w_2 + w_1)
            rel_outlier_w = w_2 / (np.size(sections[valid_radius]) * w_2 + w_1)
    
             
            # Vertical distance matrix among all sections (among their centers)
            heights = np.zeros((np.size(sections[valid_radius]), Z_field)) # Empty matrix to store heights of each section
            heights[:, 0] = np.transpose(sections[valid_radius]) #  Height (Z value) of each section
            z_dist_matrix = distance_matrix(heights, heights) # Vertical distance matrix
    
            # Horizontal distance matrix among all sections (among their centers)
            c_coord = np.zeros((np.size(sections[valid_radius]), 2))  # Empty matrix to store X, Y coordinates of each section
            c_coord[:, 0] = np.transpose(X_tree[i][valid_radius]) # X coordinates
            c_coord[:, 1] = np.transpose(Y_tree[i][valid_radius]) # Y coordinates
            xy_dist_matrix = distance_matrix(c_coord, c_coord) # Horizontal distance matrix
            
            # Tilting measured from every vertical within a tree: All verticals obtained from the set of sections within a tree.
            # For instance, if there are 10 sections, there are 10 tilting values for each section.
            tilt_matrix = np.arctan(xy_dist_matrix / z_dist_matrix) * 180 / np.pi
            
            # Summation of tilting values from each center.
            tilt_sum = np.nansum(tilt_matrix, axis = 0)
            
            # Outliers within previous vector (too low / too high tilting values). These are anomalus tilting values from ANY axis. 
            outlier_prob[i][valid_radius] = outlier_vector(tilt_sum) * abs_outlier_w
            
            # Second loop: iterates over each section (within a single tree).
            for j in range(np.size(sections[valid_radius])):
                
                # Search for anomalous tilting values from a CERTAIN axis. 
                tilt_matrix[j, j] = np.quantile(tilt_matrix[j, ~j], 0.5)
                rel_outlier = outlier_vector(tilt_matrix[j]) * rel_outlier_w # Storing those values.
                
                # Sum of absolute outlier value and relative outlier value
                outlier_prob[i][valid_radius] = outlier_prob[i][valid_radius] + rel_outlier
    
    # Output: Oulier value: 'Outlier probability' of each section based on its tilting 
    return outlier_prob



#-------------------------------------------------------------------------------------------------------------------------------------------------------
# tree_locator
#-------------------------------------------------------------------------------------------------------------------------------------------------------

def tree_locator(sections, X_c, Y_c, tree_vector, sector_perct, R, outliers, n_points_in, filename_las, threshold = 5, X_field = 0, Y_field = 1, Z_field = 2):

    '''
    -----------------------------------------------------------------------------
    ------------------           General description           ------------------
    -----------------------------------------------------------------------------

    This function generates points that locate the individualized trees and computes
    their DBH (diameter at breast height). It uses all the quality measurements defined in previous
    functions to check whether the DBH should be computed or not and to check which point should
    be used as the tree locator.
    
    The tree locators are then saved in a LAS file. Each tree locator corresponds on a one-to-one basis to the individualized trees.
    
    -----------------------------------------------------------------------------
    ------------------                 Inputs                  ------------------
    -----------------------------------------------------------------------------

    sections: numpy array. Vector containing section heights (normalized heights).
    X_c: numpy array. Matrix containing (x) coordinates of the center of the sections.
    Y_c: numpy array. Matrix containing (y) coordinates of the center of the sections. 
    tree_vector: numpy array. detected_trees output from individualize_trees.
    sector_perct: numpy array. Matrix containing the percentage of occupied sectors.
    R: numpy array. Vector containing section radia.
    outliers: numpy array. Vector containing the 'outlier probability' of each section.
    filename_las: char. File name for the output file.
    n_points_in: numpy array. Matrix containing the number of points in the inner circumferences.
    threshold: float. Minimum number of points in inner circumference for a fitted circumference to be valid.
    X_field: int. default value: 0. Index at which (x) coordinate is stored.
    Y_field: int. default value: 1. Index at which (y) coordinate is stored.
    Z_field: int. default value: 2. Index at which (z) coordinate is stored.


    -----------------------------------------------------------------------------
    -----------------                 Outputs                  ------------------
    -----------------------------------------------------------------------------

    Output is a LAS file containing the axes and two objects:
    dbh_values: numpy array. Vector containing DBH values.
    tree_locations: numpy array. matrix containing (x, y, z) coordinates of each tree_locator.
    '''
    
    dbh = 1.3 # Breast height
    
    tree_locations = np.zeros(shape = (X_c.shape[0], 3)) #Empty vector to be filled with tree locators
    n_trees = tree_locations.shape[0] # Number of trees
    
    dbh_values = np.zeros(shape = (X_c.shape[0], 1)) # Empty vector to be filled with DBH values.
    
    # This if loop covers the cases where the stripe was defined in a way that it did not include BH
    # and DBH nor tree locator cannot be obtained from a section at or close to BH. If that happens, tree axis is used
    # to locate the tree and DBH is not computed.
    if np.min(sections) > 1.3:
        
      for i in range(n_trees): 
          
          if tree_vector[i, 3] < 0:
              
              vector = -tree_vector[i, 1:4]
          
          else:
              
              vector = tree_vector[i, 1:4]
          
          diff_height = dbh - tree_vector[i, 6] + tree_vector[i, 7]  # Compute the height difference between centroid and BH
          dist_centroid_dbh = diff_height / np.cos(tree_vector[i, 8] * np.pi / 180)  # Compute the distance between centroid and axis point at BH.      
          tree_locations[i, :] = vector * dist_centroid_dbh + tree_vector[i, 4:7] # Compute coordinates of axis point at BH. 

    else:
          
        d = 1
        diff_to_dbh = sections - dbh # Height difference between each section and BH.
        which_dbh = np.argmin(np.abs(diff_to_dbh)) # Which section is closer to BH.
      
        # get surrounding sections too
        lower_d_section = which_dbh - d
        upper_d_section = which_dbh + d
        
        # Just in case they are out of bound
        if lower_d_section < 0:
            
            lower_d_section = 0
        
        if upper_d_section > sections.shape[0]:
            
            upper_d_section = sections.shape[0]
            
        # BH section and its neighbours. From now on, neighbourhood
        close_to_dbh = np.array(np.arange(lower_d_section, upper_d_section)) 
     
        for i in range(n_trees): # For each tree
        
            which_valid_R = R[i, close_to_dbh] > 0 # From neighbourhood, select only those with non 0 radius
            which_valid_out = outliers[i, close_to_dbh] < 0.30 #From neighbourhood, select only those with outlier probability lower than 10 %
            which_valid_sector_perct = sector_perct[i, close_to_dbh] > 30 # only those with sector occupancy higher than 30 %
            which_valid_points = n_points_in[i, close_to_dbh] > threshold # only those with enough points in inner circle
            
            # If there are valid sections among the selected
            if (np.any(which_valid_R)) & (np.any(which_valid_out)):
                
                # If first section is BH section and if itself and its only neighbour are valid
                if (lower_d_section == 0) & (np.all(which_valid_R)) & (np.all(which_valid_out)) & np.all(which_valid_sector_perct): # only happens when which_dbh == 0 # which_valid_points should be used here
                
                    # If they are coherent: difference among their radia is not larger than 10 % of the largest radius
                    if np.abs(R[i, close_to_dbh[0]] - R[i, close_to_dbh[1]]) < np.max(R[i, close_to_dbh]) * 0.1:
                
                        dbh_values[i] = R[i, which_dbh] * 2
        
                        tree_locations[i, X_field] = X_c[i, which_dbh].flatten() # Their centers are averaged and we keep that value
                        tree_locations[i, Y_field] = Y_c[i, which_dbh].flatten() # Their centers are averaged and we keep that value
                        tree_locations[i, Z_field] = tree_vector[i, 7] + dbh # original height is obtained
                    
                    # If not all of them are valid, then there is no coherence in any case, and the axis location is used
                    else:
                        
                        if tree_vector[i, 3] < 0:
                        
                            vector = -tree_vector[i, 1:4]
                    
                        else:
                        
                            vector = tree_vector[i, 1:4]
                    
                        diff_height = dbh - tree_vector[i, 6] + tree_vector[i, 7]  # Compute the height difference between centroid and BH
                        dist_centroid_dbh = diff_height / np.cos(tree_vector[i, 8] * np.pi / 180)  # Compute the distance between centroid and axis point at BH.      
                        tree_locations[i, :] = vector * dist_centroid_dbh + tree_vector[i, 4:7] # Compute coordinates of axis point at BH. 

                # If last section is BH section and if itself and its only neighbour are valid    
                elif (upper_d_section == sections.shape[0]) & (np.all(which_valid_R)) & (np.all(which_valid_out)):
                
                        # if they are coherent
                        if np.abs(R[i, close_to_dbh[0]] - R[i, close_to_dbh[1]]) < np.max(R[i, close_to_dbh]) * 0.15:
                    
                            # use BH section diameter as DBH
                            dbh_values[i] = R[i, which_dbh] * 2
            
                            tree_locations[i, X_field] = X_c[i, which_dbh].flatten() # use its center x value as x coordinate of tree locator
                            tree_locations[i, Y_field] = Y_c[i, which_dbh].flatten() # use its center y value as y coordinate of tree locator
                            tree_locations[i, Z_field] = tree_vector[i, 7] + dbh
                        
                        # If not all of them are valid, then there is no coherence in any case, and the axis location is used and DBH is not computed
                        else:
                            
                            if tree_vector[i, 3] < 0:
                            
                                vector = -tree_vector[i, 1:4]
                        
                            else:
                            
                                vector = tree_vector[i, 1:4]
                        
                            dbh_values[i] = 0
                            
                            diff_height = dbh - tree_vector[i, 6] + tree_vector[i, 7]  # Compute the height difference between centroid and BH
                            dist_centroid_dbh = diff_height / np.cos(tree_vector[i, 8] * np.pi / 180)  # Compute the distance between centroid and axis point at BH.      
                            tree_locations[i, :] = vector * dist_centroid_dbh + tree_vector[i, 4:7] # Compute coordinates of axis point at BH. 

                # In any other case, BH section is not first or last section, so it has 2 neighbourghs
                # 3 posibilities left: 
                # A: Not all of three sections are valid: there is no possible coherence
                # B: All of three sections are valid, and there is coherence among the three
                # C: All of three sections are valid, but there is only coherence among neighbours and not BH section or All of three sections are valid, but there is no coherence
                else:
                
                    # Case A:
                    if not ((np.all(which_valid_R)) & (np.all(which_valid_out)) & np.all(which_valid_sector_perct)):
                        
                        if tree_vector[i, 3] < 0:
                        
                            vector = -tree_vector[i, 1:4]
                    
                        else:
                        
                            vector = tree_vector[i, 1:4]
                    
                        dbh_values[i] = 0
                        
                        diff_height = dbh - tree_vector[i, 6] + tree_vector[i, 7]  # Compute the height difference between centroid and BH
                        dist_centroid_dbh = diff_height / np.cos(tree_vector[i, 8] * np.pi / 180)  # Compute the distance between centroid and axis point at BH.      
                        tree_locations[i, :] = vector * dist_centroid_dbh + tree_vector[i, 4:7] # Compute coordinates of axis point at BH. 

                
                    else:
                        
                        valid_sections = close_to_dbh # Valid sections indexes
                        valid_radia = R[i, valid_sections] # Valid sections radia
                        median_radius = np.median(valid_radia) # Valid sections median radius
                        abs_dev = np.abs(valid_radia - median_radius) # Valid sections absolute deviation from median radius
                        mad = np.median(abs_dev) # Median absolute deviation
                        filtered_sections = valid_sections[abs_dev < 3 * mad] # Only keep sections close to median radius (3 MAD criterion)
                        
                        # 3 things can happen here:
                        # There are no deviated sections --> there is coherence among 3 --> case B
                        # There are 2 deviated sections --> only median radius survives filter --> case C

                        # Case B
                        if filtered_sections.shape[0] == close_to_dbh.shape[0]:
                            
                           dbh_values[i] = R[i, which_dbh] * 2
           
                           tree_locations[i, X_field] = X_c[i, which_dbh].flatten() # Their centers are averaged and we keep that value
                           tree_locations[i, Y_field] = Y_c[i, which_dbh].flatten() # Their centers are averaged and we keep that value
                           tree_locations[i, Z_field] = tree_vector[i, 7] + dbh
                        
                        # Case C
                        else:
                            # if PCA1 Z value is negative
                            if tree_vector[i, 3] < 0:
                            
                                vector = -tree_vector[i, 1:4]
                        
                            else:
                            
                                vector = tree_vector[i, 1:4]
                        
                            dbh_values[i] = 0
                            
                            diff_height = dbh - tree_vector[i, 6] + tree_vector[i, 7]  # Compute the height difference between centroid and BH
                            dist_centroid_dbh = diff_height / np.cos(tree_vector[i, 8] * np.pi / 180)  # Compute the distance between centroid and axis point at BH.      
                            tree_locations[i, :] = vector * dist_centroid_dbh + tree_vector[i, 4:7] # Compute coordinates of axis point at BH. 

               

            # If there is not a single section that either has non 0 radius nor low outlier probability, there is nothing else to do -> axis location is used 
            else: 
            
                if tree_vector[i, 3] < 0:
                
                    vector = -tree_vector[i, 1:4]
            
                else:
                
                    vector = tree_vector[i, 1:4]
            
                diff_height = dbh - tree_vector[i, 6] + tree_vector[i, 7]  # Compute the height difference between centroid and BH
                dist_centroid_dbh = diff_height / np.cos(tree_vector[i, 8] * np.pi / 180)  # Compute the distance between centroid and axis point at BH.      
                tree_locations[i, :] = vector * dist_centroid_dbh + tree_vector[i, 4:7] # Compute coordinates of axis point at BH. 

                dbh_values[i] = 0
    
    las_tree_locations = laspy.create(point_format = 2, file_version = '1.2')
    las_tree_locations.x = tree_locations[:, X_field]
    las_tree_locations.y = tree_locations[:, Y_field]
    las_tree_locations.z = tree_locations[:, Z_field]


    las_tree_locations.write(filename_las[:-4] + "_tree_locator.las")
    
    return(dbh_values, tree_locations)