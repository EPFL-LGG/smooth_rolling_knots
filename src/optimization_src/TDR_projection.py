import numpy as np
from src.geometry_src import geom_utils
from scipy.spatial import ConvexHull, distance as dst
from src.geometry_src.geom_utils import compute_aligned_tdr_disks
from src.geometry_src import distances

def project_point_to_polyline(point, polyline):
    """
    Projects a 3D point to the closest point on a 3D polyline.

    Parameters:
    point (numpy.ndarray): A 3D point represented as a numpy array of shape (3,).
    polyline (numpy.ndarray): A polyline represented as a numpy array of shape (n, 3).

    Returns:
    numpy.ndarray: The closest point on the polyline to the input point.
    """
    # Calculate the distances from the point to each point in the polyline
    distances = dst.cdist([point.numpy()], polyline, 'euclidean')
    
    # Find the index of the closest point in the polyline
    closest_index = distances.argmin()
    
    # Return the closest point
    return polyline[closest_index]

def get_exterior_interior_indices(knot):
    # Get knot external vertices
    hull = ConvexHull(knot)
    exterior_indices = hull.vertices
    interior_indices = geom_utils.get_interior_indices(hull)
    return exterior_indices, interior_indices

def split_index_list_in_two_segments(indices):
    'Compute the index lists of contiguous indices (segments) from the input list of indices. Note: expects two resulting segments.'
    n = len(indices)//2
    diff = indices[1:] - indices[:-1]
    indices_diff_geq_two = np.where(diff>1)[0]
    assert indices_diff_geq_two.size == 1, f"Expected two segment of contiguous indices, i.e. a single jump > 1, but got {indices_diff_geq_two.size} jumps."
    idx = indices_diff_geq_two[0]
    idx_seg1 = np.zeros(n,dtype=int)
    idx_seg2 = np.zeros(n,dtype=int)
    idx_seg1[n-idx-1:] = indices[:idx+1]
    idx_seg2 = indices[idx+1:idx+n+1]
    idx_seg1[:n-idx-1] = indices[idx+n+1:]
    return idx_seg1, idx_seg2
    

def project_knot_to_disks(knot,n_tdr,a_opt,b_opt, rotated=False):

    exterior_indices, interior_indices = get_exterior_interior_indices(knot)
    idx_seg1, idx_seg2 = split_index_list_in_two_segments(exterior_indices)
    exterior_vertices1 = knot[idx_seg1]
    exterior_vertices2 = knot[idx_seg2]
    n_ext_2 = idx_seg1.shape[0]

    # Get TDR ellispes
    c_opt = np.sqrt(4*a_opt**2 - 2*b_opt**2)
    tdr_ell1, tdr_ell2 = compute_aligned_tdr_disks(n_tdr, a_opt, b_opt, c_opt, rotated=rotated)

    # Project all knot's exterior vertices to the disks
    proj_hull_vertices1 = np.zeros_like(exterior_vertices1)
    proj_hull_vertices2 = np.zeros_like(exterior_vertices2)
    d1 = distances.points_to_polyline_distance(exterior_vertices1,tdr_ell1)
    d2 = distances.points_to_polyline_distance(exterior_vertices1,tdr_ell2)
    if d1 < d2:
        disk1, disk2 = tdr_ell1, tdr_ell2
    else:
        disk1, disk2 = tdr_ell2, tdr_ell1
    for i in range(n_ext_2):
        proj_hull_vertices1[i] = project_point_to_polyline(exterior_vertices1[i], disk1)
        proj_hull_vertices2[i] = project_point_to_polyline(exterior_vertices2[i], disk2)

    return proj_hull_vertices1, proj_hull_vertices2