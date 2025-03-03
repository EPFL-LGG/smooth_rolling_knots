import src.geometry_src.geom as geom
from src.geometry_src.geom_utils import  *

from scipy.optimize import minimize_scalar

from typing import Tuple, List, Callable
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from itertools import product
from scipy.spatial import cKDTree

def rolliness(vertices:np.ndarray, metric:str="maxmin") -> Tuple[float, np.ndarray]:
    """
    Compute the rolliness of a knot.
    Metric:
    - AAD: Average Absolute Deviation 
    - RAR: Range to Average Ratio 

    Returns:
    - rho: rolliness value
    - heights: heights of the center of mass along the rolling trajectory
    """
    hull = ConvexHull(vertices)
    faces = vertices[hull.simplices]
    new_order = reorder_faces(faces, hull.neighbors)
    faces = faces[new_order]
    normals = compute_triange_normals(faces)
    centers = compute_triangle_centers(faces)
    normals = make_normals_exterior(normals, centers)
    cm = np.mean(centers, axis=0)

    # project cm on normal of each triangle
    # normal vectors starting from the center of mass
    cm_vecs = centers - cm
    
    # project cm on the normal
    heights = np.einsum('ij, ij->i', cm_vecs, normals)
    
    if metric == "AAD":
        # compute the average absolute deviation
        mean = np.mean(heights)
        rho = np.mean(np.abs(heights - mean)) / mean
    elif metric == "maxmin":
        # compute the rolliness
        mean = np.mean(heights)
        rho = (np.max(heights) - np.min(heights)) / mean

    return rho, heights

def rolling_trajectory(vertices:np.ndarray) -> np.ndarray:
    """
    Compute the rolling trajectory of a knot
    """
    hull = ConvexHull(vertices)
    faces = vertices[hull.simplices]
    new_order = reorder_faces(faces, hull.neighbors)
    faces = faces[new_order]
    shifted_faces = np.roll(faces, 1, axis=1)
    edge_midpoints = (faces + shifted_faces) / 2
    edge_lengths = np.linalg.norm(faces - shifted_faces, axis=2)
    shortest_edge = np.argmin(edge_lengths, axis=1)
    mask = np.ones(edge_midpoints.shape, dtype=bool)
    mask[np.arange(mask.shape[0]), shortest_edge] = False
    edge_midpoints = edge_midpoints[mask].reshape(-1, 2, 3)
    traj = np.mean(edge_midpoints, axis=1)
    return traj

def get_Morton_knot_exterior_segments(knot:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the exterior segments of a Morton knot
    """
    hull = ConvexHull(knot)
    exterior_indices = hull.vertices
    n = len(exterior_indices)//2
    diff = exterior_indices[1:] - exterior_indices[:-1]
    idx = np.where(diff>1)[0][0]
    idx_seg1 = np.zeros(n,dtype=int)
    idx_seg2 = np.zeros(n,dtype=int)
    idx_seg1[n-idx-1:] = exterior_indices[:idx+1]
    idx_seg2 = exterior_indices[idx+1:idx+n+1]
    idx_seg1[:n-idx-1] = exterior_indices[idx+n+1:]


    return knot[idx_seg1],knot[idx_seg2]

def get_Morton_approximate_disks(knot:np.ndarray, n_disk:int=None) -> Tuple[Tuple[np.ndarray, np.ndarray], float, float, float, float]:
    """
    Get the approximate disks of a Morton knot
    Returns
    - The aligned disks in a Tuple(disk1, disk2) 
    - a,b,c,theta 
        - a: first ellipse radius
        - b: second ellipse radius
        - c: distance between the centers
        - theta: angle between the ellipses (optimal TDR has pi/2)
    """

    if n_disk is None:
        n_disk = knot.shape[0]//2

    sgmts = get_Morton_knot_exterior_segments(knot)
    uniform_sgmts = [make_uniform(sgmt, n_disk) for sgmt in sgmts]
    _, _, centers, radii_h, radii_v, dist, angle = get_all_Morton_disk_info(uniform_sgmts)
    
    # assume symmetry between segments
    rh = radii_h[0] 
    rv = radii_v[0]

    disks = [geom.ellipse(n_disk, rh, rv) for i in range(2)]

    # align the segments
    rotations = [R.align_vectors(sgmt, disk)[0] for sgmt, disk in zip(uniform_sgmts, disks)]
    aligned_disks = [apply_local_rotation_matrix(rot.as_matrix(), disk) for rot, disk in zip(rotations, disks)]

    # center the disks
    disk_centers = [np.mean(disk, axis=0) for disk in aligned_disks]
    translations = [center - disk_center for center, disk_center in zip(centers, disk_centers)]
    aligned_disks = [disk + translation for disk, translation in zip(aligned_disks, translations)]

    return tuple(aligned_disks), rh, rv, dist, angle
    
def get_Morton_disk_angle(knot:np.ndarray=None, n:int=100) -> float:
    """
    Get the angle between the two disks of a Morton knot
    """
    if knot is not None:
        n = knot.shape[0]
    else:
        knot, _ = geom.Morton_knot(a=a, n=10_000)

    sgmts = get_Morton_knot_exterior_segments(knot=knot)
    uniform_sgmts = [make_uniform(sgmt, n) for sgmt in sgmts]
    vertical_radius_info = [get_Morton_segment_vertical_radius_vector(sgmt) for sgmt in uniform_sgmts]
    l1, l2 = vertical_radius_info[0][0], vertical_radius_info[1][0]
    angle = np.arccos(np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2)))

    return angle

def get_segment_perpendicular_vector(uniform_sgmt:np.ndarray, axis:np.ndarray, point:np.ndarray) -> np.ndarray:
    """
    Get the vector that is perpendicular to the given axis, starting from the given point and ending on the segment.
    """
    vec_from_center = uniform_sgmt - point 
    dot_products = np.array([np.abs(np.dot(vec, axis)) for vec in vec_from_center])
    # argsort the dot products to get the two vectors that are the most perpendicular to the axis
    two_closest_vectors_index = np.argsort(dot_products)[:2]
    two_closest_vectors = vec_from_center[two_closest_vectors_index]
    two_closest_vector_dot_products = dot_products[two_closest_vectors_index]
    two_closest_vector_inverted_dot_products = 1/two_closest_vector_dot_products
    invs_dot_product_sum = np.sum(two_closest_vector_inverted_dot_products)
    two_closest_vector_weights = two_closest_vector_inverted_dot_products/invs_dot_product_sum # weight the closest vectors by the inverse of their dot product (closer to zero is bigger weight)
    # radius is weighted mean of the two vectors
    apply_weights = np.einsum("ij, i -> ij", two_closest_vectors, two_closest_vector_weights)
    perp_vector = np.sum(apply_weights, axis=0)
    return perp_vector

def get_Morton_segment_vertical_radius_vector(uniform_sgmt:np.ndarray) ->Tuple[np.ndarray, int]:
    """
    Get the vertical radius vector of a Morton knot exterior segment. The segment vertices must be in order.
    """
    # get the longest vectors along the first ellipse axis (to approximate the first diameter)
    vectors = np.array([uniform_sgmt[-(i+1)] - uniform_sgmt[i] for i in range(int(np.ceil(uniform_sgmt.shape[0]/2)))])
    longest_vectors_index = np.argmax(np.linalg.norm(vectors, axis=1))
    longest_vectors = vectors[longest_vectors_index]
    return longest_vectors, longest_vectors_index

def get_Morton_radius_indices(knot:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the indices of the vertices that define the radii of the Morton knot disks
    """
    xy_projection = knot[:, :2]
    horizontal_radius_indices = maximum_distance(xy_projection)
    
    z_args = np.argsort(knot[:, 2])
    extrema = z_args[[0, 1, -2, -1]]
    vertical_radius_1_indices = extrema[[0, 2]]
    vertical_radius_2_indices = extrema[[1, 3]]

    return horizontal_radius_indices, vertical_radius_1_indices, vertical_radius_2_indices

def get_Morton_disks(knot:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the disks of a Morton knot
    """
    horizontal_radius_indices, vertical_radius_1_indices, vertical_radius_2_indices = get_Morton_radius_indices(knot)
    horizontal_radius = knot[horizontal_radius_indices, :]
    vertical_radius_1 = knot[vertical_radius_1_indices, :]
    vertical_radius_2 = knot[vertical_radius_2_indices, :]

    return horizontal_radius, vertical_radius_1, vertical_radius_2

def get_all_Morton_disk_info(uniform_sgmts:List[np.ndarray]) -> Tuple[List[Tuple[np.ndarray, int]], List[np.ndarray], List[np.ndarray], float, float, float, float]:
    """
    Get all the information for the approximating disks of a Morton knot
    Returns
    - vertical_radius_info: list of tuples (vertical radius vector, index of the longest vector)
    - horizontal_radius_info: list of horizontal radius vectors (perpendicular to the vertical radius vector)
    - centers: list of the centers of the disks
    - radii_h: list of the horizontal radii values
    - radii_v: list of the vertical radii values
    - dist: distance between the centers
    - angle: angle between the vertical radius vectors
    """
    vertical_radius_info = [get_Morton_segment_vertical_radius_vector(sgmt) for sgmt in uniform_sgmts]
    centers = [np.mean([sgmt[info[1]], sgmt[-(info[1]+1)]], axis=0) for sgmt, info in zip(uniform_sgmts, vertical_radius_info)]
    horizontal_radius = [get_segment_perpendicular_vector(sgmt, info[0], center) for sgmt, info, center in zip(uniform_sgmts, vertical_radius_info, centers)]
    radii_h = [np.linalg.norm(rad) for rad in horizontal_radius]
    radii_v = [np.linalg.norm(info[0])/2 for info in vertical_radius_info]
    dist = np.linalg.norm(centers[0] - centers[1])
    
    # angle
    l1, l2 = vertical_radius_info[0][0], vertical_radius_info[1][0]
    angle = np.arccos(np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2)))

    return vertical_radius_info, horizontal_radius, centers, radii_h, radii_v, dist, angle

def TDR_error(a:float=1, b:float=1, c:int=np.sqrt(2)) -> float:
    """
    Compute the error from the optimal TDR family
    """
    return np.abs(c**2 - (4*a**2 - 2*b**2))

def get_optimal_TDR_distance(a:float=1, b:float=1) -> float:
    """
    Get the optimal distance between the disk centers of a TDR to obtain zero error
    """
    return np.sqrt(4*a**2 - 2*b**2)

def optimize_z_stretch(data:np.ndarray, a:float=0.5831, n:int=1000, last_stretch:float=None):
    """
    Optimize the z-stretch for a given knot (NOT USED IN THE FINAL IMPLEMENTATION, USING SCIPY OPTIMIZATION INSTEAD)
    """
    if data is not None:
        knot = data
        n = knot.shape[0]
    else:
        knot, _ = geom.Morton_knot(a=a, n=n)

    max_stretch = 6
    min_error = np.inf
    best_stretch = None
    best_angle = None
    angle_eps = 1e-2
    eps = 1e-10
    if last_stretch is not None:
        delta = max_stretch/n
        start = last_stretch - delta
        n_second_part = int((max_stretch-start)/max_stretch * n)
        stretches = np.linspace(start, max_stretch, n_second_part)
        n_first_part = n - n_second_part
        stretches = np.concatenate([stretches, np.linspace(eps, start, n_first_part)])
    else:
        stretches = np.linspace(eps, max_stretch, n)

    for stretch in stretches:
        new_knot = geom.z_stretch(knot, stretch)
        try:
            angle = get_Morton_disk_angle(new_knot)
            error = np.abs(angle - np.pi/2)
            if error < min_error:
                min_error = error
                best_stretch = stretch
                best_angle = angle
        except QhullError:
            print(f"QhullError for stretch {stretch}")
            continue
        print(f"Stretch: {stretch:4f}\tAngle: {np.degrees(best_angle):4f}", end="\r")
        if min_error < angle_eps:
            break
    
    return best_stretch, best_angle

def optimize_z_stretch_scipy(data:np.ndarray):
    """
    Optimize the z-stretch for a given knot using scipy optimization
    """
    def objective(stretch):
        new_knot = geom.z_stretch(data, stretch)
        angle = get_Morton_disk_angle(new_knot)
        return np.abs(angle - np.pi/2)
    res = minimize_scalar(objective, bounds=(0, 6), options={"xatol":1e-10})
    return res.x, res.success

def optimize_z_stretch_rolliness(data:np.ndarray):
    """
    Optimize the z-stretch for a given knot using scipy optimization
    """
    def objective(stretch):
        new_knot = geom.z_stretch(data, stretch)
        rho, _ = rolliness(new_knot)
        return rho
    res = minimize_scalar(objective, bounds=(0, 6), options={"xatol":1e-10})
    return res.x, res.success

def optimize_z_stretch_angles(knot:Callable, n:int=1000) -> Tuple[float, float]:

    def objective(stretch):
        new_knot = lambda phi: geom.z_stretch(knot(phi), stretch)
        dot = max_radius_dot(new_knot, n)
        return np.abs(dot)
    
    res = minimize_scalar(objective, bounds=(0, 6), options={"xatol":1e-10})
    return res.x, res.success

def optimize_z_stretch_angles_before(knot:Callable, n:int=1000) -> Tuple[float, float]:

    def objective(stretch):
        angles = max_radius_angles(knot, n)
        stretched_knot = lambda phi: geom.z_stretch(knot(phi), stretch)
        projxy = lambda phi: geom.xy_projection(stretched_knot(phi))
        return np.abs(np.dot(projxy(angles[0]), projxy(angles[1])))
    
    res = minimize_scalar(objective, bounds=(0, 6), options={"xatol":1e-10})
    return res.x, res.success

def convex_hull_angles(knot:Callable, n:int=1000) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Determine the angles that define the convex hull. 

    Algorithrm sources from the knowledge that the hull will be defined by points within the z=0 extrema.
    1. Look for first angle of the first segment by sampling points after the first extrema (7pi/4)
    2. Look for second angle of the first segment by sampling points before the third extrema (3pi/4)
    3. First and second angles of second segment are obtained by mirroring with angle+pi

    Output: range of segment 1 (min angle, max angle), range of segment 2 (min angle, max angle)
    """

    def hull_first_angle(phi) -> float:
        hull = ConvexHull(knot(phi))
        return phi[np.min(hull.vertices)]

    # We know the zero preceding the first segment occurs at 7pi/4    
    phi = np.linspace(7*np.pi/4 - 2*np.pi, 7*np.pi/4, n, endpoint=False)
    angle_1 = hull_first_angle(phi)

    # And the zero following the first segment occurs at 3pi/4
    phi = np.linspace(3*np.pi/4, 3*np.pi/4 + 2*np.pi, n, endpoint=False)
    phi = np.flip(phi)
    angle_2 = hull_first_angle(phi)

    # Points are mirrored to the second segment by adding pi
    return (angle_1, angle_2), (angle_1+np.pi, angle_2+np.pi)

def convex_hull_angle(knot:Callable, n:int=1000) -> float:
    """
    Determine the angles that first defines the convex hull. 

    Algorithm sources from the knowledge that the hull will be defined by points within the z=0 extrema. Look for first angle of the first segment by sampling points after the first extrema (7pi/4)

    Output: min angle
    """

    def hull_first_angle(phi) -> float:
        hull = ConvexHull(knot(phi))
        return phi[np.min(hull.vertices)]

    # We know the zero preceding the first segment occurs at 7pi/4    
    phi = np.linspace(7*np.pi/4 - 2*np.pi, 7*np.pi/4, n, endpoint=False)
    angle_1 = hull_first_angle(phi)

    return angle_1

def align_tdr(knot:np.ndarray) -> np.ndarray:
    
    rot_tdr = aligned_tdr_disks(knot)

    return np.concatenate(rot_tdr, axis=0)

def aligned_tdr_disks(knot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a TDR to a knot
    """
    
    sgmts = get_Morton_knot_exterior_segments(knot)
    
    uniform_sgmts = [make_uniform(sgmt, knot.shape[0]//2) for sgmt in sgmts]
    vinfo, hinfo, _, rh, rv, _, _ = get_all_Morton_disk_info(uniform_sgmts=uniform_sgmts)

    rh = rh[0]
    rv = rv[0]

    n = knot.shape[0]//2
    tdr_disks = geom.TDR_disks(n, rh, rv)

    # align the tdr's z axis to the knot's left vertical axis
    vertical_axis = vinfo[1][0]
    horizontal_axis = hinfo[0]
    rot_x = R.align_vectors(horizontal_axis, np.array([1,0,0]))[0]
    rot_z = R.align_vectors(vertical_axis, np.array([0,0,1]))[0]
    disks = np.concatenate(tdr_disks)
    rot_tdr = apply_local_rotation_matrix(rot_x.as_matrix(), disks)
    rot_tdr = apply_local_rotation_matrix(rot_z.as_matrix(), rot_tdr)
    
    disks = np.split(rot_tdr, 2, axis=0)

    return disks

def project_segments_to_tdr(exterior_segments:List[np.ndarray], interior_segments:List[np.ndarray], tdr_disks:List[np.ndarray], use_rbf:bool=False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Project the exterior segments of a knot onto the TDR
    Apply a decayed translation to the interior segments
    
    use_rbf: 
        whether to use radial basis functions to project the segments; 
        if False, the default projection method with decayed translation is used 
    """
    disk_trees = [cKDTree(disk) for disk in tdr_disks]

    def map_to_closest_points(data, tree):
        return tree.query(data)[1]

    exterior_translations = [tree.data[map_to_closest_points(segment, tree)] - segment for segment, tree in product(exterior_segments, disk_trees)]
    normed_exterior_translations = [np.sum(np.linalg.norm(translations, axis=1)) for translations in exterior_translations]
    normed_exterior_translations = np.array(normed_exterior_translations).reshape(2, 2) # reshape to match segments
    exterior_translations_disk_idx = np.argmin(normed_exterior_translations, axis=1) 
    exterior_translations = np.array(exterior_translations).reshape(2, 2, -1, 3)

    translated_exterior_segments = [exterior_segment + exterior_translations[i, ext_idx, :, :] for exterior_segment, ext_idx, i in zip(exterior_segments, exterior_translations_disk_idx, [0, 1])]

    # map each interior segment to each disk separately (list of 2 with shape: (2, n, 3)
    interior_tdr_disk_maps = [np.array([
                tree.data[map_to_closest_points(int_sgmt, tree)]
            for tree in disk_trees]) 
        for int_sgmt in interior_segments
    ]

    # Morph the interior segments
    if use_rbf:
        from scipy.interpolate import RBFInterpolator
        exterior_segments_all = np.concatenate(exterior_segments)
        translated_exterior_segments_all = np.concatenate(translated_exterior_segments)

        # Define the interpolator based on the original and morphed exterior segments
        rbf = RBFInterpolator(exterior_segments_all, translated_exterior_segments_all, kernel='cubic', epsilon=1, smoothing=0)  # smoothing = 0 for exact interpolation (see scipy docs)
        # Apply the interpolator to the interior segments
        translated_interior_segments = [rbf(int_sgmt) for int_sgmt in interior_segments]
    else:
        # tuple of 2 with shape (2, n, 3)
        interior_disk_translations = [int_tdr_disk_map - int_sgmt for int_tdr_disk_map, int_sgmt in zip(interior_tdr_disk_maps, interior_segments)]
        # endpoint translations is list of 4 with shape (2, 1)
        size_of_endpoint_translations = [np.linalg.norm(int_disk_translation[:,i,:], axis=1) for int_disk_translation, i in product(interior_disk_translations, [0, -1])]
        endpoint_translations_disk_idx = [np.argmin(endpoint_translation) for endpoint_translation in size_of_endpoint_translations]
        # reshape to match segments (4) -> (2 segments, 2 endpoints)
        endpoint_translations_disk_idx = [endpoint_translations_disk_idx[:2], endpoint_translations_disk_idx[2:]]
        # gives list of 2 with shape (2, n, 3) (2 for each endpoint, n for each point, 3 for each coordinate)
        interior_disk_translations = [int_disk_translation[idx] for int_disk_translation, idx in zip(interior_disk_translations, endpoint_translations_disk_idx)]
        
        # apply decayed translation
        translated_interior_segments = [
            # apply decay for first endpoint, then reverse decay for second endpoint
            apply_decaying_translation(
                apply_decaying_translation(
                    int_sgmt, 
                    int_translation[0], reverse = False, type="sigmoid"
                ), 
                int_translation[1], reverse=True, type="sigmoid"
            )
            for int_sgmt, int_translation in zip(interior_segments, interior_disk_translations)
        ]

    return translated_exterior_segments, translated_interior_segments

def average_radius(knot: Callable, n:int=1000) -> Tuple[np.ndarray, np.ndarray]:

    hull_angles_1, hull_angles_2 = convex_hull_angles(knot, n)

    xy_knot = lambda phi: geom.xy_projection(knot(phi))

    range_1 = (hull_angles_1[0], np.pi/4) # -2pi to have left < right for linspace
    angles_1 = np.linspace(*range_1, n)
    avg_radius_1 = np.mean(xy_knot(angles_1), axis=0)

    range_2 = (hull_angles_2[0], 5*np.pi/4)
    angles_2 = np.linspace(*range_2, n)
    avg_radius_2 = np.mean(xy_knot(angles_2), axis=0)

    return avg_radius_1, avg_radius_2

def average_dot(knot: Callable, n:int=1000, weighted:bool=False) -> float:
    
    hull_angles_1, hull_angles_2 = convex_hull_angles(knot, n)

    xy_knot = lambda phi: geom.xy_projection(knot(phi))

    range_1 = (hull_angles_1[0], np.pi/4) # -2pi to have left < right for linspace
    angles_1 = np.linspace(*range_1, n)
    vecs_1 = xy_knot(angles_1)

    range_2 = (hull_angles_2[0], 5*np.pi/4)
    angles_2 = np.linspace(*range_2, n)
    vecs_2 = xy_knot(angles_2)

    weights = np.linalg.norm(vecs_1, axis=1) if weighted else None

    return np.average(np.einsum("ij, ij->i", vecs_1, vecs_2), weights=weights)

def max_radius_dot(knot: Callable, n:int=1000) -> float:

    angles = max_radius_angles(knot, n)
    projxy = lambda phi: geom.xy_projection(knot(phi))
    return np.dot(projxy(angles[0]), projxy(angles[1]))

def max_radius_angles(knot: Callable, n:int=1000) -> Tuple[float, float]:
    """
    Get the angles at which the maximal length radius vectors occur. 
    - knot 
    - n: number of points to sample on a single segment. 
    
    Description: Since we know the knot crosses the x-y plane at 7pi/4, pi/4, 3pi/4, and 5pi/4, we can sample n points within the ranges of these angles to find the maximal length radius vectors. 
    """

    xy_knot = lambda phi: geom.xy_projection(knot(phi))
    range_1 = (-np.pi/4, np.pi/4) # -2pi to have left < right for linspace
    angles_1 = np.linspace(*range_1, n)
    idx_1 = np.argmax(np.linalg.norm(xy_knot(angles_1), axis=1))
    angle_1 = angles_1[idx_1]
    
    return (angle_1, angle_1+np.pi)

def max_radius_angle_scipy(knot: Callable) -> float:
    
    norm_xy_knot = lambda phi: np.linalg.norm(geom.xy_projection(knot(phi)))

    def objective(phi):
        return -norm_xy_knot(phi)
    
    res = minimize_scalar(objective, bounds=(-np.pi/4, np.pi/4), options={"xatol":1e-20})
    return res.x

def orthogonal_angle_scipy(a: float, z_stretch: float) -> float:

    def objective(phi):
        return np.abs(a*(np.cos(3*phi)+np.sin(3*phi))-np.sqrt(2)*z_stretch*np.sqrt(1-a**2)*np.cos(2*phi))
    
    res = minimize_scalar(objective, bounds=(-np.pi/4, np.pi/4), options={"xatol":1e-20})
    return res.x

def get_yz_angles(knot: Callable, n:int=1000) -> Tuple[float, float]:
    phi = np.linspace(-np.pi/12, np.pi/4, n, endpoint=False)
    _, y, z = knot(phi).T
    return np.arctan2(z,y)

# def orthogonal_angle_scipy(a: float, z_stretch: float) -> float:

#     def objective(phi):
#         return 0.5*np.sqrt(2)*a*(np.cos(3*phi)+np.sin(3*phi))-z_stretch*np.sqrt(1-a**2)*np.cos(2*phi)
    
#     res = minimize_scalar(objective, bounds=(-np.pi/4, np.pi/4), options={"xatol":1e-20})
#     return res.x

def project_to_tdr(knot:Callable, n:int=100_000, use_rbf:bool=False) -> np.ndarray:
    """
    Project a knot onto its approximate TDR using the parametric representation of the knot
    Note: parametric instead of the vertices to be able to query the ordering, and preserve the order
    """

    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    knot_vertices = knot(phi)

    tdr_disks = aligned_tdr_disks(knot_vertices)
    knot_hull = ConvexHull(knot_vertices)
    exterior = knot_vertices[knot_hull.vertices]

    interior_indices = get_interior_indices(knot_hull)
    interior = knot_vertices[interior_indices]

    interior_segments = split_in_two_segments(interior)
    exterior_segments = split_in_two_segments(exterior)

    translated_exterior_segments, translated_interior_segments = project_segments_to_tdr(exterior_segments, interior_segments, tdr_disks, use_rbf)

    class AngleRange:
        """
        Class to represent an angle range
        Used to sort the segments based on the angles they correspond to, so that the order of the segments is preserved
        """
        def __init__(self, start, end):
            self.start = start
            self.end = end
        
        def __lt__(self, other):
            
            if self.start == other.start and self.end == other.end:
                return False
            
            return self.end < other.start
        
        def __repr__(self):
            return f"({self.start}, {self.end})"
            
    # ORDER THE SEGMENTS USING THE ANGLES THEY CORREPOND TO 

    exterior_angles = phi[knot_hull.vertices]
    interior_angles = phi[interior_indices]
    dphi = np.mean(np.diff(phi)) * 10
    tol = dphi

    exterior_angle_segments = split_in_segments(exterior_angles, tol)
    interior_angle_segments = split_in_segments(interior_angles, tol)

    def merge_segments_if_wrap_around_circle(segments):
        first_angle = segments[0][0]
        last_angle = segments[-1][-1]
        if first_angle==phi[0] and last_angle==phi[-1]:
            segments[0] = np.concatenate([segments[-1], segments[0]])
            return segments[:-1]
        return segments
    
    exterior_angle_segments = merge_segments_if_wrap_around_circle(exterior_angle_segments)
    interior_angle_segments = merge_segments_if_wrap_around_circle(interior_angle_segments)

    interior_angle_ranges = [(segment[0], segment[-1]) for segment in interior_angle_segments]
    exterior_angle_ranges = [(segment[0], segment[-1]) for segment in exterior_angle_segments]

    angle_ranges = list(interior_angle_ranges)
    angle_ranges.extend(exterior_angle_ranges)
    angle_ranges = [AngleRange(*angle_range) for angle_range in angle_ranges]
    angle_ranges_sorted_idx = np.argsort(angle_ranges)

    translated_sgmts = list(translated_interior_segments)
    translated_sgmts.extend(translated_exterior_segments)
    translated_sgmts = [translated_sgmts[idx] for idx in angle_ranges_sorted_idx]

    new_knot = np.concatenate(translated_sgmts)

    return new_knot

def projected_knot(a: float, p: int = 3, q: int = 2, n: int = 100_000, use_rbf: bool = False) -> np.ndarray:
    """
    Project a Morton knot onto its approximate TDR
    """
    knot, _ = geom.Morton_knot(a=a, n=n, p=p)
    z_stretch, _ = optimize_z_stretch_scipy(knot)
    knot = lambda phi: geom.z_stretch(geom.Morton_knot_parametric(phi, a=a, p=p, q=q), z_stretch)
    return project_to_tdr(knot, n, use_rbf)

# ===========================================UNUSED CODE======================================================


def transform_for_optimal_disk_distance(a:float=0.5831, n:int=1000, optimal_z_stretch:float=1.0, knot:np.ndarray=None):
    """
    DID NOT WORK – PROJECTED TO CLOSEST ZERO ROLLINESS TDR INSTEAD
    Transform a knot so that the distance between the disks is optimal 
    """

    if knot is None:
        knot, _ = geom.Morton_knot(a=a, n=n)
        knot = geom.z_stretch(knot, optimal_z_stretch)

    hull = ConvexHull(knot)
    exterior = knot[hull.vertices]
    interior = get_interior_vertices(knot, hull)    
    sgmts = split_in_two_segments(exterior)
    uniform_sgmts = [make_uniform(sgmt, n) for sgmt in sgmts]

    vertical_radius_info, horizontal_radius, centers, radii_h, radii_v, dist, angle = get_all_Morton_disk_info(uniform_sgmts=uniform_sgmts)
    optimal_dist = get_optimal_TDR_distance(radii_h[0], radii_v[0])
    # translation_axis = (centers[1] - centers[0]) / dist
    translation_axis = np.array(horizontal_radius) + np.array(centers)
    translation_axis = translation_axis[1] - translation_axis[0]
    translation_axis = translation_axis / np.linalg.norm(translation_axis)

    translation_amount = optimal_dist - dist
    translation = translation_axis * translation_amount
    sgmt_translations = [-translation/2, translation/2] # since axis direction is from sgmt0 to sgmt1
    translated_exterior = [sgmt + translation for sgmt, translation in zip(sgmts, sgmt_translations)]

    interior_sgmts = split_in_two_segments(interior)
    # reverse flags for applying the decayed translation. This logic is inverted for the subsequent ordering of the segments (where i want last to match first of next segment)
    # match the first exterior sgmt point to the first interior sgmt point, if not reverse the translation decay.
    int_0_endpoint_0 = interior_sgmts[0][0]
    int_0_endpoint_0_idx = np.argmin([
        np.min([np.linalg.norm(int_0_endpoint_0 - sgmts[0][0]),
                np.linalg.norm(int_0_endpoint_0 - sgmts[0][-1])]),
        np.min([np.linalg.norm(int_0_endpoint_0 - sgmts[1][0]),
                np.linalg.norm(int_0_endpoint_0 - sgmts[1][-1])])
    ])
    int_1_endpoint_0 = interior_sgmts[1][0]
    int_1_endpoint_0_idx = np.argmin([
        np.min([np.linalg.norm(int_1_endpoint_0 - sgmts[0][0]),
                np.linalg.norm(int_1_endpoint_0 - sgmts[0][-1])]),
        np.min([np.linalg.norm(int_1_endpoint_0 - sgmts[1][0]),
                np.linalg.norm(int_1_endpoint_0 - sgmts[1][-1])])
    ])
    int_endpoint_0 = [int_0_endpoint_0_idx, int_1_endpoint_0_idx]
    scaled_interior = [apply_decaying_translation(int_sgmt, sgmt_translations[0], idx!=0) for int_sgmt, idx in zip(interior_sgmts, int_endpoint_0)]
    scaled_interior = [apply_decaying_translation(int_sgmt, sgmt_translations[1], idx!=1) for int_sgmt, idx in zip(scaled_interior, int_endpoint_0)]

    new_interior = scaled_interior
    new_exterior = translated_exterior

    # # project the exterior segments onto the TDR circles
    # disk0_r = horizontal_radius[0], vertical_radius_info[0][0]
    # disk1_r = horizontal_radius[1], vertical_radius_info[1][0]
    # normal0 = np.cross(disk0_r[0], disk0_r[1])
    # normal1 = np.cross(disk1_r[0], disk1_r[1])

    # proj_segment0 = project_to_plane(translated_exterior[0], normal0)
    # proj_segment1 = project_to_plane(translated_exterior[1], normal1)

    # # perform decayed projection
    # projected_interior = [decayed_projection_to_plane(scaled_interior[idx], normal0, idx!=0) for idx in range(2)]
    # projected_interior = [decayed_projection_to_plane(scaled_interior[idx], normal1, idx!=1) for idx in range(2)]

    # new_interior = projected_interior
    # new_exterior = [proj_segment0, proj_segment1]
    # i want to preserve the ordering of the segments
    new_interior[0] = new_interior[0][::-1, :] if int_0_endpoint_0_idx == 0 else new_interior[0] # for correct ordering, i want last exterior sgmt point to match the first interior sgmt point
    new_interior[1] = new_interior[1][::-1, :] if int_1_endpoint_0_idx == 1 else new_interior[1] # for correct ordering, i want first exterior sgmt point to match the last interior sgmt point
    new_knot = np.concatenate([new_interior[0], new_exterior[0], new_interior[1], new_exterior[1]])

    return new_knot


def project_to_tdr_old(knot:np.ndarray) -> np.ndarray :
    """
    Firt try at projecting a knot onto its approximate TDR (ordering of the knot is not preserved)
    """

    hull = ConvexHull(knot)
    exterior = knot[hull.vertices]
    exterior_segments = split_in_two_segments(exterior)
    interior_indices = get_interior_indices(hull)
    interior = knot[interior_indices]
    interior_segments = split_in_two_segments(interior)
    tdr_disks = aligned_tdr_disks(knot)
    
    # tdr_hull = ConvexHull(np.concatenate(tdr_disks))
    # tdr_exterior_disk_1_indices = tdr_hull.vertices[np.where(tdr_hull.vertices < tdr_disks[0].shape[0])[0]]
    # tdr_exterior_disk_2_indices = tdr_hull.vertices[np.where(tdr_hull.vertices >= tdr_disks[0].shape[0])[0]] - tdr_disks[0].shape[0]
    # tdr_exterior_disk_1 = tdr_disks[0][tdr_exterior_disk_1_indices]
    # tdr_exterior_disk_2 = tdr_disks[1][tdr_exterior_disk_2_indices]
    # tdr_disks = [tdr_exterior_disk_1, tdr_exterior_disk_2]
    
    translated_exterior_segments, translated_interior_segments = project_segments_to_tdr(exterior_segments, interior_segments, tdr_disks)

    translated_exterior = np.concatenate(translated_exterior_segments)
    translated_interior = np.concatenate(translated_interior_segments)
    new_knot = np.concatenate([translated_interior, translated_exterior])
    return new_knot
