import numpy as np
from scipy.spatial import ConvexHull
from typing import Tuple
from scipy.interpolate import interp1d
from src.geometry_src.geom_utils_torch import rotate_about_axis
import torch

TORCH_DTYPE = torch.float64 

def apply_local_rotation_matrix(matrix:np.ndarray, vectors:np.ndarray) -> np.ndarray:
    """
    Apply a rotation matrix to a set of vectors around its center. 
    Ex: rotate a triangle around its center.
    """
    center = np.mean(vectors, axis=0)
    centered_vectors = vectors - center
    rotated_vectors = matrix @ centered_vectors.T
    rotated_vectors = rotated_vectors.T
    rotated_vectors = rotated_vectors + center
    return rotated_vectors

def get_interior_indices(hull:ConvexHull) -> np.ndarray:
    """
    Get the indices of the vertices that are not part of the convex hull.
    """
    n_vertices = hull.points.shape[0]
    interior_bool = np.ones(n_vertices, dtype=bool)
    interior_bool[hull.vertices] = False
    return np.where(interior_bool)[0]

def get_interior_vertices(vertices:np.ndarray, hull:ConvexHull) -> np.ndarray:
    """
    Get the vertices of the knot that are not part of the convex hull.
    """
    return vertices[get_interior_indices(hull)]

def compute_center_of_mass(vertices:np.ndarray) -> np.ndarray:
    """
    Compute the center of mass of a set of vertices.
    """
    return np.mean(vertices, axis=0)


def compute_triange_normals(triangles:np.ndarray) -> np.ndarray:
    """
    Compute the normal vector of 3D triangles. The result is normalized.
    """

    # compute the edges
    edges = np.diff(triangles, axis=1)
    # compute the normal
    if edges.shape[1] == 2:
        normals = np.cross(edges[:, 0, :], edges[:, 1, :], axis=1)
    elif edges.shape[1] == 1: # if the triangles are actually lines
        # rotate the edge by 90 degrees
        normals = np.zeros((edges.shape[0], 2))
        normals[:, 0] = -edges[:, 0, 1]
        normals[:, 1] = edges[:, 0, 0]
        
    # normalize the normal
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    return normals

def compute_triangle_centers(triangles:np.ndarray) -> np.ndarray:
    """
    Compute the center of mass of each 3D triangle in an array.
    """
    return np.mean(triangles, axis=1)

def compute_triangle_areas(triangles:np.ndarray) -> np.ndarray:
    """
    Compute the area of 3D triangles.
    """
    # compute the edges
    edges = np.diff(triangles, axis=1)
    # compute the paralellogram area
    normals = np.cross(edges[:, 0, :], edges[:, 1, :], axis=1)
    # compute the area
    areas = 0.5 * np.linalg.norm(normals, axis=1)

    return areas

def make_normals_exterior(normals:np.ndarray, centers:np.ndarray) -> np.ndarray:
    """
    Make the normals point outwards.
    """
    cm = np.mean(centers, axis=0)

    # compute the vector from the center of mass to the center of the triangle
    interiors = cm - centers

    # compute the dot product between the normal and the vector
    dot = np.einsum('ij,ij->i', normals, interiors)

    # flip the normals if the dot product is positive (meaning the normal points inwards)
    normals[dot > 0] = -normals[dot > 0]

    return normals

def reorder_faces(faces:np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """
    Reorder the faces of the convex hull such that the normals are aligned and produce a "path" of faces.
    Returns the new order of the faces, as an index list where `new_order[i]=j` indicates that the 'j'-th face should be placed at new position `i`.
    """

    normals = compute_triange_normals(faces)
    normals = make_normals_exterior(normals, compute_triangle_centers(faces))

    n_faces = faces.shape[0]
    new_order = np.zeros(n_faces, dtype=np.int32) # i:j -> new index i is mapped to old index j, -1 for not mapped
    chosen = np.zeros(n_faces, dtype=bool) # whether a face has been chosen

    current_face_index = 0 # start with the first face

    # start from 1 because the first face is already chosen in the new order
    for i in range(1, n_faces):
        
        chosen[current_face_index] = True

        neighbors_index = neighbors[current_face_index]

        # only consider the neighbors that have not been chosen
        neighbors_index = neighbors_index[np.logical_not(chosen[neighbors_index])]
        
        if len(neighbors_index) == 0:
            print(f"Weird hull, reordering failed. Reordered {i+1}/{n_faces} faces.")
            break
        
        normal = normals[current_face_index]

        neighbor_normals = normals[neighbors_index]


        # compute the closest neighbor: the one with the most similar normal
        closeness_metric = np.dot(neighbor_normals, normal)
        
        pick = np.argmax(closeness_metric)
        current_face_index = neighbors_index[pick]
        new_order[i] = current_face_index

    return new_order

def make_uniform(segment:np.ndarray, n:int) -> np.ndarray:
    """
    Make a curve segment uniform.
    """
    t = np.linspace(0, 1, segment.shape[0])
    
    # Interpolate x, y, and z separately
    cs_x = interp1d(t, segment[:, 0])
    cs_y = interp1d(t, segment[:, 1])
    cs_z = interp1d(t, segment[:, 2])
    
    t_new = np.linspace(0, 1, n)
    
    # Combine the results into a single array
    return np.column_stack((cs_x(t_new), cs_y(t_new), cs_z(t_new)))

def split_in_two_segments(vertices:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the exterior of the knot into two segments.
    """

    diff = np.diff(vertices, axis=0)
    abs_diff = np.linalg.norm(diff, axis=1)
    tol = np.max(abs_diff) * 0.5 # TODO: find a better way to determine the segment splitting condition
    segments = split_in_segments(vertices, tol)
    assert len(segments) == 2
    return segments[0], segments[1]

def scale_along_axis(vertices:np.ndarray, axis:np.ndarray, factor:float) -> np.ndarray:
    """
    Scale the vertices along an axis. 
    Logic: project the vertices to the axis, scale the projected vertices, and combine with the perpendicular projection.
    So $$ v_{scaled} = scale \\times v_{proj} + v_{perp} $$
    """
    assert np.linalg.norm(axis) == 1.0 # axis should be normalized
    # project to axis
    dot = np.dot(vertices, axis)
    proj = np.outer(dot, axis) 
    # compute the difference
    perp_proj = vertices - proj
    # scale the projected vertices
    proj = proj * factor
    # combine the two
    return proj + perp_proj

def apply_decaying_translation(vertices:np.ndarray, translation:np.ndarray, reverse:bool=False, type:str='linear') -> np.ndarray:
    """
    Apply a transformation to the vertices, where the translation decays with the vertex index.
    """
    n = vertices.shape[0]
    if type == 'linear':
        decay = np.linspace(1, 0, n)
    elif type == 'ReLU':
        decay = np.maximum(np.linspace(1, -1, n), 0)
    elif type == 'sigmoid':
        # alpha = -20
        alpha = -10
        beta = -0.5
        # beta = 0
        decay = 1 / (1 + np.exp(-alpha * (np.linspace(-1, 1, n) - beta)))
    elif type == 'abs':
        decay = np.abs(np.linspace(-1, 1, n))
    elif type == 'parabola':
        decay = np.linspace(-1, 1, n)**2
    elif type == 'vector':
        decay = np.linalg.norm(vertices, axis=1)
        decay = decay / np.max(decay)
    elif type == 'sqrt':
        decay = np.sqrt(np.linspace(1, 0, n//2))
        decay = np.concatenate([decay, np.zeros(n-len(decay))])
    elif type == 'cos':
        decay = 0.5*np.cos(np.linspace(0, np.pi, n//2)) + 0.5
        decay = np.concatenate([decay, np.zeros(n-len(decay))])
    elif type == 'exp':
        raise NotImplementedError

    if reverse:
        decay = decay[::-1]
    if translation.ndim == 1:
        decayed_translation = np.outer(decay, translation)
    else:
        decayed_translation = np.einsum('i,ij->ij', decay, translation)
    return vertices + decayed_translation

def project_to_plane(vertices:np.ndarray, normal:np.ndarray) -> np.ndarray:
    """
    Project the vertices to a plane defined by a normal.
    """
    normal = normal / np.linalg.norm(normal)
    return vertices - np.outer(np.dot(vertices, normal), normal)

def map_to_closest_points(data:np.ndarray, target:np.ndarray) -> np.ndarray:
    """
    Project the vertices to a curve.
    """
    # project the vertices onto the closest point on the curve
    from scipy.spatial import cKDTree
    tree = cKDTree(target)
    _, indices = tree.query(data)
    return indices

def split_in_segments(elements:np.ndarray, tol:float) -> np.ndarray:
    """
    Split the elements into segments.
    """
    if elements.ndim == 1:
        elements = elements[:, None]
    # find the indices of the elements that are far apart
    diff = np.diff(elements, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    split_indices = np.where(dist > tol)[0] + 1
    # split the elements
    segments = np.split(elements, split_indices)
    # merge first and last segments if they are close
    if len(segments) > 1:
        dist = np.linalg.norm(segments[0][0] - segments[-1][-1])
        if dist < tol:
            segments[0] = np.concatenate((segments[-1], segments[0]))
            segments = segments[:-1]
    return segments

def decayed_projection_to_plane(vertices:np.ndarray, normal:np.ndarray, reverse:bool=False, exponential:bool = True) -> np.ndarray:
    """
    Project the vertices to a plane defined by a normal, where the projection decays with the vertex index.
    """
    normal = normal / np.linalg.norm(normal)
    n = vertices.shape[0]
    decay = np.linspace(1, 0, n) if not reverse else np.linspace(0, 1, n)
    if exponential:
        decay = np.linspace(start=0, stop=0.5, num=n//2, endpoint=False)
        if reverse:
            decay = decay[::-1]
        decay = np.exp(-(1/(1-2*decay)+1))
        decay = np.concatenate((decay, np.zeros(n-len(decay)))) if not reverse else np.concatenate((np.zeros(n-len(decay)), decay))
    decayed_normal = np.outer(decay, normal)
    return vertices - np.outer(vertices @ decayed_normal.T, decayed_normal)

def maximum_distance(vertices:np.ndarray) -> np.ndarray:
    """
    Compute the longest vector between any two points in a set of vertices.
    """
    hull = ConvexHull(vertices)
    max_dist = 0
    max_indices = (None, None)
    for i in hull.vertices:
        for j in hull.vertices:
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist > max_dist:
                max_dist = dist
                max_indices = (i, j)

    return max_indices

def project_points_to_plane(points, plane_point, plane_normal):
    # Normalize the plane normal vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Compute the dot product of each point with the plane normal
    dot_product = np.dot(points - plane_point, plane_normal)

    # Subtract the dot product times the plane normal from each point
    projected_points = points - np.outer(dot_product, plane_normal)

    return projected_points


# DEVELOP CONVEX HULL (FAIL) ========================================================================================================================

def develop_surface(hull:ConvexHull) -> np.ndarray:
    """
    This code is an attempt to reproduce some results, in which we plot the developped convex hull of the knot on the plane. 
    It does not work properly, and I decided to focus on other aspects of the project.
    The logic I tried to implement is as follows:
    - Reorder the faces of the convex hull such that they represent the path along which the surface should be developed
    - Align the faces such that their normal is aligned with the z-axis
    - Place the first face on the plane, such that the longest edge is centered at the origin
    - For each subsequent face, find the common edge with the previous face, and match it to the edge of the previous face so as to "stack" the faces on top of each other.
    """

    faces = hull.points[hull.simplices]
    neighbors = hull.neighbors

    new_order = reorder_faces(faces, neighbors)
    reordered_faces = faces[new_order]
    faces_index = hull.simplices[new_order]
    aligned_faces = align_faces(reordered_faces)

    new_faces = np.zeros((faces.shape[0], 3, 2))

    previous_face_index = faces_index[1] # first previous is actually next face, to get the shared vertices
    direction = np.array([0, 1])
    for i in range(faces.shape[0]):
        # find the common vertices with the neighboring face
        current_face_index = faces_index[i] # vertex indices of the current face
        common_vertices_index = np.intersect1d(current_face_index, previous_face_index)
        assert len(common_vertices_index) == 2
        common_vertices = np.where(np.isin(current_face_index, common_vertices_index))[0] # convert to indices in face, as opposed to indices in vertices
        common_vertices_in_previous = np.where(np.isin(previous_face_index, common_vertices_index))[0]
        current_face = aligned_faces[i]

        if i == 0:
            # PLACE FIRST FACE 
            other_vertex = np.where(np.logical_not(np.isin(current_face_index, common_vertices_index)))[0][0] # find the vertex that is not in the common vertices
            # find the longest edge from that vertex
            edges = current_face[other_vertex] - current_face[common_vertices]
            edge_lengths = np.linalg.norm(edges, axis=1)
            longest_edge_index = np.argmax(edge_lengths)
            # center that edge at the origin
            longest_edge_length = edge_lengths[longest_edge_index]
            v1 = np.array((-0.5*longest_edge_length, 0))
            v2 = np.array((0.5*longest_edge_length, 0))
            target_edge = np.array([v1, v2])
            edge_index = np.array([other_vertex, common_vertices[longest_edge_index]])

        else:
            # target edge is the edge between the common vertices
            target_edge = new_faces[i-1][common_vertices_in_previous]
            center_target_edge = np.mean(target_edge, axis=0)
            other_vertex_in_previous = np.where(np.logical_not(np.isin(previous_face_index, common_vertices_index)))[0][0] # find the vertex that is not in the common vertices
            other_v_direction = center_target_edge - new_faces[i-1][other_vertex_in_previous]
            other_v_direction = other_v_direction / np.linalg.norm(other_v_direction)
            # i want the direction to be perpendicular to the target edge
            t_edge = target_edge[1] - target_edge[0]
            perp = np.array([t_edge[1], -t_edge[0]])
            direction = perp
            if np.dot(perp, other_v_direction) < 0:
                direction = -1*perp
            edge_index = common_vertices

        current_face = current_face[:, :2] # remove the z coordinate
        new_triangle = match_triangle_to_edge(current_face, edge_index, target_edge, direction)
        new_faces[i] = new_triangle
        previous_face_index = current_face_index

    return aligned_faces

def rotation_matrix_from_vectors(vec1:np.ndarray, vec2:np.ndarray) -> np.ndarray:
    """
    Compute the rotation matrix that rotates 2D vec1 to vec2.
    """
    # normalize the vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    # compute the angle between the vectors
    cos = np.dot(vec1, vec2)
    sin = np.linalg.norm(np.cross(vec1, vec2))
    angle = np.arctan2(sin, cos)

    # compute the rotation matrix
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation

def align_faces(faces:np.ndarray) -> np.ndarray:
    """
    Rotate the faces such that the normal of the faces is aligned with the z-axis.
    """
    normals = compute_triange_normals(faces)
    centers = compute_triangle_centers(faces)
    normals = make_normals_exterior(normals, centers)

    # bring triangles to the origin
    centered_faces = faces - centers[:,None,:]

    # target direction is the z-axis
    z = np.array([0, 0, 1])

    # compute the rotation matrix
    new_faces = np.zeros_like(faces)
    for i, normal in enumerate(normals):
        # if the face is pointing downwards, align with -z
        if normal[2] < 0:
            normal = -normal
        rot, _ = R.align_vectors(normal, z)
        new_faces[i] = apply_local_rotation_matrix(rot.as_matrix(), centered_faces[i])

    return new_faces

def match_triangle_to_edge(triangle:np.ndarray, edge_index:np.ndarray, target_edge:np.ndarray, direction:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match a triangle edge (indexed by edge_index) to an edge in space such that the triangle is facing a specified direction.
    """

    current_edge = triangle[edge_index]
    current = current_edge[1] - current_edge[0]
    target = target_edge[1] - target_edge[0]

    # rotation
    rot = rotation_matrix_from_vectors(current, target)
    new_triangle = apply_local_rotation_matrix(rot, triangle)

    # check if the triangle is facing the right direction
    rotated_edge = new_triangle[edge_index]
    center_edge = np.mean(rotated_edge, axis=0)
    other_vertex_not_in_edge = np.where(np.logical_not(np.isin(np.arange(3), edge_index)))[0][0]
    new_direction = new_triangle[other_vertex_not_in_edge] - center_edge
    if np.dot(new_direction, direction.T) < 0:
        # rotate pi
        rot = np.array([[-1, 0], [0, -1]])
        target_edge = target_edge[::-1]
        new_triangle = apply_local_rotation_matrix(rot, new_triangle)
        
    # translation
    rotated_edge = new_triangle[edge_index]
    translation = target_edge[0] - rotated_edge[0]
    new_triangle = new_triangle + translation

    return new_triangle


def compute_aligned_tdr_disks(n, a, b, c, rotated=False, p=3):
    '''Computes the TDR [a, b, c]

    Args:
        n (int): The number of points to sample on the TDR.
        a (float): The first TDR parameter.
        b (float): The second TDR parameter.
        c (float): The third TDR parameter.

    Returns:
        tdr_ellipse1 (torch.Tensor of shape (n, 3)): The first TDR ellipse.
        tdr_ellipse2 (torch.Tensor of shape (n, 3)): The second TDR ellipse.
    '''
    # Compute the TDR (numpy)
    # tdr_ell1, tdr_ell2 = geom.TDR_disks(n, a, b, c)

    # Compute the TDR
    phi = torch.linspace(0, 2*torch.pi, n, dtype=TORCH_DTYPE)
    phi1 = phi2 = phi

    # HARD CODED FIX TO GET THE ENDPOINTS OF THE ELLIPSES INSIDE THE TDR (PUT THE ENDPOINTS NOT ON THE CONVEX HULL)    
    shift = n//4
    phi1 = torch.roll(phi, shifts=shift)
    phi2 = torch.roll(phi, shifts=-shift)

    # It may happen that the constraint is violated during one step of the optimization, this avoids a crash
    # assert a > b/torch.sqrt(torch.tensor(2.0, dtype=TORCH_DTYPE)), "ratio must be greater than 1/sqrt(2)"
    if c is None:
        try:
            c = torch.sqrt(4*a**2 - 2*b**2)
        except:
            c = 0
    tdr_ell1 = torch.stack([a*torch.sin(phi1) + 0.5*c, b*torch.cos(phi1), torch.zeros(n, dtype=TORCH_DTYPE)], dim=1)
    tdr_ell2 = torch.stack([a*torch.sin(phi2) - 0.5*c, torch.zeros(n, dtype=TORCH_DTYPE), b*torch.cos(phi2)], dim=1)

    # Rotate about the x axis by 45˚
    x_axis = torch.tensor([1, 0, 0], dtype=TORCH_DTYPE)
    angle_x = -np.pi/4
    tdr_ell1 = rotate_about_axis(tdr_ell1, x_axis, angle_x)
    tdr_ell2 = rotate_about_axis(tdr_ell2, x_axis, angle_x)

    # then about the z axis by 45˚ to align with Morton's knot parametrization (if knot is not already rotated)
    if not rotated:
        z_axis = torch.tensor([0, 0, 1], dtype=TORCH_DTYPE)
        angle_z = -np.pi/4 if p % 4 == 3 else np.pi/4 # every other odd p is aligned with x=y instead of x=-y
        tdr_ell1 = rotate_about_axis(tdr_ell1, z_axis, angle_z)
        tdr_ell2 = rotate_about_axis(tdr_ell2, z_axis, angle_z)


    return tdr_ell1, tdr_ell2

