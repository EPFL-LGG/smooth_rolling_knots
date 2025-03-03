import torch
import torch.nn.functional as F

def rotate_about_axis(vectors, axis, angle):
    '''Rotates vectors about an axis by an angle.
    Args:
        vectors (torch.Tensor of shape (n_vectors, 3)): The vectors to rotate.
        axis (torch.Tensor of shape (3,)): The axis of rotation: must be normalized!!
        angle (torch.Tensor of shape (,)): The angle of rotation, using the right-hand rule.
        
    Returns:
        vectors_rotated torch.Tensor of shape (n_vectors, 3): The rotated vectors.
    '''
    if isinstance(angle, float):
        angle = torch.tensor(angle)
    vectors_rotated = vectors * torch.cos(angle) + torch.cross(axis.unsqueeze(0), vectors, dim=-1) * torch.sin(angle) + axis.unsqueeze(0) * (vectors @ axis).unsqueeze(-1) * (1.0 - torch.cos(angle))
    return vectors_rotated

rotate_many_about_same_axis = torch.vmap(rotate_about_axis, in_dims=(0, None, 0))
rotate_many_about_axis = torch.vmap(rotate_about_axis, in_dims=(0, 0, 0))

def rotate_about_axes(vectors, axes, angles):
    '''Rotates vectors about an axis by an angle.
    Args:
        vectors (torch.Tensor of shape (n_vectors, 3)): The vectors to rotate.
        axes (torch.Tensor of shape (n_vectors, 3)): The axes of rotation: must be normalized!!
        angles (torch.Tensor of shape (n_vectors,)): The angles of rotation, using the right-hand rule.
        
    Returns:
        vectors_rotated torch.Tensor of shape (n_vectors, 3): The rotated vectors.
    '''
    vectors_rotated = vectors * torch.cos(angles).unsqueeze(1) + torch.cross(axes, vectors, dim=-1) * torch.sin(angles).unsqueeze(1) + axes * torch.sum(vectors * axes, dim=-1, keepdim=True) * (1.0 - torch.cos(angles).unsqueeze(1))
    return vectors_rotated

def reflect_about_plane(pts, plane_normal):
    plane_normal = plane_normal / torch.norm(plane_normal)
    return pts - 2 * torch.sum(pts * plane_normal, dim=1).reshape(-1, 1) * plane_normal

def rotation_matrix_from_vectors(vec1, vec2):
    """ 
    Find the rotation matrix that aligns vec1 to vec2 using PyTorch.
    Adapted from: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space

    Args:
        vec1 (torch.Tensor): A 3D "source" vector.
        vec2 (torch.Tensor): A 3D "destination" vector.

    Returns:
        torch.Tensor: A 3x3 transformation matrix which, when applied to vec1, aligns it with vec2 — apply as torch.matmul(rotation_matrix, vec1) or torch.matmul(vec1, rotation_matrix.T).
    """
    a, b = (vec1 / torch.linalg.norm(vec1)).reshape(3), (vec2 / torch.linalg.norm(vec2)).reshape(3)
    if torch.dot(a, b) > 1 - 1e-8:
        return torch.eye(3)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = torch.linalg.norm(v)
    kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = torch.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

def project_vector_on_plane_along_direction(v, n, u):
    """
    Projects a vector v onto the plane defined by the normal n along the direction u.
    Source: https://math.stackexchange.com/questions/4108428/how-to-project-vector-onto-a-plane-but-not-along-plane-normal
    """
    n = n / torch.linalg.norm(n)
    u = u / torch.linalg.norm(u)
    return v - ((v @ n) / (u @ n)) * u

def point_mesh_squared_distance(pts, V_surf, F_surf):
    """
    Compute the squared distance between each point in pts and the mesh defined by V and F.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/point_mesh_distance.html

    Note: the function igl.point_mesh_squared_distance returned very reasonable values on a test mesh, while
          pytorch3d.loss.point_mesh_distance.point_mesh_face_distance did not. 
          Here we are using part of the source code of the latter that matches the behavior of the former.
          (TODO: better solution?)

    Args:
        pts (torch.Tensor of shape (N, 3)): The points.
        V_surf (torch.Tensor of shape (V, 3)): The vertices of the mesh.
        F_surf (torch.Tensor of shape (F, 3)): The faces of the mesh.
    """
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.loss.point_mesh_distance import point_face_distance

    # Cast to float32: pytorch3d requires float32 (TODO: check if this is necessary)
    pts = pts.to(torch.float32)
    V_surf = V_surf.to(torch.float32)
    F_surf = F_surf.to(torch.int64)

    meshes = Meshes(verts=[V_surf], faces=[F_surf])
    pcls = Pointclouds(points=[pts])

    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    return torch.sum(point_to_face)

def sin_angle(a, v1, v2):
    """
    Get the sin of the signed angle from v1 to v2 around axis "a". Uses right hand rule
    as the sign convention: clockwise is positive when looking along vector.
    Assumes all vectors are normalized.

    Returns:
        float: The sin of the signed angle.
    """
    return torch.dot(torch.cross(v1, v2, dim=-1), a)

def angle_around_axis(a, v1, v2):
    """
    Get the signed angle from v1 to v2 around axis "a". Uses right hand rule
    as the sign convention: clockwise is positive when looking along vector.
    Assumes all vectors are normalized **and perpendicular to a**.
    Return answer in the range [-pi, pi].

    Returns:
        float: The signed angle in the range [-pi, pi].
    """
    s = torch.clamp(sin_angle(a, v1, v2), -1.0, 1.0)
    c = torch.clamp(torch.dot(v1, v2), -1.0, 1.0)
    return torch.atan2(s, c)

def triangle_normal(v1, v2, v3):
    '''
    Args:
        v1 (torch.Tensor of shape (3,)): The first vertex of the triangle.
        v2 (torch.Tensor of shape (3,)): The second vertex of the triangle.
        v3 (torch.Tensor of shape (3,)): The third vertex of the triangle.
        
    Returns:
        n (torch.Tensor of shape (3,)): The normal of the triangle.
    '''
    e1 = (v2 - v1) / torch.linalg.norm(v2 - v1)
    e2 = (v3 - v1) / torch.linalg.norm(v3 - v1)
    n = torch.cross(e1, e2, dim=-1)
    n = n / torch.linalg.norm(n)
    return n

def triangle_area(v1, v2, v3):
    a = torch.linalg.norm(v2 - v1)
    b = torch.linalg.norm(v3 - v2)
    c = torch.linalg.norm(v1 - v3)
    s = 0.5 * (a + b + c)
    return torch.sqrt(s * (s - a) * (s - b) * (s - c))

def triangulate_polygon(pts):
    '''
    Triangulate a polygon defined by the points pts.
    
    Args:
        pts (torch.Tensor of shape (n, 3)): The points defining the polygon.
        
    Returns:
        faces (torch.Tensor of shape (n-2, 3)): The indices of the vertices of the triangles.
    '''
    n = pts.shape[0]
    faces = torch.zeros((n-2, 3), dtype=torch.int64)
    for i in range(1, n-1):
        faces[i-1] = torch.tensor([0, i, i+1])
    return faces

def compute_polygonal_area(x, y):
    "Shoelace formula, source: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates"
    return 0.5 * torch.abs(torch.dot(x, torch.roll(y, 1)) - torch.dot(y, torch.roll(x, 1)))

def compute_curve_length(pts_crv):
    '''Compute the length of a curve defined by pts_crv.'''
    return torch.sum(torch.linalg.norm(pts_crv[1:] - pts_crv[:-1], dim=1))

def compute_angle_mismatch(t1, t2):
    '''Compute the mismatch in curvature angles between two polygons. 
    Similar polygons result in zero mismatch irrespective of their scaling factor.'''

    assert t1.shape[1] == t2.shape[1] == 3
    
    curv_angles_shape = compute_curvature_angles(t1, closed_curve=True)
    curv_angles_target = compute_curvature_angles(t2, closed_curve=True)  # TODO: use signed curvature
    angles_shape = torch.pi - curv_angles_shape
    angles_target = torch.pi - curv_angles_target
    mismatch = torch.inf
    for r in range(len(angles_shape)):  # TODO: optimize this, vectorize and use convolution
        mismatch_curr = torch.sum((angles_shape.roll(r) - angles_target)**2)
        mismatch = min(mismatch, mismatch_curr)
    return mismatch

def is_quad_planar(vertices):
    '''Check if a quad is planar.'''
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    v3 = vertices[3] - vertices[0]
    na = torch.cross(v1, v2, dim=-1)
    na = na / torch.linalg.norm(na)
    nb = torch.cross(v2, v3, dim=-1)
    nb = nb / torch.linalg.norm(nb)
    atol = 1.0e-12
    if torch.allclose(na, nb, atol=atol):
        return True
    elif torch.allclose(na, -nb, atol=atol):  # TODO: does this cover all cases in which an inversion occurs?
        print("WARNING: The computed normals of a quad have opposite signs, there might have been an inversion in the quad.")
        return True  # the quad can still be considered planar
    return False

def align_point_cloud(points, pure_rotation=True):
    '''Aligns point clouds to xyz axes'''
    centered_points = points - torch.mean(points, dim=0, keepdim=True)
    cov = torch.einsum("si, sj -> ij", centered_points, centered_points)
    L, Q = torch.linalg.eigh(cov)
    Q = Q[:, torch.argsort(L, descending=True)]

    if pure_rotation and torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    rotated_pos = centered_points @ Q
    return rotated_pos
    
def point_cloud_is_planar(pts, tol=1e-8):
    '''Check if the point cloud is planar.'''
    if pts.shape[0] < 3:  # The point cloud has less than 3 points
        return True
    if pts.shape[1] == 2:  # The point cloud is 2D
        return True
    pts_centered = pts - torch.mean(pts, dim=0)
    cov = pts_centered.T @ pts_centered
    eigvals, eigvecs = torch.linalg.eig(cov)
    return torch.min(torch.abs(eigvals)) < tol

def find_shared_points(pts1, pts2, tol=1e-5):
    """
    Find the indices of the points that are present in both the given sets within a certain tolerance.

    Args:
        pts1 (torch.Tensor (n, 3)): The first set of 3D points.
        pts2 (torch.Tensor (m, 3)): The second set of 3D points.
        tol (float): The tolerance within which two points are considered to be the same.

    Returns:
        shared_indices (List[List[int]]): The indices of the shared points in the format [[i1, j1], [i2, j2], ...].
    """
    diff = pts1[:, None, :] - pts2[None, :, :]
    dist = torch.norm(diff, dim=2)
    shared_indices = torch.nonzero(dist < tol, as_tuple=False)
    return shared_indices.tolist()

def find_duplicate_points(pts, tol=1e-5):
    """
    Find the indices of the duplicate points within a certain tolerance.

    Args:
        pts (torch.Tensor (n, 3)): The set of 3D points.
        tol (float): The tolerance within which two points are considered to be the same.

    Returns:
        duplicate_indices (List[List[int]]): The indices of the duplicate points in the format [[i1, i2], [i3, i4], ...].
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist = torch.norm(diff, dim=2)
    duplicate_indices = torch.nonzero(dist < tol, as_tuple=False)
    duplicate_indices = duplicate_indices[duplicate_indices[:, 0] != duplicate_indices[:, 1]]
    duplicate_indices = duplicate_indices[duplicate_indices[:, 0] < duplicate_indices[:, 1]]  # remove flipped pairs: only keep one of the two
    return duplicate_indices.tolist()

# --------------------------------------------------------------------------------
# Curve utils
# --------------------------------------------------------------------------------

def compute_tangents(pts_crv):
    '''Computes the edge tangents of a curve defined by pts_crv.

    Args:
        pts_crv (torch.Tensor of shape (n_disc, 3)): The points defining the curve.

    Returns:
        tangents (torch.Tensor of shape (n_disc-1, 3)): The tangent vectors on the edges.
    '''
    tangents = pts_crv[1:] - pts_crv[:-1]
    tangents = tangents / torch.linalg.norm(tangents, dim=1, keepdim=True)
    return tangents

def compute_vertex_tangents(pts_crv, kind='bisecting', closed_curve=False):
    '''Computes the vertex tangents of a curve defined by pts_crv.

    Args:
        pts_crv (torch.Tensor of shape (n_disc, 3)): 
            The points defining the curve.
        kind (str, optional): 
            If 'bisecting', the vertex tangent is computed by bisectrix of the curvature angle. 
            If 'integrated', the vertex tangent is computed by integrating the tangent along the voronoi cell.
            If 'reflective', the vertex tangent is the normal of the plane that reflects one edge tangent to the next. 
                It is computed by rotating the bisecting tangent by 90˚ about the binormal 
                (not really a "tangent", TODO: change function name into compute_plane_normals?)
        closed_curve (bool, optional): 
            If True, the curve is assumed to be closed. The first and last points must be the same.

    Returns:
        vertex_tangents (torch.Tensor of shape (n_disc, 3)): The tangent vectors at the vertices.
    '''
    tangents = compute_tangents(pts_crv)
    if kind == 'integrated':
        vertex_tangents = torch.cat([(pts_crv[1] - pts_crv[0]).unsqueeze(0), pts_crv[2:] - pts_crv[:-2], (pts_crv[-1] - pts_crv[-2]).unsqueeze(0)], dim=0)
        vertex_tangents = vertex_tangents / torch.linalg.norm(vertex_tangents, dim=1, keepdim=True)
    elif kind == 'bisecting' or kind == 'reflective':
        if closed_curve:
            vt0 = (tangents[0] + tangents[-1]).unsqueeze(0)
            vtnm1 = vt0
        else:   # use edge tangent for the first and last vertex
            vt0 = tangents[0].unsqueeze(0)
            vtnm1 = tangents[-1].unsqueeze(0)
        if kind == 'bisecting':
            vertex_tangents = torch.cat([vt0, tangents[1:] + tangents[:-1], vtnm1], dim=0)
        elif kind == 'reflective':
            vertex_tangents = torch.cat([vt0, tangents[1:] - tangents[:-1], vtnm1], dim=0)
        vertex_tangents = vertex_tangents / torch.linalg.norm(vertex_tangents, dim=1, keepdim=True)
    else:
        raise ValueError("kind must be 'integrated', 'bisecting', or 'reflective'.")
    return vertex_tangents

def compute_binormals(pts_crv, closed_curve=False):
    tangents = compute_tangents(pts_crv)
    if closed_curve:
        b0 = torch.cross(tangents[-1], tangents[0], dim=0)
        bnm1 = b0
    else:  # assign the edge binormal of the second vertex to the first one; same at the end
        b0 = torch.cross(tangents[0], tangents[1], dim=0)
        bnm1 = torch.cross(tangents[-2], tangents[-1], dim=0)
    binormals = torch.cat([b0.unsqueeze(0), torch.cross(tangents[:-1], tangents[1:], dim=1), bnm1.unsqueeze(0)], dim=0)
    binormals = binormals / torch.norm(binormals, dim=1, keepdim=True)
    return binormals

# def compute_curvature_angles(pts_crv, closed_curve=False):
#     n_disc = pts_crv.shape[0]
#     tangents = compute_tangents(pts_crv)
#     curvature_angles = torch.zeros(n_disc)
#     for id_vertex in range(1, n_disc-1):
#         t0 = tangents[id_vertex-1]
#         t1 = tangents[id_vertex]
#         curvature_angles[id_vertex] = angle_around_axis(torch.cross(t0, t1, dim=-1), t0, t1)
#     if closed_curve:
#         t0 = tangents[0]
#         first_last_overlap = torch.allclose(pts_crv[0], pts_crv[-1], rtol=1e-5)
#         tn = tangents[-1] if first_last_overlap else (pts_crv[0] - pts_crv[-1]) / torch.linalg.norm(pts_crv[0] - pts_crv[-1])
#         curvature_angles[0] = angle_around_axis(torch.cross(tn, t0, dim=-1), tn, t0)
#         curvature_angles[-1] = curvature_angles[0]
#     return curvature_angles

def compute_curvature_angles(pts_crv, closed_curve=False):
    first_last_overlap = torch.allclose(pts_crv[0], pts_crv[-1], rtol=1.0e-5)
    if closed_curve and not first_last_overlap:
        tangents_non_norm = (pts_crv - torch.roll(pts_crv, 1, dims=0))
        tangents = tangents_non_norm / torch.linalg.norm(tangents_non_norm, dim=1, keepdim=True)
    else:
        tangents = (pts_crv[1:] - pts_crv[:-1]) / torch.linalg.norm(pts_crv[1:] - pts_crv[:-1], dim=1, keepdim=True)
    normal = torch.cross(tangents[:-1], tangents[1:], dim=1)
    sin_angle = torch.clamp(torch.sum(normal * normal, dim=1), -1.0, 1.0)
    cos_angle = torch.clamp(torch.sum(tangents[:-1] * tangents[1:], dim=1), -1.0, 1.0)
    curvature_angles = torch.atan2(sin_angle, cos_angle)
    return curvature_angles

def compute_torsion_angles(pts_crv, closed_curve=False):
    n_disc = pts_crv.shape[0]
    tangents = compute_tangents(pts_crv)
    binormals = compute_binormals(pts_crv, closed_curve)
    torsion_angles = torch.zeros(n_disc-1)
    for id_vertex in range(0, n_disc-1):
        torsion_angles[id_vertex] = angle_around_axis(tangents[id_vertex], binormals[id_vertex+1], binormals[id_vertex])
    return torsion_angles

def rotate_and_translate_profile_curve(profile_curve, translation, plane_normal, rotation_angle=0.0):
    """
    Rotate a profile curve to lie in the plane defined by the plane normal and translate it by the translation vector.
    Optionally, rotate the profile curve by the rotation angle about the plane normal.
    """
    assert profile_curve.shape[1] == 2 or (profile_curve.shape[1] == 3 and torch.allclose(profile_curve[:, 2], 0.0)), "Profile curve must be 2D or 3D with z-coordinates equal to zero."
    if profile_curve.shape[1] == 2:
        profile_curve = torch.cat([profile_curve, torch.zeros_like(profile_curve[:, 0]).unsqueeze(1)], dim=1)

    if rotation_angle != 0.0:
        profile_curve = rotate_about_axis(profile_curve, plane_normal, rotation_angle)
    
    plane_normal = plane_normal / torch.norm(plane_normal)
    e1 = torch.cross(plane_normal, torch.tensor([0.0, 0.0, 1.0]), dim=0)
    e1 = e1 / torch.norm(e1)
    e2 = torch.cross(plane_normal, e1, dim=0)
    rot_xyz_to_frame = torch.stack([e1, e2, plane_normal], dim=1)
    return translation + profile_curve @ rot_xyz_to_frame.T

def resample_polyline(pts, n, sample_map=None):
    """
    Resample a polyline defined by the points pts to have n points approximately equispaced along the curve.
    sample_map is a function that maps the cumulative distance to the new distance. If None, the points are spaced equally.
    """
    from scipy import interpolate
    import numpy as np

    # Convert pts to numpy array
    pts_np = pts.numpy()

    # Compute the cumulative arc length along the polyline
    distances = np.sqrt(np.sum(np.diff(pts_np, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

    # Create an interpolation function for each dimension
    interpolators = [interpolate.interp1d(cumulative_distances, pts_np[:, dim]) for dim in range(pts_np.shape[1])]

    # Generate n equally spaced points along the cumulative distance
    new_distances = np.linspace(0, cumulative_distances[-1], n)
    if sample_map is not None:
        new_distances = sample_map(new_distances)

    # Interpolate the points at the new distances
    resampled_pts_np = np.vstack([interpolator(new_distances) for interpolator in interpolators]).T

    # Convert resampled points back to torch tensor
    resampled_pts = torch.tensor(resampled_pts_np, dtype=pts.dtype)

    return resampled_pts

# --------------------------------------------------------------------------------
# Tube utils
# --------------------------------------------------------------------------------

def is_tube_pq(swept_vertices):
    '''Check if the tube is a PQ-tube, i.e. each of its quad faces is planar.

    Args:
        swept_vertices (torch.Tensor of shape (n_disc, n_cross_section, 3)): The vertices of the tube.   
    '''
    from tube_generation import compute_tube_topology

    n_disc = swept_vertices.shape[0]
    n_cross_section = swept_vertices.shape[1]
    quads, _ = compute_tube_topology(n_disc, n_cross_section)
    swept_vertices_reshaped = swept_vertices.reshape(-1, 3)
    n_quads = quads.shape[0]  # 3*(n_disc-1)
    for i in range(n_quads):
        quad = swept_vertices_reshaped[quads[i]]
        if not is_quad_planar(quad):
            return False
    return True

def compute_quad_nonplanarity(vertices):
    '''
    Compute nonplanarity according to the measure from [Pottmann et al. 2008].
    Given the 4 vertices of a quad as q1, q2, p1, p2, the nonplanarity is defined as the distance between the two lines:
        l1: p1 v q2
    and 
        l2: q1 v p2
    where (a v b) denotes the line spanned by the two points a and b.

    Args:
        vertices: torch.Tensor of shape (n_disc, n_cross_section, 3)

    Returns:
        torch.Tensor of shape (n_disc-1, n_cross_section)
    '''

    # Helper functions
    def line_coefficients(point1: torch.Tensor, point2: torch.Tensor):
        """
        Compute the coefficients of the line passing through two points in 3D space.

        Args:
            point1 (torch.Tensor): A tensor of shape (3,) representing the first point (x1, y1, z1).
            point2 (torch.Tensor): A tensor of shape (3,) representing the second point (x2, y2, z2).

        Returns:
            r0 (torch.Tensor): A tensor representing the starting point (point1).
            v (torch.Tensor): A tensor representing the direction vector of the line.
        """
        if point1.shape != (3,) or point2.shape != (3,):
            raise ValueError("Both points must be 3D tensors with shape (3,).")

        # r0 is simply the first point
        r0 = point1

        # v is the direction vector from point1 to point2
        v = point2 - point1

        return r0, v

    def minimal_distance_between_lines(r0_1: torch.Tensor, v1: torch.Tensor, r0_2: torch.Tensor, v2: torch.Tensor):
        """
        Compute the minimal distance between two lines in 3D space.

        Args:
            r0_1 (torch.Tensor): A tensor representing a point on the first line.
            v1 (torch.Tensor): A tensor representing the direction vector of the first line.
            r0_2 (torch.Tensor): A tensor representing a point on the second line.
            v2 (torch.Tensor): A tensor representing the direction vector of the second line.

        Returns:
            float: The minimal distance between the two lines.
        """
        if r0_1.shape != (3,) or v1.shape != (3,) or r0_2.shape != (3,) or v2.shape != (3,):
            raise ValueError("All inputs must be 3D tensors with shape (3,).")

        # Compute the cross product of the direction vectors
        cross_v1_v2 = torch.cross(v1, v2, dim=0)

        # If the cross product is zero, the lines are parallel or coincident
        if torch.norm(cross_v1_v2) == 0:
            # Compute the distance between a point on one line to the other line
            # Since the lines are parallel, pick any point on the second line
            diff = r0_2 - r0_1
            projection = torch.dot(diff, v1) / torch.norm(v1)
            closest_point_on_line1 = r0_1 + projection * (v1 / torch.norm(v1))
            return torch.norm(closest_point_on_line1 - r0_2).item()

        # Compute the vector between points on the two lines
        diff_r0 = r0_2 - r0_1

        # Minimal distance is the projection of diff_r0 onto the unit normal vector of the plane formed by v1 and v2
        distance = torch.abs(torch.dot(diff_r0, cross_v1_v2)) / torch.norm(cross_v1_v2)

        return distance.item()

    # Compute the nonplanarity
    n_disc, n_cross_section, _ = vertices.shape
    nonplanarities = []
    for i in range(1, n_disc):
        for j in range(n_cross_section):
            p1 = vertices[i-1, j]
            q1 = vertices[i-1, (j+1) % n_cross_section]
            p2 = vertices[i, j]
            q2 = vertices[i, (j+1) % n_cross_section]
            r0_1, v1 = line_coefficients(p1, q2)
            r0_2, v2 = line_coefficients(q1, p2)
            nonplanarities.append(minimal_distance_between_lines(r0_1, v1, r0_2, v2))
    return torch.tensor(nonplanarities).reshape(n_disc-1, n_cross_section)

def compute_cross_section_areas(swept_vertices):
    '''Compute the areas of the swept cross-sections.'''
    n_disc = swept_vertices.shape[0]
    n_cross_section = swept_vertices.shape[1]
    areas = torch.zeros(size=(n_disc,))
    for i in range(n_disc):
        centroid = torch.mean(swept_vertices[i, :, :], dim=0)
        swept_vertices_centered = swept_vertices[i, :, :] - centroid
        n = triangle_normal(*swept_vertices_centered[0:3])
        # Define the rotation that aligns the normal to the z-axis
        e1 = swept_vertices_centered[0] / torch.linalg.norm(swept_vertices_centered[0], dim=0)
        e2 = torch.cross(n, e1, dim=0)
        R = torch.stack((e1, e2, n))
        swept_vertices_centered_xy_plane = swept_vertices_centered @ R.T
        area = compute_polygonal_area(swept_vertices_centered_xy_plane[:, 0], swept_vertices_centered_xy_plane[:, 1])
        areas[i] = area
    return areas

def compute_cross_section_area(swept_vertices):
    '''Compute the areas of the swept cross-sections.'''
    swept_vertices_centered = swept_vertices - torch.mean(swept_vertices, dim=0).reshape(1, -1)
    n = triangle_normal(*swept_vertices_centered[0:3])
    # Define the rotation that aligns the normal to the z-axis
    e1 = swept_vertices_centered[0] / torch.linalg.norm(swept_vertices_centered[0], dim=0)
    e2 = torch.cross(n, e1, dim=0)
    R = torch.stack((e1, e2, n))
    swept_vertices_centered_xy_plane = swept_vertices_centered @ R.T
    return compute_polygonal_area(swept_vertices_centered_xy_plane[:, 0], swept_vertices_centered_xy_plane[:, 1])
vmap_compute_cross_section_area = torch.vmap(compute_cross_section_area, in_dims=(0,))

def compute_cross_section_radii(pts_crv, swept_vertices, on_normal_plane=False):
    '''Compute the radii of the swept cross-sections as half of the maximum distance between any two vertices of the cross-section.'''
    if on_normal_plane:
        # Assume that the curve is closed if 1) the first and the last points are the same AND 2) the first and the last cross-sections have the same normal
        n0 = triangle_normal(*swept_vertices[0, 0:3])
        n1 = triangle_normal(*swept_vertices[-1, 0:3])
        closed_curve = torch.allclose(pts_crv[0], pts_crv[-1]) and (torch.allclose(n0, n1) or torch.allclose(n0, -n1))

        # Project the vertices onto the normal bisecting planes before computing the distances
        n_disc = swept_vertices.shape[0]
        n_cross_section = swept_vertices.shape[1]
        tangents = compute_tangents(pts_crv)
        vertex_tangents = compute_vertex_tangents(pts_crv, kind='bisecting', closed_curve=closed_curve)
        swept_vertices_on_normal_plane = torch.zeros_like(swept_vertices)
        # for i in range(1, n_disc):
        #     swept_vertices_on_normal_plane[i] = pts_crv[i] + project_vector_on_plane_along_direction(swept_vertices[i] - pts_crv[i], vertex_tangents[i], tangents[i-1])
        # swept_vertices_on_normal_plane[0] = pts_crv[0] + project_vector_on_plane_along_direction(swept_vertices[0] - pts_crv[0], vertex_tangents[0], -tangents[0])
        for i in range(0, n_disc):
            for csi in range(n_cross_section):
                swept_vertices_on_normal_plane[i, csi] = pts_crv[i] + project_vector_on_plane_along_direction(swept_vertices[i, csi] - pts_crv[i], vertex_tangents[i], vertex_tangents[i])
        return compute_cross_section_radii(pts_crv, swept_vertices_on_normal_plane, on_normal_plane=False)
    else:
        n_disc = swept_vertices.shape[0]
        n_cross_section = swept_vertices.shape[1]
        radii = torch.zeros(size=(n_disc,))
        for i in range(n_disc):
            dist_matrix = torch.cdist(swept_vertices[i].unsqueeze(0), swept_vertices[i].unsqueeze(0)).squeeze(0)
            max_dist = torch.max(dist_matrix)
            radii[i] = 0.5 * max_dist
        return radii

def compute_quad_normals(vertices, quad_tube_indices, quad_disc_indices, quad_cross_section_indices):
    '''Compute the normal of a (list of) quads.

    Args:
        vertices (list of torch Tensors of shape (n_disc, n_cross_section, 3)): The vertices of the tubes, one tensor per tube.
        quad_tube_indices (torch.Tensor of shape (n_quads,)): The indices of the tubes.
        quad_disc_indices (torch.Tensor of shape (n_quads,)): The indices of the directrix points.
        quad_cross_section_indices (torch.Tensor of shape (n_quads,)): The indices of the cross-sections.
    '''
    assert quad_tube_indices.shape == quad_disc_indices.shape == quad_cross_section_indices.shape, "The indices must have the same shape."
    n_quads = quad_tube_indices.shape[0]
    normals = torch.zeros(size=(n_quads, 3))
    for i in range(n_quads):
        ti = quad_tube_indices[i]
        di = quad_disc_indices[i]
        ci = quad_cross_section_indices[i]
        ci1 = (ci + 1) % vertices[ti].shape[1]  # Wrap around (assumes closed cross-sections)
        quad = torch.stack((vertices[ti][di, ci], vertices[ti][di, ci1], vertices[ti][di+1, ci1], vertices[ti][di+1, ci]))
        n = torch.cross(quad[1] - quad[0], quad[2] - quad[0], dim=0)
        n = n / torch.norm(n)
        normals[i] = n
    return normals

def compute_quad_centers(vertices, quad_tube_indices, quad_disc_indices, quad_cross_section_indices):
    '''Compute the center of a (list of) quads.

    Args:
        vertices (list of torch Tensors of shape (n_disc, n_cross_section, 3)): The vertices of the tubes, one tensor per tube.
        quad_tube_indices (torch.Tensor of shape (n_quads,)): The indices of the tubes.
        quad_disc_indices (torch.Tensor of shape (n_quads,)): The indices of the directrix points.
        quad_cross_section_indices (torch.Tensor of shape (n_quads,)): The indices of the cross-sections.

    Returns:
        torch.Tensor of shape (n_quads, 3): The centers of the quads.
    '''
    assert quad_tube_indices.shape == quad_disc_indices.shape == quad_cross_section_indices.shape, "The indices must have the same shape."
    n_quads = quad_tube_indices.shape[0]
    centers = torch.zeros(size=(n_quads, 3))

    for i in range(n_quads):
        ti = quad_tube_indices[i]
        di = quad_disc_indices[i]
        ci = quad_cross_section_indices[i]
        ci1 = (ci + 1) % vertices[ti].shape[1]  # Wrap around (assumes closed cross-sections)
        quad = torch.stack((vertices[ti][di, ci], vertices[ti][di, ci1], vertices[ti][di+1, ci1], vertices[ti][di+1, ci]))
        centers[i] = torch.mean(quad, dim=0)
    return centers

# --------------------------------------------------------------------------------
# Parallel transport
# --------------------------------------------------------------------------------

def parallel_transport_normalized(t0, t1, v):
    '''Parallel transports a vector v from the tangent t0 to the tangent t1. Assumes t0 and t1 are normalized.'''
    sin_theta_axis = torch.cross(t0, t1)
    cos_theta = torch.dot(t0, t1)
    den = 1 + cos_theta
    if torch.abs(den) < 1e-14:
        return v
    if torch.allclose(t0, t1):
        return v
    sin_theta_axis_dot_v = torch.dot(sin_theta_axis, v)
    sin_theta_axis_cross_v = torch.cross(sin_theta_axis, v)
    res = (sin_theta_axis_dot_v / den) * sin_theta_axis + sin_theta_axis_cross_v + cos_theta * v
    return res

def parallel_transport(t0, t1, v):
    '''Parallel transports a vector v from the tangent t0 to the tangent t1.'''
    t0_normalized = t0 / torch.linalg.norm(t0)
    t1_normalized = t1 / torch.linalg.norm(t1)
    return parallel_transport_normalized(t0_normalized, t1_normalized, v)

# --------------------------------------------------------------------------------
# Interpolations
# --------------------------------------------------------------------------------

def slerp(p0, p1, n_inter):
    '''
    Args:
        p0: torch.tensor of shape (dim,)
        p1: torch.tensor of shape (dim,)
        n_inter: int, number of output points (must be greater than 2)
        
    Returns:
        torch.tensor of shape (n_inter, dim)
    '''
    ts = torch.linspace(0.0, 1.0, n_inter).unsqueeze(1)
    omega = torch.acos(torch.dot(p0, p1) / (torch.linalg.norm(p0) * torch.linalg.norm(p1)))
    sin_omega = torch.sin(omega)
    return (torch.sin((1.0 - ts) * omega) * p0.unsqueeze(0) + torch.sin(ts * omega) * p1.unsqueeze(0)) / sin_omega

# --------------------------------------------------------------------------------
# Rotation utils
# --------------------------------------------------------------------------------

def quaternion_to_matrix(quaternions):
    '''
    Convert rotations given as quaternions to rotation matrices.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.quaternion_to_matrix

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    '''
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def sqrt_positive_part(x):
    '''
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#_sqrt_positive_part
    '''
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions):
    '''
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#standardize_quaternion

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    '''
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def matrix_to_quaternion(matrix):
    '''
    Convert rotations given as rotation matrices to quaternions.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        
    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    '''
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def quaternion_to_axis_angle(quaternions):
    '''
    Convert rotations given as quaternions to axis/angle.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    '''
    norms = torch.linalg.norm(quaternions[..., 1:], dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2.0 * half_angles
    eps = 1.0e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48.0
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def axis_angle_to_quaternion(axis_angle):
    '''
    Convert rotations given as axis/angle to quaternions.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#axis_angle_to_quaternion

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    '''
    angles = torch.linalg.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1.0e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48.0
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#axis_angle_to_matrix

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_axis_angle

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

# --------------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    a = torch.tensor([torch.cos(torch.tensor(0.1)), torch.sin(torch.tensor(0.1)), 0.0]).reshape(1, 3) 
    b = torch.tensor([1.0 , 0.0, 0.0]).reshape(1, 3)
    axis_rot = torch.cross(a[0], b[0], dim=0)
    norm_axis = torch.linalg.norm(axis_rot)
    cos_theta = a[0] @ b[0]
    theta = torch.atan2(norm_axis, cos_theta)
    a_rotated = rotate_about_axis(a, axis_rot/norm_axis, theta)
    print(a_rotated)