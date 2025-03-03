import sys
sys.path.append("..")
import numpy as np
from src.geometry_src import distances
import torch
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from src.geometry_src.geom import rotated_morton_knot_parametric as morton_knot
from src.geometry_src.rolliness import convex_hull_angle
from src.optimization_src.TDR_projection import project_knot_to_disks, split_index_list_in_two_segments
from scipy.interpolate import RBFInterpolator
from src.geometry_src.geom_utils import compute_aligned_tdr_disks
from src.geometry_src import geom

TORCH_DTYPE = torch.float64

def distance(vertices1,vertices2):
    # return distances.chamfer_distance_squared(vertices1,vertices2)
    return distances.points_to_polyline_distance(vertices1,vertices2)


def evaluate_knot_tdr_distance(knot, a, b, n_tdr, exterior_indices=None, rotated=False, p=3):
    '''Evaluates the distance between the knot and the TDR [a, b, c].
    Assumes the knot it oriented according to Morton's parametrization – the computed TDR is aligned accordingly.
    
    Args:
        knot (torch.Tensor of shape (n, 3)): The knot.
        a (float): The first TDR parameter.
        b (float): The second TDR parameter.
        exterior_indices (torch.Tensor of shape (n_ext,)): The indices of the knot's exterior vertices (convex hull).
        rotated: flag to indicate if the knot is rotated (in which case we don't align the TDR)
    
    Returns:
        obj (torch.Tensor of shape (,)): The chamfer distance between the knot and the TDR.
    '''
    # Compute knot's convex hull
    if exterior_indices is None:
        hull = ConvexHull(knot)
        exterior_indices = hull.vertices
        
    idx_seg1, idx_seg2 = split_index_list_in_two_segments(exterior_indices)

    exterior_vertices1 = knot[idx_seg1]
    exterior_vertices2 = knot[idx_seg2]

    c = torch.sqrt(4*a**2 - 2*b**2)
    
    tdr_ell1, tdr_ell2 = compute_aligned_tdr_disks(n_tdr, a, b, c, rotated=rotated, p=p)

    # distance
    d_lobe1 = torch.min(distance(exterior_vertices1, tdr_ell1), distance(exterior_vertices1, tdr_ell2))
    d_lobe2 = torch.min(distance(exterior_vertices2, tdr_ell1), distance(exterior_vertices2, tdr_ell2))
    assert torch.abs(d_lobe1 - d_lobe2) < 1e-6, torch.abs(d_lobe1 - d_lobe2)  # due to symmetry, should go to zero as the discretization is refined
    return d_lobe1 + d_lobe2


def optimize_TDR_params(knot_ref, exterior_indices, rotated=True):  

    n_tdr = knot_ref.shape[0] + 1
    
    global opt_iter
    opt_iter = 0
    
    history = []
    a0, b0 = 2 - np.sqrt(2.0), 2 - np.sqrt(2.0) # make it such that c+2a = 2 (TDR is on the scale of the knot, which has width 2)
    zscale0 = 1.0
    a0_over_b0 = a0 / b0  # reparametrize to avoid the constraint a > b/sqrt(2), bounds will be set on a/b
    params0 = np.array([a0, a0_over_b0, zscale0])

    def obj_and_grad(params):
        params_torch = torch.tensor(params, dtype=TORCH_DTYPE)
        params_torch.requires_grad = True

        a = params_torch[0]
        a_over_b = params_torch[1]
        b = a / a_over_b
        zscale = params_torch[2]
        knot = knot_ref.clone()
        knot[:, 2] = zscale * knot_ref[:, 2]
        
        obj = evaluate_knot_tdr_distance(knot, a, b, n_tdr, exterior_indices, rotated=rotated)  # pass exterior_indices assuming the points defining the convex hull won't change

        obj.backward()

        return obj.item(), params_torch.grad.numpy()
    
    obj_and_grad_scipy = lambda params: obj_and_grad(params)

    # Set the same value for both lower and upper bounds to fix a parameter
    bounds = [(None, None)] * len(params0)  # Initialize with no bounds
    rel_tol = 1e-3  # make sure a/b does not go below 1/sqrt(2), bounds might not be respected exactly
    bounds = [
        (None, None), # a
        (1/torch.sqrt(torch.tensor(2.0)) * (1 + rel_tol), None), # a/b
        (0.0, None) # zscale
    ]

    fixed_indices = []
    # fixed_indices = [2]  # <---------------- uncomment to pin zscale
    print(f"Fixing {len(fixed_indices)} parameters.")

    for idx in fixed_indices:
        bounds[idx] = (params0[idx], params0[idx])

    torch.autograd.set_detect_anomaly(True)
 
    history.append(params0)

    def callback(params):
        global opt_iter
        opt_iter += 1
        history.append(params)

    result = minimize(
        obj_and_grad_scipy, params0, jac=True, 
        method='L-BFGS-B',
        options={'ftol':1.0e-12, 'gtol':1.0e-7, 'disp': True, 'maxiter': 50},
        bounds=bounds,
        callback=callback,
    )

    a_opt = result.x[0]
    b_opt = a_opt / result.x[1]
    c_opt = np.sqrt(4*a_opt**2 - 2*b_opt**2)
    zscale_opt = result.x[2]

    return a_opt, b_opt, c_opt, zscale_opt

def interpolate_interior(knot, n_tdr, a_opt, b_opt, exterior_indices, rotated=True):
    proj_hull_vertices1, proj_hull_vertices2 = project_knot_to_disks(knot, n_tdr, a_opt, b_opt, rotated=rotated)
    exterior_indices1, exterior_indices2 = split_index_list_in_two_segments(exterior_indices)
    exterior_vertices = knot[np.concatenate([exterior_indices1, exterior_indices2])]
    proj_hull_vertices = np.concatenate([proj_hull_vertices1, proj_hull_vertices2])
    rbf = RBFInterpolator(exterior_vertices, proj_hull_vertices, kernel='cubic', epsilon=1, smoothing=0)
    return rbf(knot)

def full_pipeline(a, p, n, rotated=True, n_tdr = 1000):
    endpoint = True if n % 2 == 1 else False # if n is odd, we want to include the endpoint (preferred, preserves symmetry)
    phi = np.linspace(0, 2*np.pi, n, endpoint=endpoint)
    knot = lambda phi: morton_knot(phi, a=a, p=p)
    knot_points = knot(phi)

    knot_points = torch.tensor(knot_points, dtype=TORCH_DTYPE)
    knot_ref = knot_points.clone()  # will be used as reference for z-scaling

    # Compute the convex hull a first time just to shift the parametrization s.t. node 0 is at the beginning of one of the convex hull lobes
    exterior_indices = ConvexHull(knot_points).vertices
    shift_idx = np.min(exterior_indices)
    knot_points = torch.roll(knot_points, -shift_idx, 0)

    # Recompute the convex hull with the new, shifted parametrization
    exterior_indices = ConvexHull(knot_points).vertices
    knot_ref = knot_points.clone()  # will be used as reference for z-scaling

    a,b,c,z = optimize_TDR_params(knot_points, rotated=rotated)

    stretched_knot = knot_ref.clone()
    stretched_knot[:, 2] = z * knot_ref[:, 2]

    return interpolate_interior(stretched_knot, n_tdr, a, b, exterior_indices, rotated=rotated)

