import sys
sys.path.append("..")
sys.path.append("../ext/torchcubicspline")

import torch
import numpy as np

from torchcubicspline import (
    natural_cubic_spline_coeffs, NaturalCubicSpline, NaturalCubicSplineWithVaryingTs,
)
from src.geometry_src.geom_utils_torch import rotate_about_axis
from src.geometry_src.distances import symmetric_chamfer_distance_squared, points_to_polyline_distance
from src.optimization_src.TDR_projection import get_exterior_interior_indices, split_index_list_in_two_segments
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from src.geometry_src.geom_utils import compute_aligned_tdr_disks

TORCH_DTYPE = torch.float64

# -------------------------------UTILS--------------------------------

def write_polyline_to_obj(file, pts):
        with open(file, 'w') as f:
            for pt in pts:
                f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
            # Write edges
            n_pts = pts.shape[0]
            for i in range(n_pts - 1):
                f.write(f"l {i+1} {i+2}\n")
            f.write(f"l {n_pts} 1\n")

def smooth_values(v):
    '''Smooth the values by minimizing the (soft) max variation of the differences.'''
    return torch.norm(v[1:] - v[:-1], p=2)

def uniform_values(v):
    '''Uniformly distribute the values by minimizing the (soft) max variation from the mean.'''
    return torch.norm(v - torch.mean(v), p=2)

def compute_curvature(pts, closed_curve=True, angles=True):
    if closed_curve:
    # assert closed_curve, "Only closed curves are currently supported"
        assert torch.allclose(pts[0], pts[-1], rtol=1.0e-5), "First and last points must overlap for closed curves"
        tangents = (pts[1:] - pts[:-1]) / torch.linalg.norm(pts[1:] - pts[:-1], dim=1, keepdim=True)
        normals = torch.cross(tangents, torch.roll(tangents, 1, dims=0), dim=1)
        cos_angle = torch.clamp(torch.sum(tangents * torch.roll(tangents, 1, dims=0), dim=1), -1.0, 1.0)
        sin_angle = torch.clamp(torch.sum(normals * normals, dim=1), -1.0, 1.0)
        curvature_angles = torch.atan2(sin_angle, cos_angle)
    else:
        tangents = (pts[1:] - pts[:-1]) / torch.linalg.norm(pts[1:] - pts[:-1], dim=1, keepdim=True)
        normals = torch.cross(tangents[1:], tangents[:-1], dim=1)
        cos_angle = torch.clamp(torch.sum(tangents[1:] * tangents[:-1], dim=1), -1.0, 1.0)
        sin_angle = torch.clamp(torch.sum(normals * normals, dim=1), -1.0, 1.0)
        curvature_angles = torch.atan2(sin_angle, cos_angle)

    if angles:
        return curvature_angles
    else:
        edge_lengths = torch.linalg.norm(pts[1:] - pts[:-1], dim=1)
        l = (edge_lengths[1:] + edge_lengths[:-1]) / 2
        curvature = curvature_angles / l
        return curvature

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

    def interior_pts_within_convex_hull(pts, tdr_hull):
        hull_and_pts = torch.cat((tdr_hull, pts), dim=0).numpy()
        recomputed_hull = ConvexHull(hull_and_pts)
        recomputed_hull_indices = recomputed_hull.vertices
        return np.max(recomputed_hull_indices) <= tdr_hull.shape[0] - 1

def interior_pts_within_convex_hull(pts, tdr_hull):
    hull_and_pts = torch.cat((tdr_hull, pts), dim=0).numpy()
    recomputed_hull = ConvexHull(hull_and_pts)
    recomputed_hull_indices = recomputed_hull.vertices
    return np.max(recomputed_hull_indices) <= tdr_hull.shape[0] - 1

def constrain_pts_on_plane(self, points: torch.Tensor, normal: torch.Tensor, point_on_plane: torch.Tensor) -> torch.Tensor:
    """
    Compute the signed distances of a set of points from a plane.
    
    Args:
        points (torch.Tensor): Tensor of shape (n,3) representing n points.
        normal (torch.Tensor): Tensor of shape (3,) representing the normal vector of the plane.
        point_on_plane (torch.Tensor): Tensor of shape (3,) representing a point on the plane.
    
    Returns:
        torch.Tensor: Tensor of shape (n,) containing signed distances.
    """
    normal = normal / torch.norm(normal)  # Normalize normal vector
    distances = torch.matmul(points - point_on_plane, normal)  # Project onto normal
    return torch.norm(distances, p=2)

# -------------------------------CLASS--------------------------------

class CurveOpti():

    def __init__(self, **kwargs):
        self.default_params()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def default_params(self):
        self.n_tdr_portion_indices = 20  # how many points to take from each TDR, to be attached to the interior points
        
        # PARAM: If needed (e.g. p > 3), increase the number of interior control points here
        self.n_cps_int_per_seg = 41 # Number of control points per segment of the interior part of the curve
        self.n_cps_ext_per_seg = 3 # Number of control points per segment of the exterior part of the curve

        self.w_knot = 1.0   
        self.w_tdr = 1e-4
        self.w_curvature = 1.0
        
        self.max_iter = 250

        self.factor_cps_to_pts = 16 # Factor to convert control points to points

        self.curvature_damping = 0.0

        self.tdr_damping = 2.0

    def init_TDR(self, alpha, beta, gamma, n_tdr):
        
        self.tdr_ell1, self.tdr_ell2 = compute_aligned_tdr_disks(n_tdr, alpha, beta, gamma, rotated=True)

        tdr = torch.cat([self.tdr_ell1, self.tdr_ell2], dim=0)
        tdr_hull_idx = ConvexHull(tdr).vertices
        tdr_hull_idx1 = tdr_hull_idx[tdr_hull_idx < n_tdr]
        tdr_hull_idx2 = tdr_hull_idx[tdr_hull_idx >= n_tdr] - n_tdr
        self.tdr_hull1 = self.tdr_ell1[tdr_hull_idx1]
        self.tdr_hull2 = self.tdr_ell2[tdr_hull_idx2]
        self.tdr_hull = torch.cat([self.tdr_hull1, self.tdr_hull2], dim=0)

        first_last_pt_tdr_hull1 = torch.abs(self.tdr_hull1[[0, -1]])
        first_last_pt_tdr_hull2 = torch.abs(self.tdr_hull2[[0, -1]])
        assert torch.allclose(first_last_pt_tdr_hull1, first_last_pt_tdr_hull2)  # modulo some rotations/reflections, all the extrema of the convex hull should be at the same distance from the origin
        
    def stitch_interior(self, knot):
        exterior_indices, interior_indices = get_exterior_interior_indices(knot)
        exterior_indices1, exterior_indices2 = split_index_list_in_two_segments(exterior_indices)
        print(f"Exterior indices, total and split sizes: {exterior_indices.size} -> ({exterior_indices1.size}, {exterior_indices2.size})")
        interior_indices1, interior_indices2 = split_index_list_in_two_segments(interior_indices)
        print(f"Interior indices, total and split sizes: {interior_indices.size} -> ({interior_indices1.size}, {interior_indices2.size})")

        # Check that connection is in interior points
        n_discontinuities_int = np.count_nonzero(np.diff(interior_indices) - 1)
        n_discontinuities_ext = np.count_nonzero(np.diff(exterior_indices) - 1)
        assert n_discontinuities_int == 1 and n_discontinuities_ext == 1, "The connection is not at the boundary between interior and exterior points."

        # Extract the interior points, sorted irrespective of the current location of the connection:
        # the new connection will be at the interface between interior and exterior.
        knot_interior1 = knot[interior_indices1]
        knot_interior2 = knot[interior_indices2]

        # Assemble the initial knot to be warped, s.t. the connection is at the interface between interior and exterior
        knot_warped_init = torch.cat((
            torch.flip(self.tdr_hull2, dims=[0])[-self.n_tdr_portion_indices:],
            knot_interior1,
            torch.flip(self.tdr_hull1, dims=[0])[:self.n_tdr_portion_indices],
        ), dim=0)

        knot_target = knot_interior1 # knot.clone()
        closed_curve = False

        assert self.n_cps_int_per_seg % 2 == 1, "n_cps_int_per_seg must be odd to guarantee symmetry preservation"
        self.cps_ext_indices1 = np.arange(0, self.n_cps_ext_per_seg)
        cps_int_indices1 = np.arange(self.n_cps_ext_per_seg, self.n_cps_ext_per_seg + self.n_cps_int_per_seg)
        self.cps_ext_indices2 = np.arange(self.n_cps_ext_per_seg + self.n_cps_int_per_seg, 2*self.n_cps_ext_per_seg + self.n_cps_int_per_seg)
        
        # cps_int_indices2 = np.arange(2*self.n_cps_ext_per_seg + self.n_cps_int_per_seg, 2*self.n_cps_ext_per_seg + 2*self.n_cps_int_per_seg)
        self.cps_ref = torch.cat((
            resample_polyline(torch.flip(self.tdr_hull2, dims=[0])[-self.n_tdr_portion_indices:], self.n_cps_ext_per_seg),
            resample_polyline(knot_interior1, self.n_cps_int_per_seg),
            resample_polyline(torch.flip(self.tdr_hull1, dims=[0])[:self.n_tdr_portion_indices], self.n_cps_ext_per_seg),
        ), dim=0)
        
        self.n_cps = self.cps_ref.shape[0]

        self.n_pts = (self.n_cps - 1) * self.factor_cps_to_pts + 1
        self.n_pts_int_per_seg = (self.n_cps_int_per_seg + 1) * self.factor_cps_to_pts - 1
        self.n_pts_ext_per_seg = (self.n_cps_ext_per_seg - 1) * self.factor_cps_to_pts + 1
        
        pts_ext_indices1 = np.arange(0, self.n_pts_ext_per_seg)
        self.pts_int_indices1 = np.arange(self.n_pts_ext_per_seg, self.n_pts_ext_per_seg + self.n_pts_int_per_seg)
        pts_ext_indices2 = np.arange(self.n_pts_ext_per_seg + self.n_pts_int_per_seg, 2*self.n_pts_ext_per_seg + self.n_pts_int_per_seg)
        # pts_int_indices2 = np.arange(2*self.n_pts_ext_per_seg + self.n_pts_int_per_seg, 2*self.n_pts_ext_per_seg + 2*self.n_pts_int_per_seg)

        # 1) Define the spline
        ts_cps = torch.linspace(0.0, 1.0, self.n_cps)
        # spline_coeffs = natural_cubic_spline_coeffs(ts_cps, cps_ref, close_spline=closed_curve)
        spline_coeffs = natural_cubic_spline_coeffs(ts_cps, self.cps_ref)
        spline = NaturalCubicSpline(spline_coeffs)

        # 2) Sample the curve
        ts_pts = torch.linspace(0.0, 1.0, self.n_pts)
        self.pts_ref = spline.evaluate(ts_pts)

    def compute_control_points_from_opt_params(self, params_torch, closed_curve=False):
        if closed_curve:
            cps = params_torch[:3*(self.n_cps-1)].reshape(-1, 3)
            cps = torch.cat([cps, cps[0].reshape(1, 3)], dim=0)
        else:
            cps = params_torch[:3*self.n_cps].reshape(-1, 3)
        return cps

    def compute_curve_from_opt_params(self, params_torch, closed_curve=False):
        cps = self.compute_control_points_from_opt_params(params_torch, closed_curve=closed_curve)

        # Define the spline
        ts_cps = torch.linspace(0.0, 1.0, self.n_cps)
        # spline_coeffs = natural_cubic_spline_coeffs(ts_cps, cps, close_spline=closed_curve)
        spline_coeffs = natural_cubic_spline_coeffs(ts_cps, cps)
        spline = NaturalCubicSpline(spline_coeffs)

        # Sample the curve
        ts_pts = torch.linspace(0.0, 1.0, self.n_pts)
        pts = spline.evaluate(ts_pts)

        return pts
    
    def check_interior(self, pts):
        pts_interior1 = pts[self.pts_int_indices1]
        if interior_pts_within_convex_hull(pts_interior1, self.tdr_hull):
            print("All good, all interior points are still inside the convex hull.")
        else:
            print("WARNING: Interior points outside the convex hull!")

    def optimize_curve_params(self, knot_target, closed_curve=False):
        
        # Initialize curve parameters
        if closed_curve:
            assert torch.allclose(self.cps_ref[0], self.cps_ref[-1]), "First and last points must overlap for closed curves"
            params0 = np.concatenate([
                self.cps_ref[0:-1].reshape(-1,).numpy(),  # drop the last point
            ], axis=0)
        else:
            params0 = np.concatenate([
                self.cps_ref.reshape(-1,).numpy(),  # keep all points
            ], axis=0)

        def obj_and_grad(params):
            params_torch = torch.tensor(params, dtype=TORCH_DTYPE)
            params_torch.requires_grad = True

            pts = self.compute_curve_from_opt_params(params_torch, closed_curve=closed_curve)

            # assert not closed_curve or torch.allclose(cps[0], cps[-1]), "First and last cps must overlap for closed curves"
            assert not closed_curve or torch.allclose(pts[0], pts[-1]), "First and last pts must overlap for closed curves"

            # ------- Compute objective -------
            obj = 0.0

            # Match interior points to knot
            dist_knot = points_to_polyline_distance(pts, knot_target, reduce=None)
            dist_knot = torch.sigmoid(dist_knot)
            obj += self.w_knot * dist_knot.sum()

            # Smooth curve (minimize curvature)
            # obj += self.w_curvature * torch.sum(compute_curvature(pts, closed_curve=closed_curve)**2)
            curvature = compute_curvature(pts, closed_curve=closed_curve)
            # normalize curvature
            # curvature = curvature / torch.max(torch.abs(curvature))
            curvature = torch.sigmoid(curvature)
            n_curvature = len(curvature)
            w = torch.linspace(-1.0, 1.0, n_curvature)**self.curvature_damping
            obj += self.w_curvature * torch.sum(w * curvature**2)
            
            # Attract the interior points to the TDR
            n_pts_int_selected = self.n_pts_int_per_seg // 2
            pts_junction_tdr_1 = self.pts_int_indices1[-n_pts_int_selected:]
            pts_junction_tdr_2 = self.pts_int_indices1[:n_pts_int_selected]
            weights1 = torch.linspace(0.0, 1.0, n_pts_int_selected)**self.tdr_damping  # make the attraction fade in the interior
            weights2 = torch.linspace(1.0, 0.0, n_pts_int_selected)**self.tdr_damping
            dist1 = points_to_polyline_distance(pts[pts_junction_tdr_1], self.tdr_ell1, reduce=None)
            dist2 = points_to_polyline_distance(pts[pts_junction_tdr_2], self.tdr_ell2, reduce=None)
            # dist1 = dist1 / torch.max(dist1)
            # dist2 = dist2 / torch.max(dist2)
            dist1 = torch.sigmoid(dist1)
            dist2 = torch.sigmoid(dist2)
            obj += self.w_tdr * torch.sum(weights1 * dist1)
            obj += self.w_tdr * torch.sum(weights2 * dist2)

            obj.backward()
            # ---------------------------------

            return obj.item(), params_torch.grad.numpy()

        obj_and_grad_scipy = lambda params: obj_and_grad(
            params
        )

        # Set the same value for both lower and upper bounds to fix a parameter
        bounds = [(None, None)] * len(params0)  # Initialize with no bounds

        fixed_indices = []
        fixed_indices += [i for idx in self.cps_ext_indices1 for i in (3*idx, 3*idx + 1, 3*idx + 2)]
        fixed_indices += [i for idx in self.cps_ext_indices2 for i in (3*idx, 3*idx + 1, 3*idx + 2)]
        cps_mid_idx = self.n_cps // 2
        fixed_indices += [i for idx in [cps_mid_idx] for i in (3*idx, 3*idx + 1, 3*idx + 2)]  # pin the central point of interior cps
        # fixed_indices += [i for idx in [cps_mid_idx - 1, cps_mid_idx, cps_mid_idx + 1] for i in (3*idx, 3*idx + 1, 3*idx + 2)]  # pin 3 central points of interior cps
        # fixed_indices += [i for idx in [cps_mid_idx-2, cps_mid_idx-1, cps_mid_idx, cps_mid_idx+1, cps_mid_idx+2] for i in (3*idx, 3*idx + 1, 3*idx + 2)]  # pin the 5 central points of interior cps
        print(f"Fixing {len(fixed_indices)} parameters.")

        for idx in fixed_indices:
            bounds[idx] = (params0[idx], params0[idx])

        torch.autograd.set_detect_anomaly(False)

        global opt_iter
        opt_iter = 0
        history = []

        history.append(params0)

        def callback(params):
            global opt_iter
            opt_iter += 1
            history.append(params)


        result = minimize(
            obj_and_grad_scipy, params0, jac=True, 
            method='L-BFGS-B',
            options={'ftol':1.0e-12, 'gtol':1.0e-6, 'disp': True, 'maxiter': self.max_iter},
            bounds=bounds,
            callback=callback,
        )

        print(f"Optimization took {opt_iter} iterations. Parameters: wcurv: {self.w_curvature}, wtdr: {self.w_tdr}, wknot: {self.w_knot}")

        params_opt = torch.tensor(result.x)
        cps_opt = self.compute_control_points_from_opt_params(params_opt, closed_curve=closed_curve)
        pts_opt = self.compute_curve_from_opt_params(params_opt, closed_curve=closed_curve)
        self.check_interior(pts_opt)

        return cps_opt, pts_opt

    def reconstruct_full_knot(self, pts_opt):
        # pts_int_indices1 = np.arange(n_pts_ext_per_seg - 5, n_pts_ext_per_seg + n_pts_int_per_seg + 1)
        pts_opt_interior = pts_opt[self.pts_int_indices1]
        pts_opt_interior_symmetrized = rotate_about_axis(pts_opt_interior, axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.double), angle=torch.tensor(np.pi))
        knot_full = torch.cat((
            torch.flip(self.tdr_hull2, dims=[0]),
            pts_opt_interior,
            torch.flip(self.tdr_hull1, dims=[0]),
            pts_opt_interior_symmetrized,
        ), dim=0)
        return knot_full
