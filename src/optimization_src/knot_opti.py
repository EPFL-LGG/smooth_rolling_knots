from src.utils import save_knot, load_knot
from src.optimization_src.curve_opti import CurveOpti
from src.optimization_src.TDR_opti import optimize_TDR_params

from src.geometry_src import geom
import numpy as np
import torch
from scipy.spatial import ConvexHull
from src.geometry_src.geom_utils_torch import rotate_about_axis

TORCH_DTYPE = torch.float64
morton_knot = geom.rotated_morton_knot_parametric

from matplotlib import pyplot as plt

# utils
first_nonzero_digit = lambda x: next((i for i in str(x) if i not in ['0', '.']), None)
first_nonzero_digits = lambda x: str(x)[2:]
scientific = lambda x: "{:.2e}".format(x)

class KnotOpti():

    def __init__(self, **kwargs):
        self.default_params()
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def default_params(self):
        self.save_path = "../data/knots/"

    def name(self, x):
        a = self.a
        p = self.p
        n = self.n

        if hasattr(self, "w_tdr") and hasattr(self, "w_curvature"):
            wtdr = self.w_tdr
            wcurv = self.w_curvature
            return f"KNOT_a_{first_nonzero_digits(a)}_p_{p}_n_{n//1000}k_wtdr_{scientific(wtdr)}_wcurv_{scientific(wcurv)}_{x}.obj"
        else:
            return f"KNOT_a_{first_nonzero_digits(a)}_p_{p}_n_{n//1000}k_{x}.obj"

    
    @staticmethod
    def parse_name(name):

        name_split = name.split("_")

        def extract_value(key):
            
            try:
                idx = name_split.index(key)
            except ValueError:
                return None
            
            idx = name_split.index(key)
            return name_split[idx + 1]
        
        a = float(f'0.{extract_value("a")}')
        
        p = extract_value("p")
        n = extract_value("n")
        wtdr = extract_value("wtdr")
        wcurv = extract_value("wcurv")

        p = int(p) if p is not None else None
        n = int(n.replace('k', '')) * 1000 if n is not None else None 
        wtdr = float(wtdr) if wtdr is not None else None
        wcurv = float(wcurv) if wcurv is not None else None

        return KnotOpti(a=a, p=p, n=n, w_tdr=wtdr, w_curvature=wcurv, curve_opt_params={"w_tdr": wtdr, "w_curvature": wcurv}) 

    def save(self, knot, suffix):

        name = self.name(suffix)
        path = self.save_path
        edges = np.array([[i, i+1] for i in range(knot.shape[0]-1)])
        save_knot(knot, edges, name, path)

    def load(self, suffix):
        name = self.name(suffix)
        path = self.save_path
        return load_knot(name, path)
    
    def TDR_optimization(self):
        
        endpoint = True if self.n % 2 == 1 else False # if n is odd, we want to include the endpoint (preferred, preserves symmetry)
        phi = np.linspace(0, 2*np.pi, self.n, endpoint=endpoint)
        knot = lambda phi: morton_knot(phi, a=self.a, p=self.p)
        knot_points = knot(phi)
        knot_points = torch.tensor(knot_points, dtype=TORCH_DTYPE)

        # Compute the convex hull a first time just to shift the parametrization s.t. node 0 is at the beginning of one of the convex hull lobes
        exterior_indices = ConvexHull(knot_points).vertices
        shift_idx = np.min(exterior_indices)
        knot_points = torch.roll(knot_points, -shift_idx, 0)

        # Recompute the convex hull with the new, shifted parametrization
        exterior_indices = ConvexHull(knot_points).vertices
        knot_ref = knot_points.clone()  # will be used as reference for z-scaling

        alpha,beta,gamma,z = optimize_TDR_params(knot_ref, exterior_indices)
 
        stretched_knot = knot_ref.clone()
        stretched_knot[:, 2] = z * knot_ref[:, 2]

        return knot_ref, stretched_knot, alpha, beta, gamma, z

    def curve_optimization(self):

        co = CurveOpti(**self.curve_opt_params)
        co.init_TDR(self.alpha, self.beta, self.gamma, self.n_tdr)

        self.tdr1 = co.tdr_ell1
        self.tdr2 = co.tdr_ell2

        co.stitch_interior(self.stretched_knot)
        
        cps_opt, pts_opt = co.optimize_curve_params(self.stretched_knot)

        # Check that the interior points are still inside the TDR's convex hull
        co.check_interior(pts_opt)
        
        return co.reconstruct_full_knot(pts_opt)

    def optimize(self):

        knot_ref, stretched_knot, alpha, beta, gamma, z = self.TDR_optimization()
        
        self.base_knot = knot_ref
        self.stretched_knot = stretched_knot
        self.alpha = alpha  
        self.beta = beta
        self.gamma = gamma

        self.projected_knot = self.curve_optimization()

        return self.projected_knot







