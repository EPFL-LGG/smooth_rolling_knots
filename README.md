# Smooth Rolling Knots
Code related to the Smooth-Rolling Knots project.

![rolling knot gif](./data/smooth_rolling_knots_demo.gif)

`notebooks/generate_objs.ipynb`: generates the results presented in the paper.

`blender_plots.py`: scripts blender to render the knots and populates `data/blender_plots`

`notebooks/figures.ipynb`: assembles the figures from the paper

# Optimization overview

$$
K^* \, = \, \argmin_{\mathbf{q}} \; w_{\text{knot}} E_{\text{knot}}(\mathbf{q}) 
+  w_{\text{curvature}}E_{\text{curvature}}(\mathbf{q}) 
+ w_{\text{TDR}}E_{\text{TDR}}(\mathbf{q}) 
$$

where 
- $E_{\text{knot}}$ is a deformation knot energy computed by using a point-to-polyline distance.
- $E_{\text{TDR}}$ is a distance penalty from the smooth-rolling TDRs.
- $E_{\text{curvature}}$ is a curvature energy that penalizes curvature in the interior.
- $w_{\text{knot}}$, $w_{\text{curvature}}$, and $w_{\text{TDR}}$ are user defined weights that control the importance of each term (with fixed $w_{\text{knot}}=1$). 

See the optimiziation implementation in `src/optimization_src/curve_opti.py:optimize_curve_params` .

# Usage

```
# Default parameters:
curve_opt_params = {
    'w_tdr': 1,
    'w_curvature': 1,
    'curvature_cps': 1,
    'tdr_damping': 1,
    'n_cps_int_per_seg': 7, 
    'factor_cps_to_pts': 16,
    'max_iter': 400,
}
```

Parameter description:
- `w_tdr` and `w_curvature`: weights for the TDR and curvature energy terms.
- `curvature_cps`: is a depth factor for which to minimize curvature around the junction points between the TDR and the interior of the knot. It is a multiplier to the `factor_cps_to_pts` parameter.
- `tdr_damping`: damping factors for the curvature and TDR energy terms, dissipating the energies in the knot's center. A value of 0 means no damping, 1 means linear damping, 2 means quadratic damping, etc.
- `n_cps_int_per_seg`: number of control points per segment in the knot's interior polyline.
- `factor_cps_to_pts`: number of points per control point in the knot's polyline.
- `max_iter`: maximum number of iterations for the optimization.

The following recipe has been found to work well when dealing with more complex knots: 
1. Apply the method with $w_{\text{curvature}}=0$ and $w_{\text{TDR}}=0$ to find the `n_cps_int_per_seg` and `factor_cps_to_pts` that give a good polyline representation of the knot.
2. Use the `w_tdr` and `tdr_damping` parameters to find the good TDR vs. knot interior preservation balance.
3. Apply the `w_curvature`and `curvature_damping` parameters to smooth things out. Curvature minimization isn't local, since we're dealing with polylines. If the junction curvature minimization affects the interior of the knot too much, increase the `n_cps_int_per_seg` parameter.

**Don't forget to check if $\rho=0$! There is no guarantee for this preservation.**
