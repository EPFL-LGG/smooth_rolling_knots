# FIRST THINGS FIRST: add current path to blender's python path to access project code

import os, sys

# Since blender does not support relative imports, we need to add the path to the sys.path
PATH = os.getcwd()

# check if path exists
if not os.path.exists(PATH):
    print(f"Path {PATH} does not exist. Please update the PATH variable in simulation_main.py to the correct path of the project.")
    sys.exit(1)


sys.path.insert(1, PATH)
sys.path.insert(1, PATH+'/src')
sys.path.insert(1, PATH+"/ext/torchcubicspline")

# ===================================================================================

from src import utils

utils.install_dependencies()

import numpy as np

# force reload of modules (blender keeps cached versions of modules, so changes to the code won't be reflected without a reload)
utils.reload_modules()

from optimization_src.knot_opti import KnotOpti
from src.figures_src.BlenderPlot import BlenderPlot
from src.figures_src.Configs import BlenderConfig, KnotConfig
from src.figures_src.figures import color1, color2, color3, normalize_color  
from src.optimization_src.TDR_projection import get_exterior_interior_indices

def render_base_stretched_projected(i, ko, bp, knot_config, blender_config):
        
        blender_config["renderer"] = "CYCLES"

        a,p,n = ko.a, ko.p, ko.n

        ko.save_path = PATH + "/data/knots/"

        base = ko.load("base")
        stretched = ko.load("stretched")
        projected = ko.load("projected")

        bp.scene_setup(blender_config)
        xyz = [-1.5, 1, -0.8] 
        x, y, z = xyz 
        bp.add_enclosing_planes(*xyz)
        # bp.plot_projection(base, knot_config, plane="xy", pos=z)
        # bp.plot_projection(base, knot_config, plane="yz", pos=x)
        # bp.plot_projection(base, knot_config, plane="xz", pos=y)
        bp.plot_curve(base, knot_config)
        bp.render(render_path(i, a, p, n, "base"))
        
        bp.scene_setup(blender_config)
        bp.add_enclosing_planes(*xyz)
        # bp.plot_projection(base, knot_config, plane="xy", pos=z)
        # bp.plot_projection(base, knot_config, plane="yz", pos=x)
        # bp.plot_projection(base, knot_config, plane="xz", pos=y)
        bp.plot_curve(stretched, knot_config)
        bp.render(render_path(i, a, p, n, "stretched"))

        bp.scene_setup(blender_config)
        # bp.plot_projection(base, knot_config, plane="xy", pos=z)
        # bp.plot_projection(base, knot_config, plane="yz", pos=x)
        # bp.plot_projection(base, knot_config, plane="xz", pos=y)
        bp.add_enclosing_planes(*xyz)
        bp.plot_curve(projected, knot_config)
        bp.render(render_path(i, a, p, n, "projected"))


knot_color = normalize_color((0, 120, 255, 1))
knot_color_transparent = knot_color.copy()
knot_color_transparent[3] = 1
disk_color = normalize_color((255, 153, 2, 1)) 
tdr_color = normalize_color((255, 160, 5, 1))
hull_color = normalize_color((255, 0, 0, 1))
hull_color_transparent = normalize_color((255, 0, 0, 0.7))
proj_color = normalize_color((26, 163, 26, 1))
green = normalize_color((26, 163, 26, 1))
black = normalize_color((0,0,0, 1))
colors = [knot_color, disk_color, proj_color, green, black]

first_nonzero_digit = lambda x: next((i for i in str(x) if i not in ['0', '.']), None)
first_nonzero_digits = lambda x: str(x)[2:]

plots_path = "/data/blender_plots/"
prefix = lambda i: f"figure_{i}_"
filename = lambda a, p, n, x: f"KNOT_a_{first_nonzero_digits(a)}_p_{p}_n_{n//1000}k_{x}.obj"
render_path = lambda i, a, p, n, x: PATH + plots_path + prefix(i) + f"{first_nonzero_digits(a)}_{p}_"+ x + ".png"
data_path = lambda a, p, n, x: PATH + "/data/knots/" + filename(a,p,n,x)

def figure_1():

    # name = "KNOT_a_6_p_7_n_1k_wtdr_3.00e-03_wcurv_1.20e+03_tdr2"
    name = "KNOT_a_9_p_3_n_1k_wtdr_7.00e-02_wcurv_1.00e+03_tdr2"
    # name = "KNOT_a_5_p_7_n_1k_wtdr_4.00e-03_wcurv_5.00e+03_projected"

    ko = KnotOpti.parse_name(name)
    a,p,n = ko.a, ko.p, ko.n   
    ko.save_path = PATH + "/data/knots/"

    base_knot = ko.load("base")
    stretched_knot = ko.load("stretched")
    projected_knot = ko.load("projected")
     
    knot_config = KnotConfig({
        # 'color': knot_color_transparent,
        'color': knot_color,
        'bevel_depth': 0.02,
    })
    # knot_config.color[3] = 0.80

    blender_config = BlenderConfig({
        "view_rotation": [np.radians(64), 0, np.radians(-48)], # camera
        "view_location": [-3.36, -5.54, 2.09], # camera
        "ortho_scale": 9,
        "light_position": [-10, -10, 3],
        "light_rotation": [np.radians(-110), np.radians(160), np.radians(112)],
        # "renderer": "CYCLES"
    })

    bp = BlenderPlot()

    bp.scene_setup(blender_config)

    plane_angle=np.pi/16

    bp.add_plane(rotation=(plane_angle, 0, 0), scale=[1, 10, 1], color=normalize_color((255, 255, 255, 1)))

    knot_config.pos = [-3, 0, 0]

    heights_config = KnotConfig(knot_config.__dict__)
    heights_config["color"] = hull_color
    heights_config["bevel_depth"] = 0.02
    heights_config["closed"] = False

    bp.plot_obj_with_heights(base_knot, knot_config, heights_config, plane_angle=plane_angle)
    
    knot_config.pos = [0, 0, 0]
    heights_config.pos = knot_config.pos
    heights_config["color"] = color2
    heights_config["shadow"] = True
    bp.plot_obj_with_heights(stretched_knot, knot_config, heights_config, plane_angle=plane_angle)

    knot_config.pos = [3, 0, 0]
    heights_config.pos = knot_config.pos
    heights_config["color"] = color3
    bp.plot_obj_with_heights(projected_knot, knot_config, heights_config, plane_angle=plane_angle)

    bp.render(render_path(1, a, p, n, ""))

def figure_2():

    # name = "KNOT_a_6_p_7_n_0k_wtdr_1.00e-05_wcurv_2.00e+02_base"
    # name = "KNOT_a_6_p_7_n_0k_wtdr_5.00e-03_wcurv_5.00e+02_tdr2"
    # name = "KNOT_a_6_p_7_n_0k_wtdr_4.00e-03_wcurv_5.00e+02_tdr2"
    # name = "KNOT_a_6_p_7_n_0k_wtdr_3.00e-03_wcurv_6.00e+02_tdr2"
    # name = "KNOT_a_7_p_3_n_1k_wtdr_1.00e-02_wcurv_1.50e+02_projected"
    # name  = "KNOT_a_7_p_3_n_1k_wtdr_3.00e-02_wcurv_1.00e+03_projected"
    # name  = "KNOT_a_7_p_3_n_10k_wtdr_5.00e-02_wcurv_1.00e+03_tdr2"
    
    name = "KNOT_a_75_p_3_n_1k_stretched" # THIS ONE

    # name = "KNOT_a_75_p_3_n_2k_wtdr_4.00e-02_wcurv_1.50e+04_projected"

    ko = KnotOpti.parse_name(name)
    a,p,n = ko.a, ko.p, ko.n   
    ko.save_path = PATH + "/data/knots/"

    base_knot = ko.load("base")
    stretched_knot = ko.load("stretched")
    projected_knot = ko.load("projected")
    tdr1 = ko.load("tdr2")
    tdr2 = ko.load("tdr1")

    # a = 0.75
    # p = 3
    # n = 1001
    
    # path = lambda x: data_path(a, p, n, x)

    knot_config = KnotConfig({
        'color': knot_color_transparent,
        "convex_hull_points": True,
        "hull_color": hull_color    
    })

    knot_config_transparent = KnotConfig(knot_config.__dict__)
    knot_config_transparent["color"] = knot_color_transparent
    knot_config_transparent.color[3] = 0.90
    knot_config_transparent["hull_color"] = hull_color_transparent
    
    proj_knot_config = KnotConfig(knot_config_transparent.__dict__)
    # proj_knot_config["bevel_depth"] = 0.01
    proj_knot_config["proj"] = True
    
    # to trigger blending (alpha < 1) and fixes the aliasing problem
    # proj_knot_config.color[3] = 0.99
    # proj_knot_config.hull_color[3] = 0.99

    tdr_config = KnotConfig({
        "bevel_depth": 0.02,
        "color": tdr_color,
    })

    blender_config = BlenderConfig({
        "convex_hull": True,
        "view_location": [0.8, -1.49, 0.95], # camera
        "view_rotation": [np.radians(73), 0, np.radians(49)], # camera
        "light_rotation": [np.radians(70), np.radians(15), np.radians(60)],
        "light_strength": 4,
        "ortho_scale": 4.3
    })

    line_config = KnotConfig({
        "bevel_depth": 0.002,
        "color": black,
        "closed": False
    })

    bp = BlenderPlot()

    x = -2

    knot_config_transparent.proj = False

    bp.scene_setup(blender_config)
    bp.plot_curve(base_knot, knot_config_transparent)
    bp.plot_projection(base_knot, proj_knot_config)
    bp.plot_orthogonal_lines(line_config, x)       
    bp.add_enclosing_planes(x,1,-1, planes=False) 
    bp.render(render_path(2, a, p, n, "base_knot"))

    bp.scene_setup(blender_config)
    bp.plot_curve(stretched_knot, knot_config_transparent)
    bp.plot_projection(stretched_knot, proj_knot_config)
    bp.plot_curve(tdr1, tdr_config)
    bp.plot_curve(tdr2, tdr_config)
    bp.plot_orthogonal_lines(line_config, x) 
    bp.add_enclosing_planes(x,1,-1, planes=False)           
    bp.render(render_path(2, a, p, n, "stretched_knot"))

    bp.scene_setup(blender_config)
    bp.plot_curve(projected_knot, knot_config_transparent)
    bp.plot_projection(projected_knot, proj_knot_config)
    bp.plot_curve(tdr1, tdr_config)
    bp.plot_curve(tdr2, tdr_config)
    bp.add_enclosing_planes(x,1,-1, planes=False)
    bp.plot_orthogonal_lines(line_config, x)            
    bp.render(render_path(2, a, p, n, "projected_knot"))

def figure_2_hulls():
    
    name = "KNOT_a_75_p_3_n_1k_stretched" # THIS ONE

    # name = "KNOT_a_75_p_3_n_2k_wtdr_4.00e-02_wcurv_1.50e+04_projected"

    ko = KnotOpti.parse_name(name)
    a,p,n = ko.a, ko.p, ko.n   
    ko.save_path = PATH + "/data/knots/"

    base_knot = ko.load("base")
    stretched_knot = ko.load("stretched")
    projected_knot = ko.load("projected")
    tdr1 = ko.load("tdr2")
    tdr2 = ko.load("tdr1")

    blender_config = BlenderConfig({})
    knot_config = KnotConfig({})
    
    # knot convex hull
    blender_config.view_rotation = [np.radians(57.5), 0, 0]
    blender_config.view_location[0] = 0
    blender_config.ortho_scale = 3.2
    blender_config.light_rotation = [np.radians(86), np.radians(-50), np.radians(-50)]
    blender_config.light_location = [-2.36, -1.44, 0.25]

    knot_config.convex_hull_points = False
    knot_hull_config = KnotConfig(knot_config.__dict__)
    knot_hull_config.color[3] = 0.5

    bp.scene_setup(blender_config)
    bp.plot_curve(stretched_knot, knot_config)
    bp.plot_hull(stretched_knot, knot_hull_config)
    # bp.render(render_path(2, a, p, n, "knot_hull"))
    
    # tdr convex hull
    tdr_config.bevel_depth = 0.04
    tdr_hull_config = KnotConfig(tdr_config.__dict__)
    tdr_hull_config.color[3] = 0.5
    bp.scene_setup(blender_config)
    bp.plot_curve(tdr1, tdr_config)
    bp.plot_curve(tdr2, tdr_config)
    bp.plot_hull(np.vstack([tdr1, tdr2]), tdr_hull_config)

    cm1 = tdr1.mean(axis=0)
    cm2 = tdr2.mean(axis=0)
    
    bp.plot_sphere(cm1, radius=0.02)
    bp.plot_sphere(cm2, radius=0.02)

    # bp.render(render_path(2, a, p, n, "tdr_hull"))

def figure_2_smooth():
    # smoothness vs. knot
    smooth = "KNOT_a_75_p_3_n_0k_wtdr_3.00e+00_wcurv_0.00e+00_tdr2"
    rough = "KNOT_a_75_p_3_n_0k_wtdr_0.00e+00_wcurv_0.00e+00_tdr2"

    smooth_ko = KnotOpti.parse_name(smooth)
    rough_ko = KnotOpti.parse_name(rough)

    def load_knots(ko):
         ko.save_path = PATH + "/data/knots/"
         return ko.load("base"), ko.load("stretched"), ko.load("projected"), ko.load("tdr1"), ko.load("tdr2")
    
    smooth_base, smooth_stretched, smooth_projected, smooth_tdr1, smooth_tdr2 = load_knots(smooth_ko)
    rough_base, rough_stretched, rough_projected, rough_tdr1, rough_tdr2 = load_knots(rough_ko)

    bp = BlenderPlot()

    knot_config = KnotConfig({})

    blender_config = BlenderConfig({
        "view_location": [-3.24, -2.76, 1.45], # camera
        "view_rotation": [np.radians(74), np.radians(-26), np.radians(-46)], # camera
        "light_position": [-1.28116, -1.42052, 1.08554],
        "light_rotation": [np.radians(52.0127), np.radians(-14.2718), np.radians(-26.8294)],
        # "light_strength": 4,
        "ortho_scale": 0.95
    })


    _, interior = get_exterior_interior_indices(smooth_projected)
    N = interior.shape[0]
    smooth_projected = smooth_projected[interior][N//3-10:N//2+1]

    _, interior = get_exterior_interior_indices(rough_projected)
    N = interior.shape[0]
    rough_projected = rough_projected[interior][N//3:N//2+1]

    tdr = np.vstack([smooth_tdr1, smooth_tdr2])
    exterior, _ = get_exterior_interior_indices(tdr)
    N = exterior.shape[0]
    tdr = tdr[exterior][N//3:N//2]

    bp.scene_setup(blender_config)

    # bp.add_plane(location=[-0.02, -0.26, 0], rotation=[np.radians(48), 0, np.radians(-59)])
    
    knot_config.closed = False

    knot_config.bevel_depth = 0.01
    knot_config.color = color3
    bp.plot_curve(smooth_stretched[N//4:N//2-50], knot_config)
    
    knot_config.color = color2
    knot_config.bevel_depth = 0.01
    N = smooth_tdr1.shape[0]
    bp.plot_curve(smooth_tdr1[-N//4:], knot_config)

    
    knot_config.bevel_depth = 0.02
    knot_config.color = color3
    knot_config.color[3] = 0.9
    bp.plot_curve(rough_projected, knot_config)

    knot_config.color = knot_color
    knot_config.color[3] = 0.9
    bp.plot_curve(smooth_projected, knot_config)
    
    knot_config.bevel_depth = 0.02
    knot_config.color = hull_color
    knot_config.color[3] = 0.9
    knot_config.closed = False
    bp.plot_curve(tdr, knot_config)
    


    bp.render(render_path(2, smooth_ko.a, smooth_ko.p, smooth_ko.n, "smooth_vs_rough"))

def figure_3():

    knot_config = KnotConfig({})
    blender_config = BlenderConfig({})

    # blender_config["view_location"] = [1.3, -2, 1.25] # camera
    pos = [1.52 , -1.52, 1.1]
    blender_config["view_location"] = pos # camera
    blender_config["view_rotation"] = [np.radians(66), 0, np.radians(45)] # camera
    blender_config["ortho_scale"] = 4

    light_position = [0, -1, 0.5]
    blender_config["light_position"] = light_position
    blender_config["light_rotation"] = np.radians([64, 0, 0])
    
    blender_config["ortho_scale"] = 4.5

    knot_config["bevel_depth"] = 0.03

    bp = BlenderPlot()

    knot_config.color = knot_color
    knot_config.color[:3] /= 11

    render = lambda name: render_base_stretched_projected(3, KnotOpti.parse_name(name), bp, knot_config, blender_config)
    
    render("KNOT_a_3_p_3_n_2k_wtdr_1.00e-02_wcurv_1.00e+02_projected")
    render("KNOT_a_5_p_3_n_1k_wtdr_7.00e-02_wcurv_1.00e+03_projected")
    render("KNOT_a_9_p_3_n_1k_wtdr_7.00e-02_wcurv_1.00e+03_tdr2")
    render("KNOT_a_5_p_5_n_1k_wtdr_1.00e+01_wcurv_1.00e+05_tdr2")
    render("KNOT_a_5_p_7_n_2k_wtdr_1.00e+01_wcurv_2.70e+05_tdr2")
    render("KNOT_a_5_p_9_n_3k_wtdr_1.00e+01_wcurv_4.50e+03_tdr2")

if __name__ == "<run_path>":

    # figure_1()
    # figure_2()
    # figure_2_smooth()
    figure_3()





