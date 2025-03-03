from figures_src.Configs import BlenderConfig, Config, KnotConfig
import bpy
from mathutils import Vector, Euler
from scipy.spatial import ConvexHull
from src.optimization_src.TDR_projection import split_index_list_in_two_segments, get_exterior_interior_indices

from visualization_src.data_utils import load_knot
from geometry_src.rolliness import rolliness, rolling_trajectory
import numpy as np
from src.geometry_src.geom_utils import  *
from scipy.spatial.transform import Rotation as R

class BlenderPlot():
    def __init__(self) -> None:
        pass

    def scene_setup(self, config:BlenderConfig):

        # factory settings
        # bpy.ops.wm.read_factory_settings(use_empty=True)

        # object mode   
        bpy.ops.object.mode_set(mode='OBJECT')

        # remove all 
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # remove all objects
        objs = bpy.data.objects
        for obj in objs:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        bpy.context.scene.render.engine = config.renderer
        
        # viewport transform
        pos = config.view_location
        rot = config.view_rotation
        rot = Euler(rot, 'XYZ').to_quaternion()
        dist = config.view_distance
        
        # add and place camera at view location, rotation and distance
        # camera_location = Vector(pos) + 1.1* dist * (rot @ Vector((0, 0, 1)))
        camera_location = Vector(pos) 
        bpy.ops.object.camera_add(location=camera_location, rotation=rot.to_euler())

        # set camera to orthographic
        camera = bpy.context.object
        camera.data.type = 'ORTHO'
        camera.data.ortho_scale = config.ortho_scale


        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        # space.region_3d.view_location = pos
                        # space.region_3d.view_rotation = rot
                        # space.region_3d.view_distance = dist
                        space.region_3d.view_perspective = 'CAMERA'
                        break


        # toggle camera view


        # place light 
        bpy.ops.object.light_add(type='SUN', radius=10.0, location=config.light_position, rotation=Euler(config.light_rotation))
        light = bpy.context.object
        light.data.energy = config.light_energy

        
        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

    # Override this method in subclasses to plot
    def _plot(self, **kwargs):
        raise NotImplementedError("Subclasses should implement _plot() private method.")

    def plot(self, **kwargs):

        self._plot(**kwargs)

        # update geometry
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()

    def render(self, path):

        
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()

        if bpy.context.scene.render.engine == 'CYCLES':
            bpy.context.scene.cycles.samples = 5


        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = path
        bpy.context.scene.render.film_transparent = True
        
        bpy.ops.render.render(write_still = True)
        print(f"Rendered image to {path}")

    def plot_obj_curve(self, path, config):
        points = load_knot(path, path="")
        self.plot_curve(points, config)

    def plot_obj_with_heights(self, knot, config, heights_config, plane_angle):

        points = knot
        rho, heights = rolliness(points)

        # rotate heights to get minimum first
        min_idx = np.argmin(heights)
        heights = np.roll(heights, -min_idx)

        
        traj = rolling_trajectory(points)
        traj_length = np.sum(np.linalg.norm(traj[1:, :] - traj[:-1, :], axis=1))/2
        # traj_length = 4
        traj = np.linspace(0, traj_length, len(points))

        x = -1 * np.linspace(0, traj_length, len(heights))[::-1]
        line_eq = lambda x: x * np.sin(-plane_angle)
        plane_line = line_eq(x)
        new_heights = np.array(heights) - plane_line
        height_points = np.vstack([np.zeros_like(x), x, new_heights]).T
        # height_points = np.vstack([np.zeros_like(x), x, heights]).T
        self.plot_curve(height_points, heights_config)

        heights_config_copy = Config(heights_config.__dict__)
        heights_config_copy["closed"] = True
        heights_config_copy["fill"] = True
        heights_config_copy["color"] = (0, 0, 0, 1)
        heights_config_copy["bevel_depth"] = 0.005
        heights_config_copy["shadow"] = False
        plane_line_points = np.vstack([np.zeros_like(x), x, -plane_line]).T
        height_and_plane_points = np.vstack([height_points, plane_line_points[::-1]])
        self.plot_curve(height_and_plane_points, heights_config_copy)
        
        config_copy = Config(config.__dict__)
        config_copy.pos = height_points[0, :]
        config_copy.pos[0] = config.pos[0]
        config_copy.initial_angle = plane_angle
        config_copy.rotation_axis = [1, 0, 0]
        self.plot_curve(points, config_copy)

        config_copy.bevel_depth = 0.03
        config_copy.color = (0, 0, 0, 1)

        center_of_mass = np.mean(points, axis=0)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=center_of_mass+config_copy.pos)
        sphere = bpy.context.object
        sphere.name = "center_of_mass"
        sphere.data.materials.append(bpy.data.materials.new(name="SphereMaterial"))
        sphere.data.materials[0].diffuse_color = (0, 0, 0, 1)

        #update
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()

        # deslect all objects
        bpy.ops.object.select_all(action='DESELECT')

        #plane normal line
        # line = np.array([[0, 0, 0], [0, 0, -new_heights[-1]]])
        # config_copy.bevel_depth = 0.005
        # self.plot_curve(line, config_copy)
        
        # heights_config_copy = Config(heights_config.__dict__)
        
        # N = len(height_points)
        # skip = N//20

        # for h in height_points[::skip]:
        #     config_copy.pos = h
        #     config_copy.pos[0] = config.pos[0]
        #     self.plot_curve(line, config_copy)

        #update
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()

    def plot_obj_projection(self, path, config):
        points = load_knot(path, path="")
        self.plot_projection(points, config)

    def plot_projection(self, points, config, plane="yz", pos=-2):
        config = Config(config.__dict__)
        config.proj = plane
        config.proj_dist = pos
        self.plot_curve(points, config)

    def plot_orthogonal_lines(self, config, x0):
        yz = np.linspace(-0.75, 0.75, 25)
        x = np.ones_like(yz) * x0
        line_1 = np.vstack([x, yz, yz]).T
        line_2 = np.vstack([x, yz, -yz]).T
        self.plot_curve(line_1, config)
        self.plot_curve(line_2, config)

        # orth symbol
        # config_copy = Config(config.__dict__)
        # config_copy.bevel_depth = 0.01
        # size = 0.1  
        # orth_1 = np.array([[x0, -size, -size], [x0, -2*size, 0]])
        # orth_2 = np.array([[x0, -size, size], [x0, -2*size, 0]])
        # self.plot_curve(orth_1, config_copy)
        # self.plot_curve(orth_2, config_copy)
        
    def plot_curve(self, points, config):
        
        n = points.shape[0]

        if config.convex_hull_points: 
            hull_config = Config(config.__dict__)
            hull_config["color"] = config.hull_color
            hull_config["convex_hull_points"] = False
            hull_config["closed"] = False

            exterior, interior = get_exterior_interior_indices(points)

            ext1, ext2 = split_index_list_in_two_segments(exterior)
            # int1, int2 = get_exterior_segments(interior)
            int1 = [*range(ext2[-1], n), *range(0, ext1[0]+1)]
            int2 = list(range(ext1[-1], ext2[0]+1))

            config = Config(config.__dict__)
            config.convex_hull_points = False
            config.closed = False

            self.plot_curve(points[ext1], hull_config)
            self.plot_curve(points[ext2], hull_config)

            if not config.proj:
                self.plot_curve(points[int1], config)
                self.plot_curve(points[int2], config)

        else: 

            if config.proj:
                points = np.array(points)
                if config.proj == "xy":
                    points[:, 2] = config.proj_dist
                elif config.proj == "yz":
                    points[:, 0] = config.proj_dist
                elif config.proj == "xz":
                    points[:, 1] = config.proj_dist
                
            # delete previous knot if it exists
            # deselect all objects
            bpy.ops.object.select_all(action='DESELECT')
            # add mesh object
            name = "knot"
            mesh = bpy.data.meshes.new(name=name)
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.collection.objects.link(obj)
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            edges = [(i, i+1) for i in range(n-1)]
            if config.closed:
                edges.append((n-1, 0))
            mesh.from_pydata(points, edges, [])

            # bevel knot
            # convert mesh to curve
            bpy.ops.object.convert(target='CURVE')
            curve = bpy.context.object
            curve.data.bevel_depth = config.bevel_depth
            curve.data.bevel_resolution = config.bevel_resolution
            curve.data.fill_mode = 'FULL'

            # update
            bpy.context.view_layer.update()
            bpy.context.view_layer.depsgraph.update()

            # convert back to mesh
            bpy.ops.object.convert(target='MESH')
            obj = bpy.context.object
            obj.name = name
            obj.select_set(True)

            # rotate knot by initial angle around initial_knot_rotation_axis
            obj.rotation_mode = 'AXIS_ANGLE'
            obj.rotation_axis_angle = (config.initial_angle, *config.rotation_axis)
            obj.rotation_mode = 'XYZ'

            # update
            bpy.context.view_layer.update()
            bpy.context.view_layer.depsgraph.update()

            # add material
            mat = bpy.data.materials.new(name=name)
            obj.data.materials.append(mat)
            mat.diffuse_color = config.color


            # Add Principled BSDF node and assign the correct alpha
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs['Base Color'].default_value = config.color
                bsdf.inputs['Alpha'].default_value = config.color[3]
                bsdf.inputs['Roughness'].default_value = 0.95

            # Set blend mode to alpha blend
            if config.color[3] < 1:
                mat.blend_method = 'BLEND'
            else:
                mat.blend_method = 'OPAQUE'

            # Set backface culling
            mat.show_transparent_back = False 

            # move to origin (sometimes its not ... idk why)
            obj.location = config.pos

            #update 
            bpy.context.view_layer.update()
            bpy.context.view_layer.depsgraph.update()
    
    # add plane to scene
    def add_plane(self, scale=[50, 50, 1], color=(1, 1, 1, 1), location=(0, 0, 0), rotation=(0.032724923474893676, 0, 0)): 

        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # Add plane
        bpy.ops.mesh.primitive_plane_add(size=10, location=location)
        plane_handle = bpy.context.object

        # Rotate plane
        plane_handle.rotation_euler = Euler(rotation, 'XYZ')

        # Scale plane
        plane_handle.scale = scale

        #update 
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()

        # add material
        mat = bpy.data.materials.new(name="plane")
        plane_handle.data.materials.append(mat)
        mat.diffuse_color = color

        # Add Principled BSDF node and assign the correct alpha
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Alpha'].default_value = color[3]

        # Set blend mode to alpha blend
        mat.blend_method = 'OPAQUE'

        # Set backface culling
        mat.show_transparent_back = False 

        # Apply modifications to plane mesh in blender
        # plane_handle.data.update_tag()
        # bpy.context.view_layer.depsgraph.update()

    def plot_sphere(self, location, radius=0.05):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
        sphere = bpy.context.object
        sphere.name = "center_of_mass"
        sphere.data.materials.append(bpy.data.materials.new(name="SphereMaterial"))
        sphere.data.materials[0].diffuse_color = (0, 0, 0, 1)

        #update
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()


    def plot_hull(self, points, config):

        bpy.ops.object.select_all(action='DESELECT')
        # add mesh object
        name = "knot"
        mesh = bpy.data.meshes.new(name=name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        mesh.from_pydata(points, [], [])

        # rotate knot by initial angle around initial_knot_rotation_axis
        obj.rotation_mode = 'AXIS_ANGLE'
        obj.rotation_axis_angle = (config.initial_angle, *config.rotation_axis)
        obj.rotation_mode = 'XYZ'

        # edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        # mesh to convex hull
        bpy.ops.mesh.convex_hull()
        # object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # update
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()

        # add material
        mat = bpy.data.materials.new(name=name)
        obj.data.materials.append(mat)
        mat.diffuse_color = config.color

        # Add Principled BSDF node and assign the correct alpha
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = config.color
            bsdf.inputs['Alpha'].default_value = config.color[3]

        # Set blend mode to alpha blend
        if config.color[3] < 1:
            mat.blend_method = 'BLEND'
        else:
            mat.blend_method = 'OPAQUE'

        # Set backface culling
        mat.show_transparent_back = False 

        # move to origin (sometimes its not ... idk why)
        obj.location = config.pos

        #update 
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()


    # add plane to scene
    def add_enclosing_planes(self, x, y, z, planes = True): 

        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # Add lines for axes
        xline = np.array([[x, y, z], [1, y, z]])
        yline = np.array([[x, y, z], [x, -1, z]])
        zline = np.array([[x, y, z], [x, y, 1]])

        config = KnotConfig({"color": (0, 0, 0, 1), "bevel_depth": 0.001, "pos": [0, 0, 0]})

        self.plot_curve(xline, config)
        self.plot_curve(yline, config)
        self.plot_curve(zline, config)

        if planes: 
            # Add plane
            xplanepos = np.array([x, 0, 0])
            yplanepos = np.array([0, y, 0])
            zplanepos = np.array([0, 0, z])
            xplanerot = np.array([0, np.pi/2, 0])
            yplanerot = np.array([np.pi/2, 0, 0])
            zplanerot = np.array([0, 0, np.pi/2])

            # Add plane

            # new collection for planes
            if not bpy.data.collections.get("Planes"):
                bpy.ops.collection.create(name="Planes")
            
            # link to scene
            if not "Planes" in bpy.context.scene.collection.children:
                bpy.context.scene.collection.children.link(bpy.data.collections["Planes"])  

            plane_scale = [2, 2, 2]
            
            bpy.ops.mesh.primitive_plane_add(size=10, location=xplanepos, rotation=xplanerot)
            plane_handle = bpy.context.object
            plane_handle.scale = plane_scale

            # Add plane to collection
            bpy.ops.object.collection_link(collection='Planes')

            bpy.ops.mesh.primitive_plane_add(size=10, location=yplanepos, rotation=yplanerot)
            plane_handle = bpy.context.object
            plane_handle.scale = plane_scale


            # Add plane to collection
            bpy.ops.object.collection_link(collection='Planes')

            bpy.ops.mesh.primitive_plane_add(size=10, location=zplanepos, rotation=zplanerot)
            plane_handle = bpy.context.object
            plane_handle.scale = plane_scale


            # Add plane to collection
            bpy.ops.object.collection_link(collection='Planes')

            plane_collection = bpy.data.collections["Planes"]


            # find sun light and delete it
            for obj in bpy.data.objects:
                if obj.type == 'LIGHT':
                    bpy.data.objects.remove(obj, do_unlink=True)

            # add area light
            delta = 0.2
            light_strength = 30
            bpy.ops.object.light_add(type='SUN', location=(-np.sign(x)*(np.abs(x)-delta), 0, 0), rotation=xplanerot)
            obj = bpy.context.object
            obj.data.energy = light_strength
            obj.data.cycles.max_bounces = 1

            # obj.scale = [3, 3, 3]
            bpy.ops.object.light_add(type='SUN', location=(0, -np.sign(y)*(np.abs(y)-delta), 0), rotation=yplanerot)
            obj = bpy.context.object
            obj.data.energy = light_strength
            obj.data.cycles.max_bounces = 1

            # obj.scale = [3, 3, 3]
            bpy.ops.object.light_add(type='SUN', location=(0, 0, -np.sign(z)*(np.abs(z)-delta)), rotation=zplanerot)
            obj = bpy.context.object
            obj.data.energy = light_strength
            obj.data.cycles.max_bounces = 1
            
            # obj.scale = [3, 3, 3]
        
        #update 
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()

        # Apply modifications to plane mesh in blender
        # plane_handle.data.update_tag()
        # bpy.context.view_layer.depsgraph.update()
