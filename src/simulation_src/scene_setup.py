import os
import numpy as np
import src.utils as utils
import bpy
import bmesh
from mathutils import Quaternion, Euler, Vector
from mathutils.geometry import distance_point_to_plane
import time

from src.geometry_src.rolliness import optimize_z_stretch_scipy, project_to_tdr
from src.utils import get_optimal_z_stretch
from src.data_utils import load_knot
from src.geometry_src import geom_utils
from src.figures_src.figures import color1, color2, color3, color4

import src.geometry_src.geom as geom

def update(delay: float = 0.1):
    # update geometry depsgraph for the rotation to take effect before moving the knot
    bpy.context.view_layer.update()
    bpy.context.view_layer.depsgraph.update()
    time.sleep(delay)  # wait for the object to be created
    
def remove_object(scene, name: str):
    # Remove existing objects by iterating through all objects
    objects_to_remove = []
    for obj in scene.objects:
        if obj.name.startswith(name):
            objects_to_remove.append(obj)
    
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

class SceneSetup:

    def __init__(self, config : dict):
        utils.reload_modules()
        self.load_config(config)
        self.reset_scene()

    def load_config(self, config: dict):
        """
        Load the configuration from a dictionary.
        """
        self.config = config
        self.blender_config = config.blender_config
        self.simulation_config = config.simulation_config
        self.knot_config = config.knot_config

        # plane properties
        self.plane_angle = self.blender_config.plane_angle
        self.plane_scale = self.blender_config.plane_scale

        # knot properties
        self.knot_location = self.blender_config.knot_location
        
    def add_wind(self):
        """Apply directional force using a wind force field"""
        
        force = self.simulation_config.initial_force
        
        # Create empty object for the force field
        bpy.ops.object.empty_add(type='PLAIN_AXES')
        force_field = bpy.context.active_object
        force_field.name = f"wind"

        bpy.ops.object.forcefield_toggle()
        
        # Configure wind force field
        force_field.field.type = 'WIND'
        force_field.field.strength = 50.0
        force_field.field.flow = 0.0
        force_field.field.apply_to_location = True
        force_field.field.apply_to_rotation = True

        # Wind settings
        force_field.field.use_absorption = False
        force_field.field.distance_max = 0.5
        force_field.field.distance_min = 0.0

        # Set wind direction
        bpy.context.object.rotation_euler[0] = np.pi/2  # Rotate to point in the -Y direction
        force_field.scale[0] = 100.0
        force_field.scale[1] = 2.0

        bpy.context.object.field.falloff_type = 'TUBE'
        bpy.context.object.field.use_max_distance = True

    def reset_scene(self):

        # object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # stop simulation if it is running
        bpy.ops.screen.animation_cancel(restore_frame=True)
 
        # remove all objects
        objs = bpy.data.objects
        for obj in objs:
            bpy.data.objects.remove(obj, do_unlink=True)

        # remove all orphaned data
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        
        # Listener list is persistent through runs of this script, so clear it
        if len(bpy.app.handlers.frame_change_post)>0:
            bpy.app.handlers.frame_change_post.clear()
        
        # add rigid body world and wait for it to be added
        if not bpy.context.scene.rigidbody_world:
            result = bpy.ops.rigidbody.world_add()
            while result != {'FINISHED'}: pass 

        bpy.context.scene.rigidbody_world.point_cache.frame_end = self.simulation_config.frame_end
        
        # viewport transform
        pos = self.blender_config.view_location
        rot = self.blender_config.view_rotation
        rot = Euler(rot, 'XYZ').to_quaternion()
        dist = self.blender_config.view_distance

        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.region_3d.view_location = pos
                        space.region_3d.view_rotation = rot
                        space.region_3d.view_distance = dist
                        break

        # add and place camera at view location, rotation and distance
        camera_location = Vector(pos) + 1.1*dist * (rot @ Vector((0, 0, 1)))
        bpy.ops.object.camera_add(location=camera_location, rotation=rot.to_euler())
        # ortho camera
        bpy.context.object.data.type = 'ORTHO'
        bpy.context.object.data.ortho_scale = 6

        # place light 
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        light = bpy.context.object
        light.data.energy = 5.0  # Set light energy
        light.data.color = (1.0, 1.0, 1.0)
                            
        

        # set force
        # if self.simulation_config.initial_force_length > 0:
            # bpy.context.scene.gravity =  self.simulation_config.initial_force

        
        # add objects
        self.add_plane()

        self.add_wind()

        # set CYCLES
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 5
        bpy.context.scene.cycles.sample_clamp_indirect = 1.5

    # add knot to scene
    def add_knot(self, name : str = "knot", knot_config : dict = None):   

        # deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        
        # delete previous knot if it exists
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

        if knot_config is None:
            knot_config = self.config.knot_config

        if knot_config.type == "torus":
            vertices, edges = geom.torus_knot(knot_config.p, knot_config.q, knot_config.n, knot_config.r1, knot_config.r2)
        elif knot_config.type == "Morton":
            vertices, edges = geom.Morton_knot(knot_config.a, knot_config.n, knot_config.p, knot_config.q)
            if knot_config.stretch == "optimal":
                stretch = get_optimal_z_stretch(knot_config.a)
                if stretch is None:
                    stretch, success = optimize_z_stretch_scipy(vertices)
            else:
                stretch = knot_config.stretch
            vertices = geom.z_stretch(vertices, stretch)
        elif knot_config.type == "projected":
            a = knot_config.a
            n = knot_config.n
            p = knot_config.p
            q = knot_config.q

            vertices, edges = geom.Morton_knot(a=a, n=n, p=p, q=q)
            z_stretch, success = optimize_z_stretch_scipy(vertices)
            if not success:
                raise ValueError("Failed to optimize z-stretch")

            def knot(phi):
                return geom.z_stretch(geom.Morton_knot_parametric(phi, a=a, p=p, q=q), z_stretch)
            
            vertices = project_to_tdr(knot, n=n)
            
        elif knot_config.type == "file":
            vertices = load_knot(knot_config.path, path="")
            n = vertices.shape[0]
            edges = np.linspace(0, n-1, n, dtype=int)
            edges = np.vstack([edges, np.roll(edges, 1)]).T
        else:
            raise ValueError(f"Unknown knot type: {knot_config.type}")


        if self.blender_config.make_uniform:
            vertices = geom_utils.make_uniform(vertices, knot_config.n)
        # add mesh object
        mesh = bpy.data.meshes.new(name=name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        mesh.from_pydata(vertices, edges, [])

        # bevel knot
        if self.blender_config.bevel:
            # convert mesh to curve
            bpy.ops.object.convert(target='CURVE')
            curve = bpy.context.object
            curve.data.bevel_depth = 0.04
            # curve.data.bevel_depth = 0.02
            curve.data.bevel_resolution = 20
            curve.data.fill_mode = 'FULL'

            # convert back to mesh
            bpy.ops.object.convert(target='MESH')
            obj = bpy.context.object
            obj.name = name
            obj.select_set(True)

        if self.blender_config.convex_hull:
            # edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            # mesh to convex hull
            bpy.ops.mesh.convex_hull()
            # object mode
            bpy.ops.object.mode_set(mode='OBJECT')

        # rotate knot by initial angle around initial_knot_rotation_axis
        obj.rotation_mode = 'AXIS_ANGLE'
        obj.rotation_axis_angle = (self.blender_config.initial_knot_rotation_angle, *self.blender_config.initial_knot_rotation_axis)
        obj.rotation_mode = 'XYZ'

        # make rigid body
        bpy.ops.rigidbody.object_add()

        # set mass
        obj.rigid_body.mass = self.blender_config.knot_mass
        obj.rigid_body.angular_damping = 0
        obj.rigid_body.linear_damping = 0

        # set friction of knot
        obj.rigid_body.friction = 1000

        # obj.scale[2] = knot_config.stretch

        # add material
        mat = bpy.data.materials.new(name=name)
        obj.data.materials.append(mat)
        # mat.diffuse_color = (0, 0.25, 0.53, 1)
        match self.blender_config.color:
            case 1:
                mat.diffuse_color = color1
            case 2:
                mat.diffuse_color = color2
            case 3:
                mat.diffuse_color = color3
            case 4:
                mat.diffuse_color = color4

        # move to origin (sometimes its not ... idk why)
        obj.location = self.blender_config.knot_location

        update()

        bpy.context.scene.render.filepath = self.blender_config.render_output
        os.makedirs(self.blender_config.render_output, exist_ok=True)

        # set file format
        bpy.context.scene.render.image_settings.file_format = 'AVI_RAW'


    # add plane to scene
    def add_plane(self): 
        # Add plane
        bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
        plane_handle = bpy.context.object

        # Make plane rigid body
        bpy.ops.rigidbody.object_add()

        # Set plane to passive
        plane_handle.rigid_body.type = 'PASSIVE'

        # Rotate plane
        plane_handle.rotation_euler = Euler((self.plane_angle, 0, 0), 'XYZ')

        # Scale plane
        plane_handle.scale = self.plane_scale

        # Apply modifications to plane mesh in blender
        # plane_handle.data.update_tag()
        # bpy.context.view_layer.depsgraph.update()

        # add material. should render as completely white to fit well in slides
        mat = bpy.data.materials.new(name="PlaneMaterial")
        mat.diffuse_color = (1, 1, 1, 1)  # RGBA
        plane_handle.data.materials.append(mat)

    # place knot on plane
    def place_knot_on_plane(self, knot_name: str = "knot"):

        # get the handles
        plane_handle = bpy.data.objects['Plane']
        bpy.ops.object.select_all(action='DESELECT')
        knot_handle = bpy.data.objects[knot_name]
        bpy.context.view_layer.objects.active = knot_handle
        knot_handle.select_set(True)

        # get the normal vector defining the plane
        plane_normal = plane_handle.matrix_world.to_quaternion() @ plane_handle.data.polygons[0].normal

        # rotate knot by initial angle around plane normal
        plane_normal = plane_normal.normalized()
        angle = self.blender_config.initial_angle
        
        # rotating around the normal vector
        knot_handle.rotation_euler.rotate(Quaternion(plane_normal, angle))
        
        update()

        # iterate over knot vertices to find the one with the smallest signed distance to the plane
        min_distance = np.inf
        for i, vertex in enumerate(knot_handle.data.vertices):
            vertex_global = knot_handle.matrix_world @ vertex.co
            distance = distance_point_to_plane(vertex_global, plane_handle.location, plane_normal)
            if distance < min_distance:
                min_distance = distance

        # move knot mesh by min_distance along the normal vector (min_distance is negative if the knot is below the plane)
        knot_handle.location -= min_distance * plane_normal

        update()

        if knot_handle.location == Vector((0, 0, 0)):
            print("Knot is at origin, something went wrong")
            print(f"min_distance: {min_distance}")
            print(f"plane_normal: {plane_normal}")
            print(f"knot_handle.location: {knot_handle.location}")
            print(f"plane_handle.location: {plane_handle.location}")
            
    # initialize scene
    def init_scene(self, retry: bool = True):

        # add knot
        self.add_knot(name=self.config.knot_config.name)


        update()

        # place knot on plane
        self.place_knot_on_plane(knot_name=self.config.knot_config.name)
        
        update()
        
        # add cm
        if self.blender_config.show_cm:
            self.add_center_of_mass()

        # adjust rigid body world settings
        bpy.context.scene.rigidbody_world.substeps_per_frame = 10
        bpy.context.scene.rigidbody_world.solver_iterations = 1000

        update()

        # set frame end
        bpy.context.scene.frame_end = self.simulation_config.frame_end - 1

        # set current frame to 1
        bpy.context.scene.frame_set(1)

        knot_handle = bpy.data.objects[self.knot_config.name]
        eps = 1e-2
        if abs(knot_handle.location[2]) < eps:
            print("Knot is not properly placed on the plane")
            print(f"knot_handle.location: {knot_handle.location}")
            if retry: 
                print("Retrying")
                self.init_scene(retry=False)
            else:
                raise ValueError("Knot is not properly placed on the plane. Please check the scene setup.")

    def add_torus(self, r1 : float, r2 : float):
        bpy.ops.mesh.primitive_torus_add(major_radius=r1, minor_radius=r2, major_segments=48, minor_segments=12, location=(0, 0, 0))
        torus_handle = bpy.context.object
        torus_handle.name = "torus"

    def add_center_of_mass(self):
        cm_name = self.knot_config.name + "_cm"
        cm_curve_name = self.knot_config.name + "_cm_curve"
        
        # cm_handle = bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(cm_x, cm_y, cm_z))
        # cm_handle = bpy.context.active_object

        # Create center of mass sphere using bmesh (more reliable during rendering)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.05)
        
        # Create mesh and object
        cm_mesh = bpy.data.meshes.new(cm_name)
        bm.to_mesh(cm_mesh)
        bm.free()
        
        cm_handle = bpy.data.objects.new(cm_name, cm_mesh)
        cm_handle.name = cm_name

        bpy.context.collection.objects.link(cm_handle)
        
        # create and add material to center of mass object
        cm_material = bpy.data.materials.new(name="cm_material")
        cm_material.use_nodes = True
        cm_material.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = self.blender_config.cm_color
        cm_handle.data.materials.clear()
        cm_handle.data.materials.append(cm_material)

        # initialize cm_curve object
        cm_curve = bpy.data.curves.new(name=cm_curve_name, type='CURVE')
        cm_curve.dimensions = '3D'
        cm_curve.resolution_u = 2
        cm_curve.resolution_v = 2
        cm_curve_handle = bpy.data.objects.new(cm_curve_name, cm_curve)
        cm_curve_handle.visible_shadow = False

        spline = cm_curve.splines.new('POLY')

        # material
        cm_curve_material = bpy.data.materials.new(name="cm_curve_material")
        cm_curve_material.use_nodes = True
        cm_curve_material.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = self.blender_config.cm_curve_color
        cm_curve.materials.clear()
        cm_curve.materials.append(cm_curve_material)

        # set false visible shadow from cm and curve
        # cm_handle.visible_shadow = False
        # cm_curve_handle.visible_shadow = False

        # Appearance
        cm_curve.bevel_depth = 0.02
        cm_curve.bevel_resolution = 8

        update()

        # CM
        knot_handle = bpy.data.objects[self.knot_config.name]
        _, cm_y, cm_z = knot_handle.location
        cm_x = self.blender_config.knot_location[0]
        cm_handle.location = (cm_x, cm_y, cm_z)
        spline.points.add(0)
        spline.points[0].co = (cm_x, cm_y, cm_z, 1)

        # Add Build modifier for progressive reveal
        build_modifier = cm_curve_handle.modifiers.new(name="Build", type='BUILD')
        build_modifier.frame_start = 1
        build_modifier.frame_duration = 1
        build_modifier.use_reverse = False                # Don't build in reverse
        build_modifier.use_random_order = False          # Build in vertex order
        
        # link curve object to scene
        bpy.context.collection.objects.link(cm_curve_handle)

        update()