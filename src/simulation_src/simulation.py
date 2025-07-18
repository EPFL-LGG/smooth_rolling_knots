import bpy
import bmesh
import numpy as np
from typing import Tuple
from mathutils.geometry import distance_point_to_plane
from mathutils import Vector

import os
import src.utils as utils
from src.simulation_src.scene_setup import update, remove_object

class Simulation:
    """
    Simulation class that manages the simulation data. 
    """

    def __init__(self, config):
        utils.reload_modules()
        self.blender_config = config.blender_config
        self.knot_config = config.knot_config
        self.simulation_config = config.simulation_config
        self.data_config = config.data_config
        self.online_path = self.data_config.path + "/online.csv"
        self.offline_path = self.data_config.path + "/offline.csv"
        self.num_frames = self.simulation_config.frame_end
        self.plane_normal = None

        # Column names for time series file
        self.columns = ["frame",
                        # center of mass in plane coordinates
                        "center_of_mass_X",
                        "center_of_mass_Y",
                        "center_of_mass_Z" # distance from plane / height
                        ]
        contact_point_columns = []
        for i in range(3):
            contact_point_columns.append(f"contact_point_{i}_X")
            contact_point_columns.append(f"contact_point_{i}_Y")

        self.columns.extend(contact_point_columns)

        self.columns_string = ",".join(self.columns) + "\n"

    def reset_simulation(self):

        self.end_simulation = False
        
        self.data = {}
        
        for column in self.columns:
            self.data[column] = np.zeros(self.num_frames)
        
        # create online time series file
        if self.data_config.online:
            if not os.path.exists(self.data_config.data_path):
                os.makedirs(self.data_config.data_path)
            if not os.path.exists(self.data_config.path):
                os.makedirs(self.data_config.path)
            with open(self.online_path, 'w') as file:
                file.write(self.columns_string)

        if not self.plane_normal:
            plane_handle = bpy.data.objects['Plane']
            self.plane_normal = plane_handle.matrix_world.to_quaternion() @ plane_handle.data.polygons[0].normal
            self.side_axis = self.plane_normal.cross(Vector((0, 1, 0)))
            self.side_axis.normalize()
            self.slope_axis = self.plane_normal.cross(self.side_axis)
            self.slope_axis.normalize()

    def dump_offline(self, frame, scene):
        with open(self.offline_path, 'w') as file:
            file.write(self.columns_string)
            for i in range(frame):
                data = self._make_data_string(i+1)
                file.write(data)

    def compute_global_center_of_mass(self, knot_handle) -> Tuple[float, float, float]:
        """
        Compute the global center of mass of the knot in the world coordinates.
        """
        v = knot_handle.matrix_world.translation
        return v.x, v.y, v.z

    def compute_center_of_mass(self, knot_handle, plane_handle) -> Tuple[float, float, float]:
        """
        Compute the center of mass of the knot in the plane coordinates.
        """
        v = knot_handle.matrix_world.translation - plane_handle.location
        cm_x = v.dot(self.side_axis)
        cm_y = v.dot(self.slope_axis)

        cm_z = distance_point_to_plane(knot_handle.matrix_world.translation, plane_handle.location, self.plane_normal)
        
        return cm_x, cm_y, cm_z

    def shading_attempt(self, knot_handle, scene, contact_points, plane_handle):
        # select plane
        bpy.context.view_layer.objects.active = plane_handle
        plane_handle.select_set(True)

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)  # Unwrap UVs
        bpy.ops.object.mode_set(mode='OBJECT')

        plane_material = bpy.data.materials.new(name="PlaneMaterial")
        # clear materials
        plane_handle.data.materials.clear()
        plane_handle.data.materials.append(plane_material)
        
        image_size = 1024  # Set the size of the texture image
        texture_image = bpy.data.images.new("ContactPointsImage", width=image_size, height=image_size)

        # center of the image is the center of the plane
        center = (image_size // 2, image_size // 2)

        # draw contact points
        for point in contact_points:
            x = int(center[0] + point[0] * image_size / 50)
            y = int(center[1] + point[1] * image_size / 50)
            texture_image.pixels[y * image_size + x] = 1

        plane_material.use_nodes = True
        material_output = plane_material.node_tree.nodes.get('Material Output')
        principled_bsdf = plane_material.node_tree.nodes.get('Principled BSDF')

        texture_node = plane_material.node_tree.nodes.new('ShaderNodeTexImage')
        texture_node.image = texture_image

        plane_material.node_tree.links.new(texture_node.outputs[0], principled_bsdf.inputs[0])
    
    def compute_contact_points(self, knot_handle, plane_handle, scene) -> np.array:
        vertices = knot_handle.data.vertices
        eps = 1e-3 # distance from plane for collision
        contact_points = []
        plane_normal = plane_handle.matrix_world.to_quaternion() @ plane_handle.data.polygons[0].normal
        side_axis = plane_normal.cross(Vector((0, 1, 0)))
        side_axis.normalize()
        slope_axis = plane_normal.cross(side_axis)
        slope_axis.normalize()
        last_vertex = None
        max_neighbor_distance = 0
        for i, vertex in enumerate(vertices):
            # distance from vertex to plane
            vertex_global = knot_handle.matrix_world @ vertex.co
            
            # compute distance between neighboring vertices (for cluster merging later)
            if last_vertex is not None and (vertex_global - last_vertex).length > max_neighbor_distance:
                max_neighbor_distance = (vertex_global - last_vertex).length
            last_vertex = vertex_global
            
            height = distance_point_to_plane(vertex_global, plane_handle.location, plane_normal)
            if height < eps:
                # express in plane coordinates
                v = vertex_global - plane_handle.location
                x = v.dot(side_axis)
                y = v.dot(slope_axis)
                contact_points.append((x, y))

        if len(contact_points) == 0:
            print("No contact points!")
            return np.zeros((3, 2))

        # neighborhood distance for merging clusters
        eps2 = max_neighbor_distance + eps
        
        cluster = set()
        contact_point_clusters = [cluster]
        last_cp = None
        for cp in contact_points:
            if len(cluster) == 0 or np.linalg.norm(np.array(cp) - np.array(last_cp)) < eps2:
                cluster.add(cp)
            else:
                cluster = set()
                contact_point_clusters.append(cluster)
                cluster.add(cp)
            last_cp = cp

        # compare clusters and merge if they are close
        merged_clusters = []
        for i, cluster1 in enumerate(contact_point_clusters):
            for j, cluster2 in enumerate(contact_point_clusters[i+1:]):
                for cp1 in cluster1:
                    for cp2 in cluster2:
                        if np.linalg.norm(np.array(cp1) - np.array(cp2)) < eps2:
                            contact_point_clusters[i] = contact_point_clusters[i].union(contact_point_clusters[j])
                            merged_clusters.append(j)

        # remove merged clusters
        contact_point_clusters = [contact_point_clusters[i] for i in range(len(contact_point_clusters)) if i not in merged_clusters]

        # contact points are the centers of mass of the clusters
        contact_points = [np.mean(list(cluster), axis=0) for cluster in contact_point_clusters]

        if len(contact_points) == 1:
            print("Only one contact point!")
        elif len(contact_points) > 2:
            print(f"{len(contact_points)} contact points! Somethin' ain't right bruv")

        
        # keep only 3 contact points (shouldn't happen, but just in case)

        # fill in with zeros if there are less than 3 contact points (which should always be the case!)      
        while len(contact_points) < 3:
            contact_points.append((0, 0))

        contact_points = np.array(contact_points)[:3]

        return np.array(contact_points)

    def dump(self, scene : bpy.types.Scene, callbacks: list = []) -> bool:
        """
        Dump data at each frame change. Method is called by the listener on frame change.
        """

        frame = scene.frame_current

        for callback in callbacks:
            callback(scene, frame)

        # Ensure changes are visible to renderer 
        # update(0) # delay = 0 for simulation speed

        return self.end_simulation or frame > self.num_frames

    def _make_data_string(self, frame: int) -> str:
        data = [str(self.data[column][frame-1]) for column in self.columns]
        data = ",".join(data) + "\n"
        return data

    # Callbacks
    def reset_simulation_callback(self, scene: bpy.types.Scene, frame: int):
        """
        Callback to reset the simulation data at the beginning of the simulation.
        """
        if frame == 1:
            self.reset_simulation()
            scene.frame_end = self.num_frames

    def handle_force_initial_force_callback(self, scene: bpy.types.Scene, frame: int):
        """
        DEPRECATED! Using constant wind + mass instead for multiple knots.
        Callback to handle the initial force applied to the knot.
        """
        if frame == 1:
            # add initial force
            scene.gravity = self.simulation_config.initial_force

        # remove initial force
        if frame > self.simulation_config.initial_force_length:
            scene.gravity = (0, 0, -9.8)

    def dump_data_callback(self, scene: bpy.types.Scene, frame: int):
        """
        Callback to dump the center of mass data at each frame.
        """
        # COMPUTE DATA
        knot_handle = scene.objects['knot']
        plane_handle = scene.objects['Plane']

        cm_x, cm_y, cm_z = self.compute_center_of_mass(knot_handle, plane_handle)
        self.data["center_of_mass_X"][frame-1] = cm_x
        self.data["center_of_mass_Y"][frame-1] = cm_y
        self.data["center_of_mass_Z"][frame-1] = cm_z

        contact_points = self.compute_contact_points(knot_handle, plane_handle, scene)
        for i in range(3):
            self.data[f"contact_point_{i}_X"][frame-1] = contact_points[i][0]
            self.data[f"contact_point_{i}_Y"][frame-1] = contact_points[i][1] 

        
        # DUMP DATA
        if self.data_config.online:
            # append data to time series file
            with open(self.online_path, 'a') as file:
                data = self._make_data_string(frame)
                file.write(data)     

        
        distance_travelled = np.sqrt(cm_x**2 + cm_y**2)
        if distance_travelled>=self.simulation_config.target_distance or frame==self.num_frames:
            self.dump_offline(frame, scene)
            self.end_simulation = True
            
    def update_cm_callback(self, scene: bpy.types.Scene, frame: int):
        """
        Callback to update the center of mass display at each frame.
        """
        knot_handle = scene.objects[self.knot_config.name]
        self.update_cm(scene, frame, knot_handle)
        
    def update_cm(self, scene, frame, knot):

        cm_name = self.knot_config.name + "_cm"
        cm_curve_name = self.knot_config.name + "_cm_curve"
        cm_curve_handle = scene.objects[cm_curve_name]
        cm_spline = cm_curve_handle.data.splines[0]
        cm_handle = scene.objects[cm_name]

        if not hasattr(self, 'cmx_hist'):
            self.cmx_hist = []

        if len(self.cmx_hist) < self.num_frames: 

            # update center of mass object
            cm_x, cm_y, cm_z = self.compute_global_center_of_mass(knot)

            if len(self.cmx_hist) >= frame:
                self.cmx_hist[frame-1] = cm_x
            else:
                self.cmx_hist.append(cm_x)

            if frame > 1:
                cm_spline.points.add(1)

            # add new point to the spline
            cm_spline.points[frame - 1].co = (0, cm_y, cm_z, 1)
            build_modifier = cm_curve_handle.modifiers.get("Build")
            build_modifier.frame_duration = max(frame-1, build_modifier.frame_duration)

        else:
            _, cm_y, cm_z = cm_spline.points[frame-1].co[:3]
            cm_x = self.cmx_hist[frame-1]

        cm_handle.location = (cm_x, cm_y, cm_z)
        cm_curve_handle.location[0] = cm_x

    def update_camera_callback(self, scene: bpy.types.Scene, frame: int):
        """
        Callback to update the camera position at each frame.
        """
        knot_handle = scene.objects[self.knot_config.name]  
        camera_handle = scene.camera

        # set camera position to center of mass
        _, cm_y, _ = self.compute_global_center_of_mass(knot_handle)
        
        # threshold = -12
        threshold = -1
        camera_handle.delta_location[1] = min(0, cm_y-threshold)
