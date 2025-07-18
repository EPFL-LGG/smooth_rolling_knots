import bpy
from src.simulation_src import scene_setup, simulation
import src.utils as utils

class SimulationManager:   
    """
    Simulation manager class that runs a list of simulation configurations in sequence.
    """ 

    def __init__(self, configs : list):
        utils.reload_modules()
        self.configs = configs
        self.current_config = None
        self.i = 0
        bpy.app.handlers.frame_change_post.clear()


    def validate_config(self, config:dict) -> dict:
        # if "simulation_config" not in config:
        #     raise ValueError("simulation_config not found in config")
        # if "data_config" not in config:
        #     raise ValueError("data_config not found in config")
        # if "knot_config" not in config:
        #     raise ValueError("knot_config not found in config")
        # if "blender_config" not in config:
        #     raise ValueError("blender_config not found in config")
        return config

    def simulate(self, config : dict):
        
        config = self.validate_config(config)

        # create scene 
        if not config.multi_config:
            ss = scene_setup.SceneSetup(config)
            ss.init_scene()
        else:
            ss = scene_setup.SceneSetup(config.configs[0])
            ss.init_scene()
            for c in config.configs[1:]:
                ss.load_config(c)
                ss.init_scene()

        # create simulation object
        # although the simulation is taken care of by blender, this object is used to monitor and manipulate the simulation data
        if not config.multi_config:
            self.sim_setup(config)
        else:
            for c in config.configs:
                self.sim_setup(c)

    def sim_setup(self, config):
        sim = simulation.Simulation(config)

        callbacks = [sim.reset_simulation_callback]
        if config.data_config.online:
            # if online data export is enabled, add a callback to the simulation object to collect data at each frame change
            callbacks.append(sim.dump_data_callback)

        if config.blender_config.show_cm:
            # if the center of mass should be shown, add a callback to the simulation object to update the center of mass at each frame change
            callbacks.append(sim.update_cm_callback)

        if config.blender_config.pin_camera:
            # if the camera should be pinned to the knot, add a callback to the simulation object to update the camera position at each frame change
            callbacks.append(sim.update_camera_callback)

        # listener for simulation frame change
        def on_frame_change(scene):

            done = sim.dump(scene, callbacks)
            if done:
                # clear listener list (since it's persistent throughout runs)
                bpy.app.handlers.frame_change_post.clear()
                # run next simulation
                self.run_next()
        
        # add listener to frame change (simulation object collects data at each frame change)
        bpy.app.handlers.frame_change_post.append(on_frame_change)

        # change frame to trigger listener at frame 1
        bpy.context.scene.frame_set(1)

        if config.blender_config.autoplay:
            bpy.ops.screen.animation_play()

    def run_next(self):
        if self.configs:
            self.current_config = self.configs.pop(0)
            self.i += 1

            print(f"Running simulation {self.i}")
            self.simulate(self.current_config)
            if hasattr(self.current_config, 'blender_config') and self.current_config.blender_config.loop:
                self.configs.append(self.current_config)
        else:
            print("All simulations complete")