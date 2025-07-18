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

# ===================================================================================

import utils
# install dependencies to the blender python environment
utils.install_dependencies()

import numpy as np
import json
import yaml

from src.simulation_src import simulation_manager
from src.simulation_src.config import Config
from src.simulation_src.scene_setup import update

# UTILS =============================
def load_config(path : str, config : dict = None, keys : list = None) -> dict:
    """
    Load a config file from path. 
    If config is not None, we only update the specified keys from the loaded config to current config.
    """
    print(f"Loading config from {path}")
    return Config(yaml.load(open(path,"r"), Loader=yaml.FullLoader))

def write_config(path : str, name : str, config : dict):
    if not os.path.exists(path):
        os.makedirs(path)
    config_path = path + f"/{name}.yaml"
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def load_knot_config(config_name : str) -> dict:
    path = 'knot_configs.json'
    with open(path) as f:
        knot_configs = json.load(f) 
        return knot_configs[config_name]
 
# deep copy of dictionary, with support for nested dictionaries
def dict_deep_copy(d) -> dict:
    if not isinstance(d, dict):
        return d
    return {k : dict_deep_copy(v) for k, v in d.items()}

def a_configs(config : dict) -> list: 
    """
    Create a list of configs with different values of a.
    """
    deltas = np.linspace(-0.1, 0.1, 100+1)
    configs = []
    for delta in deltas:
        # deep copy config
        new_config = dict_deep_copy(config)
        a = new_config.knot_config.config.a + delta
        new_config.knot_config.config.a = a
        new_config.plot_config.label = r"$a={:4f}$".format(a)
        data_path = config.data_config.data_path
        config_name = f"morton_{a:4f}".replace(".", "_") 
        trace_path = f"{data_path}/{config_name}"
        new_config.data_config.path = trace_path
        configs.append(new_config)
    return configs

# MAIN =============================

if __name__ == "<run_path>":

    # force reload of modules (blender keeps cached versions of modules, so changes to the code won't be reflected without a reload)
    utils.reload_modules()

    SimulationManager = simulation_manager.SimulationManager

    print("Running main.py")

    # config_path = 'data/simulations/configs/default_config.yaml'
    # # config_path = 'data/simulations/configs/center_of_mass.yaml'
    config_path = 'data/simulations/configs/cm_75_base.yaml'
    # config_path = 'data/simulations/configs/cm_75_opt.yaml'
    # config_path = 'data/simulations/configs/multi.yaml'
    # config_path = 'data/simulations/configs/projected_knot_config.yaml'
    # config_path = 'data/simulations/configs/opt_knot_config.yaml'
    # config_path = 'data/simulations/configs/knot_config.yaml'

    config = load_config(config_path)

    # optionally load multiple configs to run multiple simulations in a row
    configs = [config]

    sim_manager = SimulationManager(configs)
    sim_manager.run_next()





