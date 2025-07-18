import yaml
from pathlib import Path
from typing import Union

DEFAULT_CONFIG_PATH = Path('data/simulations/configs/default_config.yaml')

class BaseConfig:
    
    def __init__(self, config:dict, default_config:Union[Path, dict, None] = None):
        
        if isinstance(default_config, (Path, str)):
            with open(default_config, 'r') as file:
                default_config = yaml.load(file, Loader=yaml.FullLoader)

        self.load_config(default_config)
        self.load_config(config)

    def load_config(self, config: dict): 
        for key in config:
            setattr(self, key, config[key])

class Config(BaseConfig):
    def __init__(self, config:dict, default_config_path:Path = DEFAULT_CONFIG_PATH):

        if "multi_config" in config:
            print("Loading multi-config")
            # if multi_config is present, it means the config is a list of configs
            # so we load each config in the list
            self.configs = [Config(yaml.load(open(c,"r"), Loader=yaml.FullLoader)) for c in config["configs"]]
            self.multi_config = True
            return

        self.multi_config = False
        super().__init__(config, default_config_path)

    def load_config(self, config:dict):
        
        # set class member, or update config member if it already exists
        set_or_update = lambda cls, key, value: setattr(self, key, cls(value)) if not hasattr(self, key) else getattr(self, key).load_config(value)

        # convert to class members
        for key in config:
            if key == 'plot_config':
                set_or_update(PlotConfig, key, config[key])
            elif key == 'knot_config':
                set_or_update(KnotConfig, key, config[key])
            elif key == 'blender_config':
                set_or_update(BlenderConfig, key, config[key])
            elif key == 'simulation_config':
                set_or_update(SimConfig, key, config[key])
            elif key == 'data_config':
                set_or_update(DataConfig, key, config[key])
            else:
                setattr(self, key, config[key])
        
        if self.plot_config.label == "a":
            self.plot_config.label = self.knot_config.a

class SimConfig(BaseConfig):
    def __init__(self, config:dict):
        super().load_config(config)
        
class PlotConfig(BaseConfig):
    def __init__(self, config:dict):
        super().load_config(config)
        # check label
        if not hasattr(self, 'label'):
            self.label = None

class KnotConfig(BaseConfig):
    def __init__(self, config:dict):
        self.load_config(config)

    def load_config(self, config:dict):
        # convert to class members
        for key in config:
            if key == "name":
                setattr(self, key, f"knot_{config[key]}")
            elif key == "config":
                for other_key in config[key]:
                    setattr(self, other_key, config[key][other_key])
            else:
                setattr(self, key, config[key])


class BlenderConfig(BaseConfig):
    def __init__(self, config:dict):
        super().load_config(config)

class DataConfig(BaseConfig):
    def __init__(self, config:dict):
        super().load_config(config)