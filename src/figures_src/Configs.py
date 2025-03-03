import json, yaml, os, copy

class Config:
    def __init__(self, config:dict):
        self.path = os.getcwd() + "/src/figures_src/configs/"
        # convert to class members
        for key in config:
            setattr(self, key, copy.deepcopy(config[key]))

    @staticmethod
    def from_json(json_str: str):
        config = json.loads(json_str)
        return Config(config)
    
    @staticmethod
    def from_yaml(yaml_str: str):
        config = yaml.load(open(yaml_str, 'r'), Loader=yaml.FullLoader)
        return Config(config)
    
    def load_defaults(self, default_path:str):
        default_config = self.from_yaml(default_path)
        for key in default_config.__dict__:
            if key not in self.__dict__:
                setattr(self, key, getattr(default_config, key))

    def __setitem__(self, key, value):
        setattr(self, key, value)

class BlenderConfig(Config):
    def __init__(self, config:dict):
        super().__init__(config)
        self.load_defaults(self.path + "default_blender_config.yaml")

class KnotConfig(Config): 
    def __init__(self, config:dict):
        super().__init__(config)
        self.load_defaults(self.path + "default_knot_config.yaml")
    
