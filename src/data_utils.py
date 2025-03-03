import numpy as np
import os

# DATA UTILS ========================================================================================================

def save_data(data:dict, filename:str, path:str="./data/"):
    """
    Save data to a file.
    """
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
        # elif isinstance(value, dict):
        #     new_file_name = filename.split(".")[0] + "_" + key + ".json"
        #     save_data(value, new_file_name, path)
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    if filename.endswith(".json"):
        import json
        with open(path + filename, "w") as f:
            json.dump(data, f)
        return
    elif filename.endswith(".yaml"):
        import yaml
        with open(path + filename, "w") as f:
            yaml.dump(data, f)
        return
    raise ValueError("Unknown file format")

def load_data(filename:str, path:str="./data/") -> dict:
    """
    Load dictionary data from a file.
    """
    import os
    if not os.path.exists(path):
        raise ValueError("Path does not exist")
    if filename.endswith(".json"):
        import json
        with open(path + filename, "r") as f:
            return json.load(f)
    elif filename.endswith(".yaml"):
        import yaml
        with open(path + filename, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    raise ValueError(f"Unknown file format: {filename}")

def get_data(path:str="./data/rolliness/") -> dict:
    # Get all data from a directory
    import os
    data = {}
    for r, d, f in os.walk(path):
        for file in f:
            try:
                p = r + "/"
                data[file] = load_data(file, p)
            except Exception as e:
                print(e)
    return data

# def save_knot(obj: np.ndarray, filename:str, path:str="./data/"):
#     """
#     Save an object to a file.
#     """
#     import pickle
#     with open(path + filename, "wb") as f:
#         pickle.dump(obj, f)

def save_knot(obj: np.ndarray, edges: np.ndarray, filename:str, path:str="./data/"):
    """
    Save an object to an .obj file.
    """
    with open(os.path.join(path, filename), "w") as f:
        # Write vertices
        for vertex in obj:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write edges
        for edge in edges:
            f.write(f"l {edge[0]} {edge[1]}\n")


def load_pickled_knot(filename:str, path:str="./data/"):
    """
    Load an object from a file.
    """
    import pickle
    with open(path + filename, "rb") as f:
        return pickle.load(f)

def load_knot(filename:str, path:str="./data/") -> np.ndarray:
    """
    Load an object from an .obj file.
    """
    vertices = []
    edges = []

    try: 
        with open(os.path.join(path, filename), "r") as f:
            for line in f:
                if line.startswith("v "):
                    vertices.append([float(x) for x in line.split()[1:]])
                elif line.startswith("l "):
                    edges.append([int(x) for x in line.split()[1:]])
        return np.array(vertices)
    except Exception as e:
        return load_pickled_knot(filename, path)
