import numpy as np

def normalize_color(color):
    color = np.array(color, dtype=np.float64)
    color[:3] /= 255.0
    return color

color1 = normalize_color((0, 120, 255, 1))
color2 = normalize_color((255, 153, 2, 1))
color3 = normalize_color((26, 163, 26, 1))