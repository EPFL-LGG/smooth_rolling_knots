from typing import Tuple, List, Callable
import numpy as np
from scipy.spatial import ConvexHull


def torus_knot(p : int = 2, q : int = 3, n : int = 200, r1 : float = 4, r2 : float = 2) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    ''' # Create a torus knot
    - r1 is the radius of the circle from the origin (z axis)
    - r2 is the radius of the circle around the r1 circle axis
    - p is number of times the knot wraps around the r1 circle
    - q is number of times the knot wraps around the r2 circle
    
    returns vertices and edges

    note: if p and q are not relatively prime, then the torus knot has more than one component
    '''
    phi = np.linspace(0, 2 * np.pi, n+1)
    # remove last element to avoid duplicate at 0 and 2pi
    phi = phi[:-1]
    r = r2 * np.cos(q*phi) + r1
    X = r * np.cos(p*phi)
    Y = r * np.sin(p*phi)
    Z = -r2* np.sin(q*phi)

    edges = np.linspace(0, n-1, n, dtype=int)
    edges = np.vstack([edges, np.roll(edges, 1)]).T

    return np.array([X, Y, Z]).T, edges

def Morton_knot(a : float = 0.5831, n : int = 200, p : int = 3, q : int = 2) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    # Create a Morton knot
    - a is the parameter that determines the knot
    - p and q are the parameters that determine the torus revolutions
    - n is the number of points in the knot

    returns vertices and edges
    """
    b = np.sqrt(1 - a**2)

    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vertices = Morton_knot_parametric(phi, a, p, q)

    edges = np.linspace(0, n-1, n, dtype=int)
    edges = np.vstack([edges, np.roll(edges, 1)]).T

    return vertices, edges

def Morton_knot_parametric(phi:float, a : float = 0.5831, p : int = 3, q : int = 2) -> np.ndarray:
        """
        # Create a Morton knot
        - phi is the curve parameter
        - a is the parameter that determines the knot geometry
        - p and q are the parameters that determine the torus revolutions
        - n is the number of points in the knot
        
        returns vertices
        """
        b = np.sqrt(1 - a**2)
    
        # impose R + r = 1
        c = a/(1 + b)
    
        denom = 1 - b * np.sin(q*phi)
        X = c * a * np.cos(p*phi) / denom 
        Y = c * a * np.sin(p*phi) / denom
        Z = c * b * np.cos(q*phi) / denom

        knot_points = np.array([X, Y, Z]).T

        # In some cases we need to rotate the knot to correctly fit the ellipses afterwards
        if p%4 == 1:
            rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]])
            knot_points = np.einsum("ij,kj->ki",rotation,knot_points)
        return knot_points

def grad_Morton_knot_parametric(phi:float, a : float = 0.5831, p : int = 3, q : int = 2) -> np.ndarray:
    """
    Gradient of the Morton knot
    """
    b = np.sqrt(1 - a**2)
    c = a/(1 + b)
    
    # Gradient computed via wolfram mathematica
    denom = (b * np.sin(q*phi) - 1)**2
    X = c * a * (b * q * np.cos(p * phi) * np.cos(phi * q) + p * np.sin(p * phi) * (-1 + b * np.sin(phi * q))) / denom
    Y = c * a * (b * q * np.cos(phi * q) * np.sin(p * phi) + np.cos(p * phi) * (p - b * p * np.sin(phi * q))) / denom
    Z = c * b * q * (b - np.sin(phi * q)) / denom

    return np.array([X, Y, Z]).T

def Morton_knot_no_radius_constraint(a : float = 0.5831, c : int = 1, n : int = 200, p : int = 3, q : int = 2) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Exploratory code: testing out the effect of the radius constraint on the Morton knot

    # Create a Morton knot
    - a is the parameter that determines the knot
    - p and q are the parameters that determine the torus revolutions
    - n is the number of points in the knot

    """
    b = np.sqrt(1 - a**2)

    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    denom = 1 - b * np.sin(q*phi)
    X = c * a * np.cos(p*phi) / denom 
    Y = c * a * np.sin(p*phi) / denom
    Z = c * b * np.cos(q*phi) / denom

    edges = np.linspace(0, n-1, n, dtype=int)
    edges = np.vstack([edges, np.roll(edges, 1)]).T

    return np.array([X, Y, Z]).T, edges

def Morton_knot_exterior_resolution(a : float = 0.5831, n : int = 200, p : int = 3, q : int = 2) -> np.ndarray:
        """
        Exploratory code: testing out the effect of the resolution of the exterior segments

        # Create a Morton knot
        - a is the parameter that determines the knot geometry
        - p and q are the parameters that determine the torus revolutions
        - n is the number of points in the knot

        """
        b = np.sqrt(1 - a**2)
    
        # impose R + r = 1
        c = a/(1 + b)
    
        denom = lambda phi: 1 - b * np.sin(q*phi)
        # denom = 1
        x = lambda phi: c * a * np.cos(p*phi) / denom(phi)
        y = lambda phi: c * a * np.sin(p*phi) / denom(phi)
        z = lambda phi: c * b * np.cos(q*phi) / denom(phi)

        phi = np.linspace(0, 2 * np.pi, n+1)
        # remove last element to avoid duplicate at 0 and 2pi
        phi = phi[:-1]

        X = x(phi)
        Y = y(phi)
        Z = z(phi)
        
        # get hull
        vertices = np.array([X, Y, Z]).T
        hull = ConvexHull(vertices)
        
        # get interior vertices
        interior_bool = np.ones(n, dtype=bool)
        interior_bool[hull.vertices] = False
        interior_vertices = vertices[interior_bool]

        X = interior_vertices[:, 0]
        Y = interior_vertices[:, 1]
        Z = interior_vertices[:, 2]

        # find the angles generating the segments that define the convex hull
        angles = phi[hull.vertices]
        delta_angle = 2*np.pi/n
        
        bound_idx = np.where(np.diff(angles) - delta_angle > 1e-6)[0]
        
        segment_idx_ranges = [(bound_idx[i]+1, bound_idx[i+1]) for i in range(len(bound_idx)-1)]

        if angles[0] - angles[-1] > 1e-6: # join the first and last segments 
            segment_idx_ranges.append((bound_idx[-1]+1, bound_idx[0]))
        else: # add the first and last segments separately
            segment_idx_ranges.append((0, bound_idx[0]))
            segment_idx_ranges.append((bound_idx[-1]+1, len(angles)-1))
    
        total_angle_range = 0
        expand_range_by = 2*np.pi/n
        for r in segment_idx_ranges:
            if r[1] - r[0] > 0:
                angle_range = angles[r[1]] - angles[r[0]]
            else:
                angle_range = 2*np.pi - (angles[r[0]] - angles[r[1]])
            
            total_angle_range += 2*expand_range_by
        
        total_angle_range = total_angle_range if total_angle_range < 2*np.pi else 2*np.pi

        for r in segment_idx_ranges:
            if r[1] - r[0] > 0:
                start = angles[r[0]] - expand_range_by
                end = angles[r[1]] + expand_range_by
            else:
                start = angles[r[0]] - expand_range_by
                end = 2*np.pi + angles[r[1]] + expand_range_by
            
            n_points = int(n * (end - start) / total_angle_range)
            segment_phi = np.linspace(start, end, n_points)

            X = np.concatenate([X, x(segment_phi)])
            Y = np.concatenate([Y, y(segment_phi)])
            Z = np.concatenate([Z, z(segment_phi)])

        return np.array([X, Y, Z]).T

def Henrion_knot(n : int = 200) -> np.ndarray:
        """
        A knot I was curious about
        """
        phi = np.linspace(0, 2 * np.pi, n+1)
        # remove last element to avoid duplicate at 0 and 2pi
        phi = phi[:-1]
        X = np.cos(phi) + 2*np.cos(2*phi)
        Y = np.sin(phi) + 2*np.sin(2*phi)
        Z = 2*np.sin(3*phi)
    
        return np.array([X, Y, Z]).T

def torus(r1 : float = 4, r2 : float = 2, n=100) -> np.ndarray:
    """
    Create a torus
    - r1 is the radius of the circle from the origin (z axis)
    - r2 is the radius of the circle around the r1 circle axis (tube radius)
    - n is the number of points in the torus
    """
    phi = np.linspace(0, 2 * np.pi, n)
    theta = np.linspace(0, 2 * np.pi, n)
    phi, theta = np.meshgrid(phi, theta)
    x = (r1 + r2 * np.cos(theta)) * np.cos(phi)
    y = (r1 + r2 * np.cos(theta)) * np.sin(phi)
    z = r2 * np.sin(theta)
    vertices = np.array([x, y, z]).T
    return vertices

def z_stretch(v : np.ndarray, stretch : float = 1) -> np.ndarray:
    """
    Apply a stretch to the z axis of a set of vertices
    """
    new_v = v.copy()
    if len(v.shape) == 1:
        new_v[2] *= stretch
    else:
        new_v[:, 2] *= stretch
    return new_v

def xy_projection(v : np.ndarray) -> np.ndarray:
    if len(v.shape) == 1:
        x,y,z = v
    else:
        x,y,z = v.T

    xy = 0.5*(x+y)
    return np.array([xy, xy, z]).T

def rotated_morton_knot_parametric(phi:float, a : float = 0.5831, p : int = 3, q : int = 2, z_stretch : int = 1) -> np.ndarray:
    b = np.sqrt(1 - a**2)
    c = a/(1 + b)
    d = c/(1-b*np.sin(q*phi))
    x = d*a*(np.cos(p*phi)-np.sin(p*phi))/np.sqrt(2)
    y = d*a*(np.cos(p*phi)+np.sin(p*phi))/np.sqrt(2)
    z = z_stretch*d*b*np.cos(q*phi)
    knot_points = np.array([x, y, z]).T

    # Rotate the knot depending on the p value
    # p % 8 == 3 -> 0˚
    # p % 8 == 1 -> 90˚
    # p % 8 == 7 -> 180˚
    # p % 8 == 5 -> 270˚
    angle = {
        3: 0, 
        1: np.pi/2, 
        7: np.pi, 
        5: 3*np.pi/2
    }[p % 8]
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], 
                         [np.sin(angle), np.cos(angle), 0], 
                         [0, 0, 1]])
    knot_points = np.einsum("ij,kj->ki", rotation, knot_points)

    return knot_points


def oloid(dist:float=1, n:int = 100) -> ConvexHull:
    """
    Create an oloid
    - dist is the distance between the two circles (oloid is actually distance 1, zero rolliness is distance sqrt2)
    - n is the number of points in each of the disks

    returns the scipy ConvexHull of the oloid disks
    """
    
    circle1, circle2 = oloid_disks(dist, n)

    vertices = np.concatenate([circle1, circle2])
    
    return ConvexHull(vertices)

def oloid_disks(dist:float=1, n:int = 100) -> ConvexHull:
    """
    Create an oloid's defining disks
    - dist is the distance between the two circles (oloid is actually distance 1, zero rolliness is distance sqrt2)
    - n is the number of points in each of the disks

    returns the two disks as a tuple
    """

    phi = np.linspace(0, 2 * np.pi, n)
    circle1 = np.array([np.cos(phi), np.sin(phi), np.zeros(n)]).T
    circle2 = np.array([np.zeros(n), np.cos(phi), np.sin(phi)]).T

    circle2[:,1] -= dist/2
    circle1[:,1] += dist/2

    return (circle1, circle2)

def sphericon(n:int = 100) -> ConvexHull:
    """
    Create a sphericon
    - n is the number of points in each of the half circles
    """
 
    half_circle1, half_circle2 = sphericon_half_circles(n)

    vertices = np.concatenate([half_circle1, half_circle2])

    return ConvexHull(vertices)

def sphericon_half_circles(n:int = 100) -> np.ndarray:
    """
    Create a sphericon's defining half circles
    - n is the number of points in each of the half circles

    returns the two half circles as a tuple
    """

    phi1 = np.linspace(0, np.pi, n)
    phi2 = np.linspace(np.pi/2, 3*np.pi/2, n)
    half_circle1 = np.array([np.cos(phi1), np.sin(phi1), np.zeros(n)]).T
    half_circle2 = np.array([np.zeros(n), np.cos(phi2), np.sin(phi2)]).T
    return (half_circle1, half_circle2)

def TDR(n:int = 100, a:float=1, b:float=1, c:float=None) -> ConvexHull:
    """
    Create a TDR
    - n is the number of points in each of the disks
    - a is the radius of the disks along the x axis
    - b is the radius of the disks along the y and z axes
    - c is the distance between the two disks 
        - if None, c is calculated to obtain zero rolliness

    """
    
    disks = TDR_disks(n, a, b, c)
    
    vertices = np.concatenate(disks)

    return ConvexHull(vertices)

def TDR_disks(n:int = 100, a:float=1, b:float=1, c:float=None) -> np.ndarray:
    """
    Create a TDR's defining disks
    - n is the number of points in each of the disks
    - a is the radius of the disks along the x axis
    - b is the radius of the disks along the y and z axes
    - c is the distance between the two disks 
        - if None, c is calculated to obtain zero rolliness
    
    returns the two disks as a tuple
    """

    phi = np.linspace(0, 2*np.pi, n)
    phi1 = phi2 = phi

    # HARD CODED FIX TO GET THE ENDPOINTS OF THE ELLIPSES INSIDE THE TDR (PUT THE ENDPOINTS NOT ON THE CONVEX HULL)    
    shift = n//4
    phi1 = np.roll(phi, shift)
    phi2 = np.roll(phi, -shift)

    assert a > b/np.sqrt(2), "ratio must be greater than 1/sqrt(2)"
    
    # a = 0.5/np.sqrt(2) + 0.3
    if c is None:
        c = np.sqrt(4*a**2 - 2*b**2)
    ellipse1 = np.array([a*np.sin(phi1) + 0.5*c, b*np.cos(phi1), np.zeros(n)]).T
    ellipse2 = np.array([a*np.sin(phi2) - 0.5*c, np.zeros(n), b*np.cos(phi2)]).T
    return (ellipse1, ellipse2)

def disk(n:int = 100, r:float=1) -> np.ndarray:
    """
    Create a disk in the xy plane with z=0 
    - n is the number of points in the disk
    - r is the radius of the disk
    """
    phi = np.linspace(0, 2*np.pi, n)
    circle = np.array([r*np.cos(phi), r*np.sin(phi), np.zeros(n)]).T

    return circle

def ellipse(n:int = 100, a:float=1, b:float=0.5) -> np.ndarray:
    """
    Create an ellipse in the xy plane with z=0
    - n is the number of points in the ellipse
    - a is the radius of the ellipse along the x axis
    - b is the radius of the ellipse along the y axis
    """
    phi = np.linspace(0, 2*np.pi, n)
    ellipse = np.array([a*np.cos(phi), b*np.sin(phi), np.zeros(n)]).T

    return ellipse

# FROM MAYAVI DOCS
def _mayavi_torus_knot(n_mer, n_long):
    phi = np.linspace(0, 2*np.pi, 2000)
    return [ np.cos(phi*n_mer) * (1 + 0.5*np.cos(n_long*phi)),
            np.sin(phi*n_mer) * (1 + 0.5*np.cos(n_long*phi)),
            0.5*np.sin(n_long*phi),
            np.sin(phi*n_mer)]

if __name__ == "__main__":
    # morton_knot = Morton_knot()
    knot1, _ = Morton_knot(n=100)
    knot2 = Morton_knot_exterior_resolution(n=100)

    from matplotlib import pyplot as plt
    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211, projection="3d")
    ax2 = fig.add_subplot(212, projection="3d")

    ax1.scatter3D(knot1[:, 0], knot1[:, 1], knot1[:, 2])
    ax2.scatter3D(knot2[:, 0], knot2[:, 1], knot2[:, 2])

    print("done")




    