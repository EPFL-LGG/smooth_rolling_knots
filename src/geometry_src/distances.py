import numpy as np
import torch


def chamfer_distance_squared(vertices1, vertices2):
    '''Computes the squared (asymmetric) chamfer distance between vertices1 and vertices2.
    Note: the computation is not symmetric, so not technically a distance.
    
    Args:
        vertices1 (torch.Tensor of shape (n_vertices1, 3)): The first set of vertices.
        vertices2 (torch.Tensor of shape (n_vertices2, 3)): The second set of vertices.
    
    Returns:
        dist_sq (torch.Tensor of shape (,)): The squared chamfer distance between the two sets of vertices.
    '''
    dists_sq = torch.sum((vertices1.unsqueeze(0) - vertices2.unsqueeze(1)) ** 2, dim=-1)
    dist_sq = torch.mean(torch.min(dists_sq, dim=0).values)
    return dist_sq

def symmetric_chamfer_distance_squared(vertices1, vertices2):
    '''Computes the symmetric chamfer distance between two sets of vertices.
    
    Args:
        vertices1 (torch.Tensor of shape (n_vertices1, 3)): The first set of vertices.
        vertices2 (torch.Tensor of shape (n_vertices2, 3)): The second set of vertices.
    
    Returns:
        obj (torch.Tensor of shape (,)): The symmetric chamfer distance between the two sets of vertices.
    '''
    dists_sq = torch.sum((vertices1.unsqueeze(0) - vertices2.unsqueeze(1)) ** 2, dim=-1)
    obj = torch.mean(torch.min(dists_sq, dim=1).values) + torch.mean(torch.min(dists_sq, dim=0).values)
    
    return obj

##############################################
###        POINT TO POINT DISTANCE        ###
##############################################

def point_point_distance(A,B):
    return torch.linalg.norm(B-A)

def norm2(x):
    return x.dot(x)

##############################################
###           POINT EDGE DISTANCE          ###
##############################################

def point_edge_distance_param(P,A,B):
    """
    Args:
        P : point
        A : first edge endpoint
        B : second edge endpoint
    """
    AB = B - A
    AP = P - A
    return min(1,max(0,AB.dot(AP)/ norm2(AB)))

def point_edge_closest_point(P,A,B):
    """
    Args:
        P : point
        A : first edge endpoint
        B : second edge endpoint
    """
    AB = B - A
    t = point_edge_distance_param(P,A,B)
    return A + t * AB

def point_edge_distance(P,A,B):
    """
    Args:
        P : point
        A : first edge endpoint
        B : second edge endpoint
    """
    C = point_edge_closest_point(P,A,B)
    return point_point_distance(P,C)


##############################################
###         POINT POLYLINE DISTANCE        ###
##############################################


def point_to_polyline_distance(P,poly):
    
    AB = poly[1:]-poly[:-1]
    norm2AB = torch.einsum("ij,ij->i",AB,AB)
    AP = P.reshape(1,-1) - poly[:-1]
    proj = torch.einsum("ij,ij->i",AB,AP)
    edge_param = proj/norm2AB
    
    edge_param = torch.minimum(edge_param,torch.ones_like(edge_param))
    edge_param = torch.maximum(edge_param,torch.zeros_like(edge_param))

    edge_points = poly[:-1] + edge_param.reshape(-1,1) * AB

    Ppoints = P.reshape(1,-1) - edge_points
    edge_distances = torch.linalg.norm(Ppoints,dim=1)
    
    return torch.min(edge_distances)

def points_to_polyline_distance(points, poly, reduce='mean'):
    """Computes the distance from a set of points to a polyline
    Note: the computation is not symmetric, so not technically a distance.
    
    Args:
        points (torch.Tensor of shape (n, d)): The set of points.
        poly (torch.Tensor of shape (p, d)): The polyline.
        reduce (str): The reduction operation to apply to the distances. If None, the per-point distances are returned.
    
    Returns:
        dist (torch.Tensor of shape (,)): The (asymetric) distance from the set of points to the polyline.
    """
    n = points.shape[0]
    d = points.shape[1]
    p = poly.shape[0]
    
    AB = poly[1:]-poly[:-1]
    AP = points.reshape(n,1,d) - poly[:-1].reshape(1,p-1,d)
    norm2AB = torch.einsum("ij,ij->i",AB,AB)
    norm2AB[norm2AB==0] = 1 # Avoid division by 0 in case two vertices of the polyline are equal 
    proj = torch.einsum("ij,kij->ki",AB,AP)
    edge_params = torch.einsum("ki,i->ki",proj,1/norm2AB)

    edge_params = torch.minimum(edge_params,torch.ones_like(edge_params))
    edge_params = torch.maximum(edge_params,torch.zeros_like(edge_params))

    edge_points = poly[:-1].reshape(1,p-1,d) + edge_params.reshape(n,p-1,1) * AB.reshape(1,p-1,d)

    Ppoints = points.reshape(n,1,d) - edge_points
    edge_distances = torch.linalg.norm(Ppoints,dim=2)

    m = torch.min(edge_distances, 1)

    if reduce is None:
        return m.values
    elif reduce == 'mean':
        return torch.mean(m.values) 
    else:
        raise ValueError(f"Unknown reduction operation: {reduce}")