from __future__ import division

import math
import itertools

import numpy as np
from numpy.linalg import norm

from spatial import (
    Surface,
    unit,
)


PHI = (1 + np.sqrt(5)) / 2
ZERO = np.zeros(3)


class Sphere(Surface):
    def __init__(self, vertices, faces, radius=None, center=None, *args, **kwargs):
        super(Sphere, self).__init__(vertices, faces, *args, **kwargs)
        if center is None:
            center = vertices.mean(axis=0)
        if radius is None:
            radius = norm(vertices[0] - center)
        self.radius = radius
        self.center = center

    @property
    def unit_vertices(self):
        shifted_vertices = self.vertices - self.center
        scaled_vertices = shifted_vertices / self.radius
        return scaled_vertices

    @classmethod
    def from_tessellation(cls, radius=1.0, center=ZERO, iterations=3):
        vertices, triangles = tessellated_sphere(radius, center, iterations)
        sphere = cls(vertices, triangles, radius=radius, center=center)
        return sphere


class pointdict(dict):
    """ Create a dict that indexes by sorted tuples """
    @classmethod
    def make_index(cls, idx):
        return tuple(sorted(idx))

    def __getitem__(self, idx):
        return dict.__getitem__(self,self.make_index(idx))

    def __setitem__(self, idx, value):
        return dict.__setitem__(self, self.make_index(idx), value)


def create_icosahedron(radius=1.0, center=ZERO):
    # Scaled to radius
    one, phi, _ = radius * unit(np.array([1.0, PHI, 0]))
    
    vertices = np.array([
        (-one, 0, phi),
        (one, 0, phi),
        (-one, 0, -phi),
        (one, 0, -phi),

        (0, phi, one),
        (0, phi, -one),
        (0, -phi, one),
        (0, -phi, -one),

        (phi, one, 0),
        (-phi, one, 0),
        (phi, -one, 0),
        (-phi, -one, 0),
    ])
    vertices += center

    triangles = np.array([
        (0, 4, 1),
        (0, 9, 4),
        (9, 5, 4),
        (4, 5, 8),
        (4, 8, 1),

        (8, 10, 1),
        (8, 3, 10),
        (5, 3, 8),
        (5, 2, 3),
        (2, 7, 3),

        (7, 10, 3),
        (7, 6, 10),
        (7, 11, 6),
        (11, 0, 6),
        (0, 1, 6),

        (6, 1, 10),
        (9, 0, 11),
        (9, 11, 2),
        (9, 2, 5),
        (7, 2, 11),
    ])
    
    return vertices, triangles


def project_point_to_sphere(inner_point, centroid=ZERO, radius=1.):
    d = inner_point - centroid
    a = np.dot(d, d)
    b = 2 * np.dot(centroid, d)
    c = np.dot(centroid, centroid) - radius**2
    t = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    point = 2 * centroid + t * d
    return point


def subdivide_icosahedron(vertexes, existing_triangles):
    new_vertexes = list(vertexes)               # Make a copy of vertex list
    new_triangles = []                          # Create list of triangles
    index = itertools.count(len(new_vertexes))  # Track added vertex indicies

    # Extrapolate information about sphere
    centroid = vertexes.mean(axis=0)
    radius = norm(vertexes[0] - centroid)
    subdivided_edges = pointdict()
    for triangle in existing_triangles:
        a_idx, b_idx, c_idx = triangle
        edges = [(a_idx, b_idx),
                 (b_idx, c_idx),
                 (c_idx, a_idx)]

        # Calculate new verticies (checking for already subdivided edges)
        subdivided_vertexes = []
        for edge in edges:
            try:
                existing_idx = subdivided_edges[edge]
                subdivided_vertexes.append(existing_idx)
            except KeyError:
                u_idx, v_idx = edge
                # Look up coordinates for existing verticies
                u = vertexes[u_idx]
                v = vertexes[v_idx]

                # Compute new vertex
                w_m = (u + v) / 2
                w = project_point_to_sphere(w_m, centroid, radius)

                # Cache edge subdivision, store new edge, create new index
                w_idx = index.next()
                new_vertexes.append(w)
                subdivided_edges[edge] = w_idx
                subdivided_vertexes.append(w_idx)

        d_idx, e_idx, f_idx = subdivided_vertexes
        
        # Create new triangles (by vertex IDs)
        triangle1 = (a_idx, d_idx, f_idx)
        triangle2 = (d_idx, b_idx, e_idx)
        triangle3 = (f_idx, e_idx, c_idx)
        triangle4 = (f_idx, d_idx, e_idx)

        # Record new triangles
        new_triangles.append(triangle1)
        new_triangles.append(triangle2)
        new_triangles.append(triangle3)
        new_triangles.append(triangle4)

    return np.array(new_vertexes), np.array(new_triangles)


def iterations_needed_for_triangles(num_triangles):
    # Faces(iterations) = 20 * 4**iterations
    return int(math.ceil(math.log(num_triangles/20, 4)))


def tessellated_sphere(radius=1.0, center=ZERO, 
                       iterations=None, 
                       min_triangles=None):
    if iterations is None and min_vertices is None:
        raise ValueError("Please specific either # of iterations or vertices")
    if min_triangles is not None:
        iterations = iterations_needed_for_triangles(min_triangles)
    vertexes, triangles = create_icosahedron(radius=1.0, center=ZERO)
    for iteration in range(iterations):
        vertexes, triangles = subdivide_icosahedron(vertexes, triangles)
    vertexes *= radius
    vertexes += center
    return vertexes, triangles


