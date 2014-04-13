from __future__ import division

import numpy as np
from numpy.linalg import norm
from scipy.spatial import KDTree

from collections import defaultdict
from itertools import (
    chain,
    imap,
    permutations,
    islice,
)


def unit(x):
    return x / norm(x)


class setdict(defaultdict):
    """ A dictionary class that maps each key to a specific mutable set """
    def __init__(self, items=[]):
        super(setdict, self).__init__(set)
        self.update(items)
    
    def update(self, items):
        for item in items:
            self.add(*item)

    def add(self, key, value):
        self[key].add(value)

    def get_list(self, index):
        return list(self[index])

    def get_all(self, indicies):
        found = set()
        for index in indicies:
            found.update(self[index])
        return found

    def get_all_list(self, indicies):
        return list(self.get_all(indicies))


def neighbor_map(triangles):
    """ Create a dictionary to quickly retrieve the set of all neighboring
        vertices for a particular vertex index using the triangle definitions
    """
    get_vertex_pairs = lambda vertices: permutations(vertices, 2)
    triangle_permutations = imap(get_vertex_pairs, triangles)
    all_vertex_pairs = chain.from_iterable(triangle_permutations)
    neighbors = setdict(all_vertex_pairs)
    return neighbors


def triangle_map(triangles):
    """ Create a dictionary to quickly retrieve the IDs of all
        triangles a vertex is part of
    """
    vertex_triangles = setdict()
    for idx, face in enumerate(triangles):
        for vertex in face:
            vertex_triangles.add(vertex, idx)
    return vertex_triangles


class Surface(object):
    
    def __init__(self, vertices, faces, point_normals=None):
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        self.point_normals = point_normals  # Unused

        # Faces to Index
        self.face_ids = dict(map(reversed, 
                                 enumerate(map(tuple, map(sorted, faces)))))
        self.neighbors = neighbor_map(faces)
        self.face_map = triangle_map(faces)
        self.space = KDTree(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    @property
    def extents(self):
        lowest = self.vertices.min(axis=0)
        highest = self.vertices.max(axis=0) 
        return np.array([lowest, highest]).transpose()

    @property
    def centroid(self):
        return self.vertices.mean(axis=0)

    @property
    def triangles(self):
        return self.vertices[self.faces]

    def neighboring_vertices(self, idx, dist, cycles=False):
        """ idx: Index of vertex to begin at
            dist: Number of steps to take on connectivity graph
            cycles: Allow graph to loop back and include previously counted vertices
        """
        neighbors = [idx]
        seen = set()
        for step in range(dist):
            seen.update(neighbors)
            queue, neighbors = list(neighbors), set()
            for neighbor in queue:
                near_by = self.neighbors[neighbor]
                if not cycles:
                    near_by = (n for n in near_by if n not in seen)
                neighbors.update(near_by)
        return list(neighbors)

    def approximate_nearest_point(self, point, steps=2):
        nearest_vertex_dist, nearest_idx = self.space.query(point)
        indicies = self.neighboring_vertices(nearest_idx, dist=steps, cycles=True)
        nearby_faces = self.face_map.get_all_list(indicies)
        faces = self.faces[nearby_faces]
        triangles = self.vertices[faces]
        distance, triangle, nearest_point = nearest_triangle(point, triangles)
        return distance, triangle, nearest_point

    def get_face_id(self, vertex_ids):
        return self.face_id

    def nearest_surface_point(self, point, search_limit=None):
        close_faces = self.get_faces_near(point, search_limit)
        triangles = self.vertices[close_faces]
        distance, triangle, nearest_point = nearest_triangle(point, triangles)
        return distance, triangle, nearest_point

    # UNUSED
    def get_faces_near(self, point, search_limit=None):
        if search_limit is not None:
            vertex_indexes = self.space.query_ball_point(point, search_limit)
            near = self.face_map.get_all_list(vertex_indexes)
            return self.faces[near]
        else:
            return self.faces

    # UNUSED
    def distance(self, point, search_limit=None):
        dist, tri, surf_pt = self.find_nearest_on_surface(point, search_limit=search_limit)
        return dist

    @classmethod
    def from_vet_file(cls, io):
        nV, nE, nT = map(int, next(io).split())
        vert_lines = np.loadtxt(islice(io, nV), dtype=np.float)
        edge_lines = np.loadtxt(islice(io, nE), dtype=np.int)
        tri_lines = np.loadtxt(islice(io, nT), dtype=np.int) 
        
        # Ignoring edges (mistake!)
        # Todo: reorder triangles so traversals are correct based on order
        vertices = vert_lines[:,0:3]
        triangles = tri_lines[:,3:6] - 1  # For 0-indexing

        other = {
            'point_normals': vert_lines[3:6]
        }

        return cls(vertices, triangles, **other)


def project_to_plane(vector, plane_normal):
    normal_component = np.dot(vector, plane_normal) * plane_normal
    plane_component = vector - normal_component
    return plane_component


def triangle_normals(triangles):
    """ Create a matrix of unit normal vectors for a set of triangles
        defined by 3 vertices. Vertices must be provided in counter 
        clockwise order. This is per the MSRoll vet file definition.
    """
    # Only need num triangles and vertex coords
    normals = np.zeros(triangles.shape[0:2])
    for idx, triangle in enumerate(triangles):
        normals[idx] = triangle_normal(triangle)
    return normals


def triangle_normal(triangle):
    """ Compute the unit normal vector of a triangle """
    v0, v1, v2 = triangle
    normal = unit(np.cross(v1- v0, v2 - v0))
    return normal


def project_with_extrema(points, axis):
    proj = np.dot(points, axis)
    low, high = np.min(proj), np.max(proj)
    return proj, low, high


def nearest_triangle(point, triangles):
    """ Find the nearest triangle to a given point in a collection of triangles 
        Returns: distance to the triangle, 
                 the closest triangle,
                 the clostest point in the triangle to the reference point
    """
    shortest_dist, closest_triangle, closest_point = np.inf, None, None
    for triangle in triangles:
        dist, pt = distance_to_triangle(point, triangle, with_point=True)
        if dist < shortest_dist:
            shortest_dist = dist
            closest_triangle = triangle
            closest_point = pt
    return shortest_dist, closest_triangle, closest_point


def distance_to_triangle(point, triangle, with_point=False):
    # Distance could be calculated faster and cleaner by rotating
    # and translating both the triangle and point to a consistent 
    # frame of reference but this is a quicky, dirty, and ugly solution
    # for now.

    v0, v1, v2 = triangle
    e0 = v1 - v0  # Vertex 0 to Vertex 1
    e1 = v2 - v0  # Vertex 0 to Vertex 2
   
    l = v0 - point
    a = np.dot(e0, e0)
    b = np.dot(e0, e1)
    c = np.dot(e1, e1)
    d = np.dot(e0, l)
    e = np.dot(e1, l)
    f = np.dot(l, l)

    delta = a*c - b*b
    s = b*e - c*d
    t = b*d - a*e

    if s+t <= delta:
        if s < 0:

            if t < 0:  # Region 4
                if d < 0:
                    t = 0
                    if -d >= a:
                        s = 1
                        dist2 = a + 2*d + f
                    else:
                        s = -d/a
                        dist2 = d*s + f
                elif e >= 0:
                    s = 0
                    t = 0
                    dist2 = f
                elif -e >= c:
                    s = 0
                    t = 1
                    dist2 = c + 2*e + f
                else:
                    s = 0
                    t = -e/c
                    dist2 = e*t + f

            else:  # Region 3
                s = 0
                if e >= 0:
                    t = 0
                    dist2 = f
                elif -e >= c:
                    t = 1
                    dist2 = c + 2*e + f
                else:
                    t = -e/c
                    dist2 = e*t + f

        elif t < 0:  # Region 5
            t = 0
            if d >= 0:
                s = 0
                dist2 = f
            elif -d >= a:
                s = 1
                dist2 = a + 2*d + f
            else:
                s = -d/a
                dist2 = d*s + f

        else:  # Region 0
            invDelta = 1/delta
            s = s * invDelta
            t = t * invDelta
            dist2 = s*(a*s + b*t + 2*d) + t*(b*s + c*t + 2*e) + f

    elif s < 0:  # Region 2
        tmp0 = b + d
        tmp1 = c + e
        if tmp1 > tmp0:
            # Min on edge s+t==1
            numer = tmp1 - tmp0
            denom = a - 2*b + c
            if numer >= denom:
                s = 1
                t = 0
                dist2 = a + 2*d + f
            else:
                s = numer/denom
                t = 1-s
                dist2 = s*(a*s + b*t + 2*d) + + t*(b*s + c*t + 2*e) + f
        else:
            # Min on edge s=0
            s = 0
            if tmp1 <= 0:
                t = 1
                dist2 = c + 2*e + f
            elif e >= 0:
                t = 0
                dist2 = f
            else:
                t = -e/c
                dist2 = e*t + f

    elif t < 0:  # Region 6
        tmp0 = b + e
        tmp1 = a + d
        if tmp1 > tmp0:
            numer = tmp1 - tmp0
            denom = a - 2*b + c
            if numer >= denom:
                t = 1
                s = 0
                dist2 = c + 2*e + f
            else:
                t = numer/denom
                s = 1 - t
                dist2 = s*(a*s + b*t + 2*d) + t*(b*s + c*t + 2*e) + f
        else:
            t = 0
            if tmp1 <= 0:
                s = 1
                dist2 = a + 2*d + f
            elif d >= 0:
                s = 0
                dist2 = f
            else:
                s = -d/a
                dist2 = d*s + f

    else:  # Region 1
        numer = c + e - b - d
        if numer <= 0:
            s = 0
            t = 1
            dist2 = c + 2*e + f
        else: 
            denom = a - 2*b + c
            if numer >= denom:
                s = 1
                t = 0
                dist2 = a + 2*d + f
            else:
                s = numer/denom
                t = 1-s
                dist2 = s*(a*s + b*t + 2*d) + t*(b*s + c*t + 2*e) + f

    if dist2 < 0:  # Rounding errors
        dist = 0.
    else:
        dist = np.sqrt(dist2)

    if with_point:
        nearest = v0 + s*e0 + t*e1
        return dist, nearest
    else:
        return dist

