from __future__ import division


import itertools
import operator

import numpy as np
from numpy.linalg import norm


from spatial import (
    triangle_normal,
    project_with_extrema,
)

# Sloppy way to do this, but easy to understand

# Only one step
six_adjacency = adjacency_6 = [(i,j,k) for i in (-1,0,1) 
                                       for j in (-1,0,1) 
                                       for k in (-1,0,1) 
                                       if not (i == j == k == 0) 
                                       and sum(map(abs, (i, j, k))) <= 1]
# Max two steps
eighteen_adjacency = adjacency_6 = [(i,j,k) for i in (-1,0,1) 
                                            for j in (-1,0,1) 
                                            for k in (-1,0,1) 
                                            if not (i == j == k == 0) 
                                            and sum(map(abs, (i, j, k))) <= 2]
# All steps
twentysix_adjacency = adjacency_6 = [(i,j,k) for i in (-1,0,1) 
                                             for j in (-1,0,1) 
                                             for k in (-1,0,1) 
                                             if not (i == j == k == 0)]

unit_cube_vertices = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                      (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

unit_cube_faces = [(0, 1, 2, 3), (4, 5, 6, 7),  # XY Faces
                   (0, 1, 4, 5), (2, 3, 6, 7),  # XZ Faces
                   (0, 2, 4, 6), (1, 3, 5, 7)]  # YZ Faces

UNIT_AXES = np.array([[1,0,0],[0,1,0],[0,0,1]])


def pw_sum(u, v):
    """ Tuple pair-wise sum """
    return tuple(np.array(u) + np.array(v))


class PseudoGrid(list):
    WRAPPER = np.array

    def __init__(self, *args, **kwargs):
        super(PseudoGrid, self).__init__(*args)
        self._resolution = kwargs.get('resolution')
        self._array = None

    def __getitem__(self, coords):
        point = [dim[coord] for coord, dim in zip(coords, self)]
        return self.WRAPPER(point)

    def axis(self, i):
        return super(PseudoGrid, self).__getitem__(i)

    def position_on_axes(self, point):
        low = np.array(map(operator.itemgetter(0), self.extents))
        res = np.array(self._resolution)
        idx = tuple(np.trunc((point - low) / res).astype(np.int))
        return idx 

    def map_to_axes(self, points):
        low = np.array(map(operator.itemgetter(0), self.extents))
        res = self._resolution
        indicies = np.trunc((points - low) / res).astype(np.int)
        return indicies

    def all_indicies_in_grid(self, indices):
        limits = np.array(map(len, self))
        return np.all(indicies >= 0) and np.all(indicies < limits)

    def all_points_in_grid(self, points):
        lower, upper = np.array(self.extents).T
        return np.all(points <= upper) and np.all(points >= lower)

    def nearest(self, point):
        index = self.position_on_axes(point)
        deltas = self[index] - point
        nearest_index = index - (deltas > .5)
        print nearest_index, index
        return nearest_index

    def point_in_voxel(self, point, voxel):
        bounds = self.get_voxel_bounds(voxel)
        return all(low <= dim <= high for dim, (low, high) in zip(point, bounds))

    def triangle_intersects_voxel(self, triangle, voxel, normal=None):
        # Separating line theorm
        # http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/tribox.pdf

        # Translate to Origin
        vertices = self.get_voxel_corners(voxel)
        #voxel_center = vertices.mean(axis=0)

        bounds = np.array(self.get_voxel_bounds(voxel))

        #bounds_translated = bounds - np.array([voxel_center, voxel_center]).T
        #vertices_translated = vertices - voxel_center
        #triangle_translated = triangle - voxel_center

        along_dims = triangle.transpose()
        extents = [(np.min(v), np.max(v)) for v in along_dims]
        dim_tests = zip(extents, bounds)

        # (0) If a vertex is inside the bounding box
        # There is an intersection
        for vertex in triangle:
            if all(l <= pt <= h for pt, (l, h) in zip(vertex, bounds)):
                return True

        # (1)
        # This is normally performed in the voxel extraction
        if any(h < lb or l > hb for (l, h), (lb, hb) in dim_tests):
            return False
        
        # (2)
        # Test Triangle Normal
        if normal is None:
            normal = triangle_normal(triangle)
        tri_offset = np.dot(normal, triangle[0])
        projected, low, high = project_with_extrema(vertices, normal)
        if high < tri_offset or low > tri_offset:
            return False

        # (3)
        # Test edge cross products
        a, b, c = triangle
        edges = np.array([a - b, b - c, c - a])
        for i, j in itertools.product(edges, UNIT_AXES):
            axis = np.cross(i, j)
            box_proj, box_low, box_high = project_with_extrema(vertices, axis)
            tri_proj, tri_low, tri_high = project_with_extrema(triangle, axis)
            if box_low >= tri_high or box_high <= tri_low:
                return False

        return True
        

#unit_cube_faces = [(0, 1, 2, 3), (4, 5, 6, 7),  # Z Faces
#                   (0, 1, 4, 5), (2, 3, 6, 7),  # Y Faces
#                    (0, 2, 4, 6), (1, 3, 5, 7)]  # X Faces

    def get_voxel_index(self, point):
        "Voxels are indexed by the lowest coordinant on each axis"
        return tuple(idx - 1 for idx in self.position_on_axes(point))

    def get_voxel_corner_indicies(self, index):
        corners = [pw_sum(index, delta) for delta in unit_cube_vertices]
        return corners

    def get_voxel_face_indicies(self, index):
        corners = self.get_voxel_corner_indicies(index)
        faces = [tuple(corners[idx] for idx in face)
                                    for face in unit_cube_faces]
        return faces

    def get_voxel_bounds(self, index):
        by_axes = zip(self, index)
        bounds = [(axis[idx], axis[idx+1]) for axis, idx in by_axes]
        return bounds

    def get_voxel_corners(self, index):
        corners = self.get_voxel_corner_indicies(index)
        return np.array(map(self.__getitem__, corners))

    def get_voxel_faces(self, index):
        indicies = self.get_voxel_face_indicies(index)
        return [map(self.__getitem__, face_indicies) for face_indicies in indicies]

    def get_voxel(self, point):
        index = self.get_voxel_index(point)
        corners = self.get_voxel_corners(index)
        faces = self.get_voxel_faces(index)
        return faces, corners

    def get_bounding_box_indicies(self, points):
        extents = np.array([points.min(axis=0), points.max(axis=0)])
        index_bounds = [self.position_on_axes(dim) for dim in extents]
        return index_bounds

    def get_bounding_voxels(self, points):
        bounds = self.get_bounding_box_indicies(points)
        index_ranges = [range(l, h+1) for l, h in zip(*bounds)]
        voxels = itertools.product(*index_ranges)
        return voxels

    def quantize(self, point):
        index = self.nearest(point)
        coords = self[index]
        return coords

    def enumerate(self):
        for idx in np.ndindex(self.shape):
            yield idx, self[idx]

    def enumerate_over(self, points):
        for point in points:
            index = self.nearest(point)
            coords = self[index]
            yield index, coords

    @property
    def extents(self):
        return map(lambda d: (min(d), max(d)), self)

    @property
    def shape(self):
        return tuple(map(len, self))

    @property
    def resolution(self):
        if self._resolution is not None:
            return self._resolution
        else:
            n_dims = len(self)
            zeros = [0] * n_dims
            ones = [1] * n_dims
            return abs(self[ones] - self[zeros])

    def get_array(self, **kwargs):
        return np.zeros(self.shape, **kwargs)

    @classmethod
    def from_extents(cls, extents, resolution=1, padding=None, cube=False):
        dims = []
        if not isinstance(resolution, (list, tuple)):
            resolution = [resolution] * len(extents)
        extents = np.apply_along_axis(make_bounds, 1, extents)
        if padding is not None:
            extents = np.apply_along_axis(lambda a: stretch_bounds(a, steps=padding), 1, extents)
        if cube:
            cells = [h-l for l,h in extents]
            max_cells = max(cells)
            square_up = [(max_cells - num_cells) for num_cells in cells]
            extents = [stretch_bounds(bounds, steps=steps, bottom=False) for bounds, steps in zip(extents, square_up)]
        for idx, (low, high) in enumerate(extents):
            dims.append(np.arange(low, high, resolution[idx]))
        return cls(dims, resolution=resolution)


def make_bounds(extents):
    low, high = extents
    return np.floor(low), np.ceil(high)


def stretch_bounds(extents, steps=1, top=True, bottom=True):
    low, high = extents
    if top:
        high += steps
    if bottom:
        low -= steps
    return low, high


def enumerate_gridpoints(dims):
    for coords in np.ndindex(*map(len, dims)):
        point = np.array([dim[pos] for pos, dim in zip(coords, dims)])
        yield coords, point




def rotation_matrix(axis,theta):
    axis = axis/norm(axis)
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def adjacent_indicies(index, shape=None, step=1):
    index = tuple(index)
    n_dims = len(index)
    if shape is None:
        shape = [np.inf] * n_dims
    if isinstance(step, (int, np.int)):
        step = [step] * n_dims
    step = np.array(step, dtype=int)
    def gen(idx):
        for dim, offset in enumerate(idx):
            left, right = offset - 1, offset + 1
            if left >= 0:
                yield tuple(idx[0:dim] + (left,) + idx[dim+1:])
            if right < shape[dim]:
                yield tuple(idx[0:dim] + (right,) + idx[dim+1:])
    if np.any(step > 1):
        unique = set()
        for neighbor in adjacent_indicies(index, shape, step-1):
            next_neighbors = gen(neighbor)
            unique.update(next_neighbors)
        return list(unique)
    else:
        return list(gen(index))


def adjacent(index, shape=None, connectivity=6):
    index = np.array(index)
    n_dims = len(index)
    if connectivity == 6:
        adjacency = six_adjacency
    elif connectivity == 18:
        adjacency = eighteen_adjacency
    elif connectivity == 26:
        adjacency = twentysix_adjacency
    else:
        raise ValueError("No such thing as {} adjacnecy".format(connectivity))
    left = np.zeros(len(index))
    if shape is None:
        shape = [np.inf] * n_dims
    right = np.array(shape)
    neighbors = []
    for delta in adjacency:
        neighbor = index + delta
        if np.all(left <= neighbor) and np.all(neighbor < right):
            neighbors.append(neighbor)
    return np.array(neighbors)


