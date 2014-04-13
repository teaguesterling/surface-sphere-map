#!/usr/bin/env python
from __future__ import print_function, division

from collections import defaultdict
import operator
import itertools

import numpy as np
from spatial import (
    Surface,
    triangle_normal,
    nearest_triangle,
)
from lattice import PseudoGrid
import image3d

from del2 import del2


def voxelize_mesh(mesh, resolution=1., fill=True):
    axes = PseudoGrid.from_extents(mesh.extents, resolution=resolution,
                                                 stretch=1)
    e = axes.extents
    image = axes.get_array(dtype=bool)
    voxel_triangles = defaultdict(list)
    for i, triangle in enumerate(mesh.triangles):
        normal = triangle_normal(triangle)
        voxels = list(axes.get_bounding_voxels(triangle))
        for voxel in voxels:
            if axes.triangle_intersects_voxel(triangle, voxel, 
                                                normal=normal):
                image[voxel] = True
                voxel_triangles[voxel].append(triangle)
    if fill:
        for i, slice in enumerate(image):
            for j, row in enumerate(slice):
                filling = False
                peek = zip(row, row[1:])
                for k, (cell, next_cell) in enumerate(peek):
                    if cell and not next_cell:
                        voxel = (i,j,k)
                        voxel_corners = axes.get_voxel_corners((i,j,k))
                        cell_axis = voxel_corners[1] - voxel_corners[0]
                        voxel_center = voxel_corners.mean(axis=0)
                        voxel_tris = voxel_triangles[voxel]
                        far_point = voxel_center + cell_axis / 2
                        _d, far_triangle, _p = nearest_triangle(far_point, voxel_tris)
                        far_normal = triangle_normal(far_triangle)
                        direction = np.dot(cell_axis, far_normal)
                        if direction < 0:
                            filling = True
                        elif direction >= 0:
                            filling = False
                        if np.count_nonzero(row[k+1:]) == 0:
                            filling = False
                    elif filling:
                        image[i,j,k] = True
                
    return axes, image


def gradient_field(image):
    field = make_field(*np.gradient(image), shape=image.shape)
    return field


def gvf(image, iterations=None):
    if iterations is None:
        iterations = int(np.sqrt(reduce(operator.mul, image.shape)))
    image = _mirror_boundary(image)
    
    # Image gradient
    Gx, Gy, Gz = np.gradient(image)
    G2 = Gx**2 + Gy**2 + Gz**2
    
    # Initial field (Gradient)
    u, v, w = Gx, Gy, Gz

    for i in range(iterations):
        u = _mirror_boundary(u)
        v = _mirror_boundary(v)
        w = _mirror_boundary(w)

        u += 6 * del2(u) - (u - Gy) * G2
        v += 6 * del2(v) - (v - Gy) * G2
        w += 6 * del2(w) - (w - Gz) * G2
        
    return u, v, w 


def _mirror_boundary(f):
    N, M,  O = f.shape

    return f

    f = np.array(f)

    X = np.arange(2,M-1)
    Y = np.arange(2,N-1)
    Z = np.arange(2,O-1)
    
    
    
    f[0], f[N] = f[2], f[:-1]
    

def make_field(x, y, z, shape=None):
    if shape is None:
        shape = x.shape
    size = reduce(operator.mul, list(shape))
    vals = np.array([x, y, z])
    vectors = np.asarray(zip(*map(np.ndarray.flatten, vals)))
    field = vectors.reshape(shape + [len(shape)])
    return field



def test(path, res=1, fill=False):
    res = float(res)
    fill = bool(fill)
    with open(path) as f:
        surface = Surface.from_vet_file(f)
    axes, image = voxelize_mesh(surface, resolution=res, fill=fill)
    with open("{0}.image".format(path), 'w') as f:
        image3d.dump(image, f)
    image3d.console_show(image, .25)
    image3d.maya_show_field(*np.gradient(image))
     


if __name__ == '__main__':
    import sys
    test(*sys.argv[1:])
