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


def square_up_image(I, padding=1, mode='constant', constant_values=0):
    x, y, z = (np.max(I.shape) - I.shape + padding) / 2
    extend = (x, x), (y, y), (z, z)
    padded = np.pad(I, extend, mode=mode, constant_values=constant_values)
    return padded, extend


def voxelize_mesh(mesh, resolution=1.0, 
                        cube=True, 
                        padding=5, 
                        fill=False, 
                        triangle_map=False):
    axes = PseudoGrid.from_extents(mesh.extents, resolution=resolution,
                                                 padding=padding,
                                                 cube=cube)
    image, tri_voxels = voxelize_mesh_boundary(mesh, axes, resolution=resolution,
                                                           triangle_map=triangle_map)
    if fill:
        image = fill_voxelized_mesh(image, axes, tri_voxels)
    if triangle_map:
        return image, axes, tri_voxels
    else:
        return image, axes


def voxelize_mesh_boundary(mesh, axes, resolution=1.0, triangle_map=False):
    image = axes.get_array(dtype=bool)
    triangle_voxels = defaultdict(list)
    for i, triangle in enumerate(mesh.triangles):
        normal = triangle_normal(triangle)
        voxels = list(axes.get_bounding_voxels(triangle))
        for voxel in voxels:
            if not triangle_map and image[voxel]:  # Skip existing voxels
                continue
            elif axes.triangle_intersects_voxel(triangle, voxel, 
                                                normal=normal):
                image[voxel] = True
                if triangle_map:
                    triangle_voxels[voxel].append(i)
    return image, triangle_voxels
        

def fill_voxelized_mesh(image, axes, triangle_voxels):
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
                    voxel_tris = triangle_voxels[voxel]
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
    return image


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
