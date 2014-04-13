#!/usr/bin/env/python
from __future__ import print_function

import os
import sys
import time

import numpy as np

try:
    from mayavi import mlab
except ImportError:
    pass


def dump(image, io):
    dims = image.shape
    voxels = np.transpose(np.nonzero(image))
    print("{0} {1} {2}".format(*dims), file=io)
    for voxel in voxels:
        print("{0} {1} {2}".format(*voxel), file=io)


def load(io):
    dims = map(int, next(io).split())
    image = np.zeros(dims)
    for line in io:
        voxel = tuple(map(int, line.split()))
        image[voxel] = True
    return image


def show(image_axes, image, mesh):
    voxels = np.array(zip(*np.nonzero(image)))
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_axis_off()
    ax.axison = False 
    
    i = image
    import os, time
    

    #surf = ax.plot_trisurf(*mesh.vertices.transpose(), triangles=mesh.faces)
    #surf.set_alpha(.1)
    #plt.show()


def console_show(image, delay=.25):
    for s in image:
        os.system('clear')
        for r in s:
            for c in r:
                print('#' if c else ' ', end="")
            print("")
        time.sleep(delay)


def maya_show(image):
    mlab.contour3d(image)
    mlab.show()

def maya_show_field(u, v, w):
    mlab.quiver3d(u, v, w)
    mlab.show()
    

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        image = load(f)
    if '--maya' in sys.argv:
        maya_show(image)
    elif '--maya-grad' in sys.argv:
        maya_show_field(*np.gradient(image))
    else:
        console_show(image, float(sys.argv[2]))
