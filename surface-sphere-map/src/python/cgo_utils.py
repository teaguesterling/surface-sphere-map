'''
Square and Tetrahedra representations

(c) 2013 Thomas Holder

License: BSD-2-Clause
'''

from pymol import cgo

def cgo_cube(x, y, z, r):
    r *= 3**-.5
    return [
       cgo.BEGIN, cgo.TRIANGLE_STRIP,
       cgo.NORMAL,  0.,  0.,  1.,
       cgo.VERTEX, x+r, y+r, z+r,
       cgo.VERTEX, x+r, y-r, z+r,
       cgo.VERTEX, x-r, y+r, z+r,
       cgo.VERTEX, x-r, y-r, z+r,
       cgo.END,
       cgo.BEGIN, cgo.TRIANGLE_STRIP,
       cgo.NORMAL,  1.,  0.,  0.,
       cgo.VERTEX, x+r, y-r, z-r,
       cgo.VERTEX, x+r, y+r, z-r,
       cgo.VERTEX, x+r, y-r, z+r,
       cgo.VERTEX, x+r, y+r, z+r,
       cgo.END,
       cgo.BEGIN, cgo.TRIANGLE_STRIP,
       cgo.NORMAL,  0.,  1.,  0.,
       cgo.VERTEX, x+r, y+r, z-r,
       cgo.VERTEX, x-r, y+r, z-r,
       cgo.VERTEX, x+r, y+r, z+r,
       cgo.VERTEX, x-r, y+r, z+r,
       cgo.END,
       cgo.BEGIN, cgo.TRIANGLE_STRIP,
       cgo.NORMAL,  0.,  0., -1.,
       cgo.VERTEX, x-r, y-r, z-r,
       cgo.VERTEX, x-r, y+r, z-r,
       cgo.VERTEX, x+r, y-r, z-r,
       cgo.VERTEX, x+r, y+r, z-r,
       cgo.END,
       cgo.BEGIN, cgo.TRIANGLE_STRIP,
       cgo.NORMAL, -1.,  0.,  0.,
       cgo.VERTEX, x-r, y+r, z+r,
       cgo.VERTEX, x-r, y-r, z+r,
       cgo.VERTEX, x-r, y+r, z-r,
       cgo.VERTEX, x-r, y-r, z-r,
       cgo.END,
       cgo.BEGIN, cgo.TRIANGLE_STRIP,
       cgo.NORMAL,  0., -1.,  0.,
       cgo.VERTEX, x-r, y-r, z+r,
       cgo.VERTEX, x+r, y-r, z+r,
       cgo.VERTEX, x-r, y-r, z-r,
       cgo.VERTEX, x+r, y-r, z-r,
       cgo.END,
   ]
