from __future__ import division

import contextlib
import operator

import numpy as np
from scipy import (
    gradient,
    sqrt,
)
from scipy.ndimage.filters import (
    convolve,
    laplace,
)
from scipy.ndimage.morphology import distance_transform_edt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

LAPLACIAN_3D_STENCIL = np.array([
    [[0,  0,  0],
     [0,  1,  0],
     [0,  0,  0]],
    [[0,  1,  0],
     [1, -6,  1],
     [0,  1,  0]],
    [[0,  0,  0],
     [0,  1,  0],
     [0,  0,  0]]
])
DISCRETE_LAP_3D = {
    'weights': LAPLACIAN_3D_STENCIL,
    #'mode': 'constant', 
    #'cval': 0
}


def reference_gvf3d(f, iter=None, mu=0.2, dt=.5, ds=(1,1,1), epsilon=None):
    N = reduce(operator.mul, f.shape)
    if iter is None:
        iter = int(1/dt * N**(1/3))
    if epsilon is None:
        epsilon = N * 10e-5
    dx, dy, dz = ds
    F = gradient(f, *ds)
    Fx, Fy, Fz = F
    B = Fx*Fx + Fy*Fy + Fz*Fz
    Bdt = B * dt
    Cx, Cy, Cz = F * Bdt
    r = (mu * dt) / (dx * dy * dz)
    R = r * np.ones(f.shape)
    u0, v0, w0 = Fx, Fy, Fz
    delta0 = np.inf
    diverge = 0
    for i in range(iter):
        u = (1-Bdt)*u0 + R*laplace(u0) + Cx
        v = (1-Bdt)*v0 + R*laplace(v0) + Cy
        w = (1-Bdt)*w0 + R*laplace(w0) + Cz
        delta = np.sum(abs(u-u0) + abs(v-v0) + abs(w-w0))
        u0, v0, w0 = u, v, w
        if delta <= epsilon:
            break
        elif delta >= delta0:
            diverge += 1
        if diverge > 5:
            raise ValueError("Divergence detected for 5 consecutive steps. Use smaller value of dt")
        else:
            delta0, diverge = delta, 0
    return u0, v0, w0


def reference_ggvf3d(f, iter=None, K=0.1, dt=1, ds=(1,1,1), epsilon=None):
    N = reduce(operator.mul, f.shape)
    if iter is None:
        iter = int(1/dt * sqrt(N))
    if epsilon is None:
        epsilon = N * 10e-5
    dx, dy, dz = d
    F = gradient(f)
    Fx, Fy, Fz = F
    M = np.sqrt(Fx*Fx + Fy*Fx + Fz*Fz)
    g = np.exp(-(M/K))
    h = 1 - g
    Gdt = (g * dt) / (dx * dy * dz)
    Hdt = 1 - Gdt
    Hx, Hy, Hz = h * Fx, h * Fy, h * Fz
    u0, v0, w0 = Fx, Fy, Fz
    delta0 = np.inf
    diverge = 0
    for i in range(iter):
        u = Gdt*laplace(u0) - Hdt*u0 + Hx
        v = Gdt*laplace(v0) - Hdt*v0 + Hy
        w = Gdt*laplace(w0) - Hdt*w0 + Hz
        delta = np.sum(abs(u-u0) + abs(v-v0) + abs(w-w0))
        u0, v0, w0 = u, v, w
        if delta <= epsilon:
            break
        elif delta >= delta0:
            diverge += 1
        if diverge > 5:
            raise ValueError("Divergence detected for 5 consecutive steps. Use smaller value of dt")
        else:
            delta0, diverge = delta, 0
    return u0, v0, w0
    

def make_field_array(x, y, z, shape=None):
    if shape is None:
        shape = x.shape
    shape = list(shape)
    size = reduce(operator.mul, list(shape))
    vals = np.array([x, y, z])
    vectors = np.asarray(zip(*map(np.ndarray.flatten, vals)))
    field = vectors.reshape(shape + [len(shape)])
    return field


def make_field_array(*components):
    C = np.array(components)
    D = len(C.shape)
    F = np.rollaxis(C, 0, D)
    return F


