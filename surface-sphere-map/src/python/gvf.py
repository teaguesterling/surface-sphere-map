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


def _explicit_gvf_step(u, B, r, C, dt):
    return (1 - B * dt) * u + r * convolve(u, **DISCRETE_LAP_3D) + C * dt


def _explicit_gvf_step(u, B, r, C):
    return (1 - B) * u + r * laplace(u) + C



def explicit_gvf3d(f, k=None, mu=0.2, dt=.5, d=(1,1,1), epsilon=None):
    N = reduce(operator.mul, f.shape)
    if k is None:
        k = int((1/dt) * sqrt(N))
    if epsilon is None:
        epsilon = N * 10e-5
    dx, dy, dz = d
    F = gradient(f)
    Fx, Fy, Fz = F
    B = Fx**2 + Fy**2 + Fz**2
    Bdt = B * dt
    Cx, Cy, Cz = F * Bdt
    r = (mu * dt) / (dx * dy * dz)
    R = r * np.ones(f.shape)
    u0, v0, w0 = Fx, Fy, Fz
    delta0 = np.inf
    diverge = 0
    for i in range(k):
        u = (1-B)*u0 + R*laplace(u0) + Cx
        v = (1-B)*v0 + R*laplace(v0) + Cz
        w = (1-B)*w0 + R*laplace(w0) + Cy
        delta = np.sum(abs(u-u0) + abs(v-v0) + abs(w-w0))
        u0, v0, w0 = u, v, w
        if delta <= epsilon:
            break
        elif delta >= delta0:
            diverge += 1
        if diverge > 5:
            raise ValueError("Divergence detected. Use smaller value of dt")
        else:
            delta0 = delta
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
