#!/usr/bin/env python
from __future__ import print_function

from copy import deepcopy
import itertools

import numpy as np


class VertexArray(np.ndarray):
    DTYPE = [
        ('coords', [('x', np.double),
                    ('y', np.double),
                    ('z', np.double),]),
        ('norm', [('x', np.float),
                  ('y', np.float),
                  ('z', np.float),]),
        ('values', np.double, 3),
        ('component', np.int),
        ('atom_id', np.int),
        ('hue', np.int),
    ]
    INDEXED_DTYPE = [('vertex_id', np.int)] + DTYPE
    
    def __new__(cls, values=None, *args, **kwargs):
        kwargs['dtype'] = kwargs.get('dtype', cls.DTYPE)
        if values == None:
            if len(args) == 0 and 'shape' not in kwargs:
                kwargs['shape'] = (1,)
            values = np.zeros(*args, **kwargs)
        else:
            values = np.asarray(values, *args, **kwargs)
        return values

    @classmethod
    def CustomFormat(cls, num_values=3, name=None):
        dtype = deepcopy(cls.DTYPE)
        # Values component is 3rd item
        # Num values is 3rd component of dtype
        dtype[2][2] = num_values
        
        if name is None:
            name = "VertexFns{0}".format(num_values)

        newcls = type(name, (cls,), {'DTYPE': dtype})
        return newcls


def load(io, options={}):
    counts = next(io).split()
    nVertexes, nEdges, nTriangles = map(int, counts[:3])
    if len(counts) > 3 and 'function_values' not in options:
        nFnValues = int(counts[3])
        options['function_values'] = nFnValues
    else:
        nFnValues = 3

    verts = VertexArray(shape=(nVertexes,), dtype=VertexArray.INDEXED_DTYPE)
    for idx, line in enumerate(itertools.islice(io, nVertexes)):
        vertex = parse_vertex_line(line, options)
        vertex.insert(0, idx+1)
        verts[idx] = np.array([tuple(vertex)], dtype=verts.dtype)
    
    return verts
 

def parse_vertex_line(line, options={}):
    fn_values = options.get('function_values', 3)
    tokens = line.split()
    coords = tuple(tokens[0:3])
    norm = tuple(tokens[3:6])
    values = tokens[6:6+fn_values]
    component = tokens[-3]
    atom_id = tokens[-2]
    hue = tokens[-1]
    data = [coords, norm, values, component, atom_id, hue]
    return data

