#!/usr/bin/env python

import numpy as np


class WrappedArray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        wrapper = kwargs.get('wrapper', cls)
        obj = np.ndarray(*args, **kwargs).view(type=cls)
        obj.wrapper = wrapper
        return wrapper

    def __getattr__(self, key):
        try:
            return self[key].view(type=cls)
        except IndexError:
            raise AttributeError("Invalid attribute {0}".format(key))

    def __setattr__(self, key, value):
        try:
            return self[key] = value
        except IndexError:
            raise AttributeError("Invalid attribute {0}".format(key))


class ArrayElement(np.ndarray):
    def __getitem__(self, *args):
        
        
    

class ArrayType(np.ndarray):
    # This is just a place holder
    DTYPE = np.float
    WRAPPER = ArrayElement
 
    def __new__(cls, values=None, *args, **kwargs):
        dtype = kwargs.get('dtype', cls.DTYPE)
        wrapper = kwargs.get('wrapper', cls.WRAPER)

        if values == None:
            if len(args) == 0 and 'shape' not in kwargs:
                kwargs['shape'] = (1,)
            obj = np.zeros(dtype=dtype, *args, **kwargs)
        else:
            obj = np.asarray(values, dtype=dtype, *args, **kwargs)

        obj.wrapper = wrapper
        return obj
    
    def __getitem__(self, *args):
        return np.ndarray.__getitem__(*args).view(type=self.wrapper)


