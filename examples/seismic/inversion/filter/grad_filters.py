from devito import Operator, Eq, Function

import numpy as np


__all__ = [ 
    'MaskFilter', 
    'IlluminationCompensationFilter', 
    'FirstDerivativeNormalizationFilter',
    'DropKeysFilter'
]

class MaskFilter():
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, grad):
        for k, g in grad.items():
            g.data[:] *= self.mask
        return grad


class IlluminationCompensationFilter():
    def __init__(self, illum_prefix='illum_'):
        self.prefix = illum_prefix

    def __call__(self, grad: dict):
        illum_keys = []
        for k, illum in grad.items():
            if k.startswith(self.prefix):
                prop = k[len(self.prefix):]
                if prop in grad:
                    self.compensate_illumination(grad[prop], illum)
                    illum_keys.append(k)
        # for k in illum_keys:
        #     grad.prop(k)
        return grad

    def compensate_illumination(self, grad: Function, illum: Function):
        Operator([Eq(grad, grad/(illum + 1e-9))])()
        return grad


class FirstDerivativeNormalizationFilter():
    def __init__(self, parameter_names, slice=None):
        self._first_iteration = True
        self.alpha0 = {}
        self.slice = slice
        self.parameter_names = parameter_names
    
    def reset(self):
        self._first_iteration = True
    
    def __call__(self, grad):
        if self._first_iteration:
            self._first_iteration = False
            for k, g in grad.items():
                if k in self.parameter_names:
                    if self.slice is not None:
                        self.alpha0[k] = np.abs(g.data[self.slice]).max()
                    else:
                        self.alpha0[k] = np.abs(g.data).max()

        for k, g in grad.items():
            if k in self.parameter_names:
                g.data[:] *= 1/self.alpha0[k]
        return grad
    

class DropKeysFilter():
    def __init__(self, keys: list):
        self.keys = keys
    
    def __call__(self, grad: dict):
        for k in self.keys:
            grad.pop(k, None)

        return grad