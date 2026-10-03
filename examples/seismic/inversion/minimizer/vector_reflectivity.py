from scipy.optimize import minimize
import numpy as np
from devito import *

from examples.seismic.vector_reflectivity import VectorReflectivityAcousticWaveSolver
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Model, Receiver, plot_shotrecord, plot_image, plot_velocity

from examples.seismic.inversion.minimizer.minimizer import AbstractMinimizer

import matplotlib.pyplot as plt

import gc

__all__ = ['VectorReflectivityAcousticMinimizer']

class VectorReflectivityAcousticMinimizer(AbstractMinimizer):
    def __init__(self, dobs_list, geometry, src_positions, velocity_parameter='m'):
        super().__init__([velocity_parameter, 'r_x', 'r_y'])
        self.dobs_list = dobs_list
        self.geometry = geometry
        self.src_positions = src_positions
        self.grad_postprocess_pipeline.append(self.vector_pullback_filter)

    @property
    def velocity_property(self):
        return self.parameter_names[0]

    def on_procedure_initialization(self, model0):
        model_ref = self.model_converter.reference_model
        self.solver = VectorReflectivityAcousticWaveSolver(model_ref, self.geometry)

        # self.dhw_list = []
        # model_hw_ndarray = self.model_converter.model_to_ndarray(model0)
        # slc = [slice(model0.nbl, -model0.nbl)]*len(model0.shape)
        # if self.velocity_property == 'm':
        #     model_hw_ndarray['m'] = model_ref.m.data[slc].max()
        # elif self.velocity_property == 'vp':
        #     model_hw_ndarray['vp'] = model_ref.vp.data[slc].min()
        # model_hw_ndarray['r_x'] = 0
        # model_hw_ndarray['r_y'] = 0
        # model_hw = self.model_converter.ndarray_to_model(model_hw_ndarray)
        # for isrc in range(self.src_positions.shape[0]):
        #     self.geometry.src_positions[0, :] = self.src_positions[isrc, :]
        #     dhw, _, _ = self.solver.forward(model=model_hw)
        #     self.dhw_list.append(dhw)


    def loss_grad(self, model0: Model):
        grid = self.model_converter.reference_model.grid
        solver: VectorReflectivityAcousticWaveSolver = self.solver
        geometry = solver.geometry
        space_order = self.model_converter.reference_model.space_order
        src_positions = self.src_positions

        grad = {
            # 'm': Function(name="gradMfull", grid=grid, space_order=space_order, staggered=NODE),
            self.velocity_property: Function(name="grad" + self.velocity_property.upper() + "full", grid=grid, space_order=space_order, staggered=NODE),
            'r': VectorFunction(name="gradRfull", grid=grid, space_order=space_order, staggered=[NODE, NODE]),
            # 'illum_m': Function(name="illumMfull", grid=grid, space_order=space_order, staggered=NODE),
            "illum_" + self.velocity_property: Function(name="illum" + self.velocity_property.upper() + "full", grid=grid, space_order=space_order, staggered=NODE),
            'illum_r': VectorFunction(name="illumRfull", grid=grid, space_order=space_order, staggered=[NODE, NODE]),
        }
        
        # grad_m = grad['m']
        grad_v = grad[self.velocity_property]
        grad_r = grad['r']
        # illum_m = grad['illum_m']
        illum_v = grad["illum_" + self.velocity_property]
        illum_r = grad['illum_r']

        loss = 0
        for isrc in range(src_positions.shape[0]):
            residual = Receiver(name='residual', grid=grid,
                            time_range=geometry.time_axis, 
                            coordinates=geometry.rec_positions)
            

            # Update source location
            geometry.src_positions[0, :] = src_positions[isrc, :]
            
            dobs = self.dobs_list[isrc]
            dsyn, u0, _ = solver.forward(model=model0, save=True)
            # dsyn.data[:] -= self.dhw_list[isrc].data[:]

            self._compute_adjoint_source(residual, dobs, dsyn)
            loss += .5*norm(residual)**2
            dgrad_v, dgrad_r, _ = solver.gradient(rec=residual, u=u0, model=model0)
            dillum_v, dillum_r, _ = solver.illumination(u0, illumVp_or_M=illum_v)

            del u0

            gc.collect()

            Operator([
                Eq(grad_v, grad_v + dgrad_v),
                Eq(grad_r, grad_r + dgrad_r),
                Eq(illum_v, illum_v + dillum_v),
                Eq(illum_r, illum_r + dillum_r)
            ])()

            del dgrad_v
            del dgrad_r
            del dillum_v
            del dillum_r

            gc.collect()

        print(loss)

        return loss, grad

    def compensate_illumination_filter(self, grad):
        # self._compensate_illumination(grad['m'], grad['illum_m'])
        self._compensate_illumination(grad['r'][0], grad['illum_r'][0])
        self._compensate_illumination(grad['r'][1], grad['illum_r'][1])

        grad.pop('illum_' + self.velocity_property)
        grad.pop('illum_r')
        return grad
    
    def vector_pullback_filter(self, grad):
            r = grad['r']
            grad['r_x'] = r[0]
            grad['r_y'] = r[1]
            grad.pop('r')
            return grad
    

    def _compensate_illumination(self, grad: Function, illum: Function):
        Operator([Eq(grad, grad/(illum + 1e-9))])()
        return grad
    

    def _compute_adjoint_source(self, residual, dobs, dsyn):
        for i in range(dobs.shape[1]):
            scale = np.max(np.abs(dsyn.data[:, i]))
            scale = max(
                scale,
                np.max(np.abs(dobs.data[:, i]))
            ) + 1e-9
            # rho = (dsyn.data[:, i]) @ (dobs.data[:, i]) / ((dsyn.data[:, i]) @ (dsyn.data[:, i]))
            rho = (dsyn.data[:, i]/scale) @ (dobs.data[:, i]/scale) / ((dsyn.data[:, i]/scale) @ (dsyn.data[:, i])/scale + 1e-9)
            residual.data[:, i] = rho * dsyn.data[:, i] - dobs.data[:, i]
        # residual.data[:] = dsyn.data[:] - dobs.data[:]
