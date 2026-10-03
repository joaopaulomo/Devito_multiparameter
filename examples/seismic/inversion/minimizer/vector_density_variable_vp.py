from scipy.optimize import minimize
import numpy as np
from devito import *

# Novo import para densidade variável
from examples.seismic.acoustic.variable_density.wavesolver import (
    VariableDensityAcousticWaveSolver,
)
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Model, Receiver, plot_shotrecord, plot_image, plot_velocity

from examples.seismic.inversion.minimizer.minimizer import AbstractMinimizer

import matplotlib.pyplot as plt
import gc

__all__ = ["VariableDensityAcousticMinimizer"]


class VariableDensityAcousticMinimizer(AbstractMinimizer):
    def __init__(
        self,
        dobs_list,
        geometry,
        src_positions,
        velocity_parameter="vp",
        buoyancy_parameter=None,
    ):
        
        params = [velocity_parameter]
        if buoyancy_parameter is not None:
            params.append(buoyancy_parameter)
        super().__init__(params)

        self.dobs_list = dobs_list
        self.geometry = geometry
        self.src_positions = src_positions

    @property
    def velocity_property(self):
        return self.parameter_names[0]

    @property
    def buoyancy_property(self):
        return self.parameter_names[1] if len(self.parameter_names) > 1 else None

    def on_procedure_initialization(self, model0):
        model_ref = self.model_converter.reference_model
        
        self.solver = VariableDensityAcousticWaveSolver(model_ref, self.geometry)

    def loss_grad(self, model0: Model):
        grid = self.model_converter.reference_model.grid
        solver: VariableDensityAcousticWaveSolver = self.solver
        geometry = solver.geometry
        space_order = self.model_converter.reference_model.space_order
        src_positions = self.src_positions

        v_prop = self.velocity_property
        b_prop = self.buoyancy_property
        invert_b = b_prop is not None

       
        grad = {
            v_prop: Function(
                name="grad" + v_prop.upper() + "full",
                grid=grid,
                space_order=space_order,
                staggered=NODE,
            ),
        }
        if invert_b:
            grad[b_prop] = Function(
                name="grad" + b_prop.upper() + "full",
                grid=grid,
                space_order=space_order,
                staggered=NODE,
            )

        grad_v = grad[v_prop]
        grad_b = grad[b_prop] if invert_b else None

        loss = 0
        for isrc in range(src_positions.shape[0]):
            residual = Receiver(
                name="residual",
                grid=grid,
                time_range=geometry.time_axis,
                coordinates=geometry.rec_positions,
            )

            geometry.src_positions[0, :] = src_positions[isrc, :]

            dobs = self.dobs_list[isrc]
            
            dsyn, u0, _ = solver.forward(model=model0, save=True)

            self._compute_adjoint_source(residual, dobs, dsyn)
            loss += 0.5 * norm(residual) ** 2

            
            (dgrad_m, dgrad_b), _ = solver.gradient(rec=residual, u=u0, model=model0)

            del u0
            gc.collect()

            if invert_b:
                Operator([Eq(grad_v, grad_v + dgrad_m), Eq(grad_b, grad_b + dgrad_b)])()
            else:
                Operator([Eq(grad_v, grad_v + dgrad_m)])()

            del dgrad_m
            del dgrad_b

            gc.collect()

        print(f"Current Loss: {loss}")
        Operator(Eq(grad_v, -2 / model0.vp**3 * grad_v))()
        if invert_b:
            Operator(Eq(grad_b, -1 / model0.b**2 * grad_b))()

        return loss, grad

    def _compute_adjoint_source(self, residual, dobs, dsyn):
        for i in range(dobs.shape[1]):
            scale = np.max(np.abs(dsyn.data[:, i]))
            scale = max(scale, np.max(np.abs(dobs.data[:, i]))) + 1e-9

            num = (dsyn.data[:, i] / scale) @ (dobs.data[:, i] / scale)
            den = (dsyn.data[:, i] / scale) @ (dsyn.data[:, i] / scale) + 1e-9
            rho = num / den

            residual.data[:, i] = rho * dsyn.data[:, i] - dobs.data[:, i]
