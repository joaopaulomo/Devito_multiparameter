from typing import List, Tuple
import abc
import gc
from scipy.optimize import minimize

import numpy as np
from devito import Function, VectorFunction, norm, NODE
from examples.seismic import Receiver

from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import setup_geometry, Model, plot_shotrecord

from examples.seismic.inversion.data_source.seismic_data_source import (
    AbstractSeismicDataSource,
)
from examples.seismic.inversion.minimizer.minimizer import AbstractMinimizer
from devito import Operator, Eq

__all__ = ["AcousticMinimizer"]


class AcousticMinimizer(AbstractMinimizer):
    def __init__(
        self, dobs_datasource: AbstractSeismicDataSource, velocity_property="vp"
    ):
        super().__init__([velocity_property])
        self.dobs_datasource = dobs_datasource

    @property
    def velocity_property(self):
        return self.parameter_names[0]

    def on_procedure_initialization(self, model0):
        geometry = self.dobs_datasource.geometry()
        self.solver = AcousticWaveSolver(self.model_converter.reference_model, geometry)

        if self.velocity_property == "vp":

            def _correct_vp_grad(grad):
                grad["vp"].data[:] *= -2 / grad["attr_vp"].data ** 3
                return grad

            # self.grad_postprocess_pipeline.append(_correct_vp_grad)

    def loss_grad(self, model0):
        grid = self.model_converter.reference_model.grid
        solver: AcousticWaveSolver = self.solver
        geometry = solver.geometry
        space_order = self.model_converter.reference_model.space_order
        src_positions = self.dobs_datasource.src_positions

        grad = {
            self.velocity_property: Function(
                name="gradfull", grid=grid, space_order=space_order, staggered=NODE
            ),
            # f'illum_{self.velocity_property}': Function(name="illumfull", grid=grid, space_order=space_order, staggered=NODE),
            f"attr_{self.velocity_property}": getattr(model0, self.velocity_property),
        }

        grad_v = grad[self.velocity_property]
        # illum = grad[f'illum_{self.velocity_property}']

        loss = 0

        for isrc in range(self.dobs_datasource.nshots):
            residual = Receiver(
                name="residual",
                grid=grid,
                time_range=geometry.time_axis,
                coordinates=geometry.rec_positions,
            )

            # dillum = Function(name="dillum", grid=grid, space_order=space_order, staggered=NODE)

            # Update source location
            geometry.src_positions[0, :] = src_positions[isrc, :]

            dobs = self.dobs_datasource.get_data(isrc)
            dsyn, u0, _ = solver.forward(model=model0, save=True)

            # Compute smooth data and full forward wavefield u0
            # Compute gradient from data residual and update objective function
            # self._compute_adjoint_source(residual, dobs, dsyn)
            residual.data[:] = dsyn.data[:] - dobs.data[:]

            loss += 0.5 * norm(residual) ** 2
            dgrad, _ = solver.gradient(rec=residual, u=u0, model=model0)
            # self._compute_illumination(dillum, u0, model0)

            Operator(Eq(grad_v, grad_v + dgrad))()
            # Operator(Eq(illum, illum + dillum))()
            del u0
            del dgrad
            gc.collect()

        print(loss)
        return loss, grad

    # def _compute_illumination(self, illum, u, model):
    # dt = model.critical_dt
    # Operator([Eq(illum, illum + u.dt2 * u.dt2)])(dt=dt)
    # return illum

    # def _compensate_illumination(self, grad: Function, illum: Function):
    # Operator([Eq(grad, grad / (illum + 1e-9))])()
    # return grad

    def _compute_adjoint_source(self, adjoint_source, dobs, dsyn):
        if adjoint_source.grid.distributor.is_parallel:
            # If we run with MaPI, we have to compute the residual via an operator
            # First make sure we can take the difference and that receivers are at the
            # same position
            assert np.allclose(dobs.coordinates.data[:], dsyn.coordinates.data)
            assert np.allclose(
                adjoint_source.coordinates.data[:], dsyn.coordinates.data
            )
            # Create a difference operator
            diff_eq = Eq(
                adjoint_source,
                dsyn.subs({dsyn.dimensions[-1]: adjoint_source.dimensions[-1]})
                - dobs.subs({dobs.dimensions[-1]: adjoint_source.dimensions[-1]}),
            )
            Operator(diff_eq)()
        else:
            # A simple data difference is enough in serial
            adjoint_source.data[:] = dsyn.data[:] - dobs.data[:]
