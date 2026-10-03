from devito import *
from devito.tools import memoized_meth
from examples.seismic import Model, AcquisitionGeometry

from examples.seismic.vector_reflectivity.operators import *


class VectorReflectivityAcousticWaveSolver():
    def __init__(self, model: Model, geometry: AcquisitionGeometry, kernel='OT2', space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry: AcquisitionGeometry = geometry

        assert self.model.grid == geometry.grid

        self.space_order = space_order
        self.kernel = kernel

        # Cache compiler options
        self._kwargs = kwargs


    @property
    def dt(self):
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if self.kernel == 'OT4':
            return self.model.dtype(1.73 * self.model.critical_dt)
        return self.model.critical_dt


    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_adj(self, save=None):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)


    @memoized_meth
    def op_born(self):
        raise NotImplementedError('`BornOperator` was not implemented')
        return BornOperator(self.model, save=None, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True):
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_illum(self):
        return IlluminationOperator(self.model, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)


    def forward(self, src=None, rec=None, u=None, model: Model =None, save=None, **kwargs):
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec
        
        model = model or self.model

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        kwargs.pop('b', None)

        summary = self.op_fwd(save).apply(
            src = src, rec = rec, u = u, dt = kwargs.pop('dt', self.dt), **kwargs
        )

        return rec, u, summary


    def adjoint(self, rec, srca=None, v=None, model=None, save=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        rec : SparseTimeFunction or array-like
            The receiver data. Please note that
            these act as the source term in the adjoint run.
        srca : SparseTimeFunction or array-like
            The resulting data for the interpolated at the
            original source location.
        v: TimeFunction, optional
            The computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or self.geometry.new_src(name='srca', src_type=None)
        model = model or self.model

        # Create the adjoint wavefield if not provided
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        kwargs.pop('b', None)

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj(save).apply(srca=srca, rec=rec, v=v,
                                      dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, v, summary


    def gradient(self, rec, u, v=None, gradVp_or_M=None, gradR=None,
                        model=None, **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u : TimeFunction
            Full wavefield `u` (created with save=True).
        v : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Gradient field and performance summary.
        """
        dt = kwargs.pop('dt', self.dt)
        model = model or self.model

        grad_velocity_name = 'grad' + str.capitalize(model.vp_m_edit_mode)
        gradVp_or_M = gradVp_or_M or Function(name=grad_velocity_name, grid=self.model.grid)
        gradR = gradR or VectorFunction(name='gradR', grid=self.model.grid)
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        kwargs.update(model.physical_params(**kwargs))
        'b' in kwargs and kwargs.pop('b')
        

        kwargs[grad_velocity_name] = gradVp_or_M
        # summary = self.op_grad().apply(rec=rec, gradVp=gradVp_or_M, gradR_x=gradR[0], gradR_y=gradR[1],
        #                             v=v, u=u, dt=dt, **kwargs)
        summary = self.op_grad().apply(rec=rec, gradR_x=gradR[0], gradR_y=gradR[1],
                                    v=v, u=u, dt=dt, **kwargs)

        return gradVp_or_M, gradR, summary
    

    def illumination(self, u, illumVp_or_M=None, illumR=None,
                        model=None, **kwargs):
        """
        Computes the illumination.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u : TimeFunction
            Full wavefield `u` (created with save=True).
        v : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Source illumination and performance summary.
        """
        dt = kwargs.pop('dt', self.dt)
        model = model or self.model

        illum_velocity_name = 'illum' + str.capitalize(model.vp_m_edit_mode)
        illumVp_or_M = illumVp_or_M or Function(name=illum_velocity_name, grid=self.model.grid)
        illumR = illumR or VectorFunction(name='illumR', grid=self.model.grid)

        kwargs[illum_velocity_name] = illumVp_or_M
        summary = self.op_illum().apply(illumR_x=illumR[0], illumR_y=illumR[1],
                                    u=u, dt=dt, **kwargs)

        return illumVp_or_M, illumR, summary

    # born = jacobian
    jacobian_adjoint = gradient