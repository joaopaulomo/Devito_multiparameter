from devito import *
from examples.seismic.model import Model
from examples.seismic.utils import AcquisitionGeometry


def get_velocity_term(model: Model):
    if model.vp_m_edit_mode == 'vp':
        return -grad(model.vp)/model.vp
    else:
        return grad(0.5 * model.m)/model.m


def get_forward_equation(model: Model, field: TimeFunction):
    m = model.m
    r = model.r
    velocity_term = get_velocity_term(model)
    
    return m * field.dt2 - field.laplace + (2*r + velocity_term).T * grad(field) + model.damp * field.dt


def get_adjoint_equation(model: Model, field: TimeFunction):
    m = model.m
    r = model.r
    velocity_term = get_velocity_term(model)

    return m * field.dt2 - field.laplace - div((2*r + velocity_term) * field, .5) + model.damp * field.dt.T


def get_velocity_gradient(model: Model, field: TimeFunction):
    u = field

    vp = model.vp
    m = model.m

    if model.vp_m_edit_mode == 'vp':
        return (-2/vp**3 * u.dt2 + 1/vp**2 * (vp.dx * u.dx + vp.dy * u.dy) - 1/vp * u.laplace)
        # return u.dt2
        # return u
    else:
        return (u.dt2 + 1/2 * (-(m.dx * u.dx + m.dy * u.dy)/m**2 + u.laplace/m))
        # return u


def ForwardOperator(model: Model, geometry: AcquisitionGeometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium through a 
    wave equation in terms of the P-wave velocity and the vector reflectivity.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order, staggered=NODE)
    src = geometry.src
    rec = geometry.rec

    m = model.m

    pde = get_forward_equation(model, u)
    stencil = Eq(u.forward, solve(pde, u.forward))

    s = model.grid.stepping_dim.spacing

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2  / m)
    
    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)

    # Substitute spacing terms to reduce flops
    return Operator([stencil] + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model: Model, geometry: AcquisitionGeometry, space_order=4,
                    kernel='OT2', save=False, **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    v = TimeFunction(name='v', grid=model.grid, 
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order, staggered=NODE)

    srca = geometry.new_src(name='srca', src_type=None)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing

    m = model.m

    pde = get_adjoint_equation(model, v)
    stencil = Eq(v.backward, solve(pde, v.backward))

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    return Operator([stencil] + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def GradientOperator(model, geometry, space_order=4, **kwargs):
    """
    Construct a gradient operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    """
    m = model.m

    # Gradient symbol and wavefield symbols
    gradVp_or_M = Function(name='grad' + str.capitalize(model.vp_m_edit_mode), grid=model.grid)
    gradR = VectorFunction(name='gradR', grid=model.grid)

    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt, 
                     time_order=2, space_order=space_order, staggered=NODE)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order, staggered=NODE)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing

    pde = get_adjoint_equation(model, v)
    eqn = [Eq(v.backward, solve(pde, v.backward))]

    # gradient_update = [
    #     Inc(gradVp_or_M, - get_velocity_gradient(model, u) * v),
    #     Inc(gradR, - 2*grad(u) * v)
    # ]
    gradient_update = [
        Eq(gradVp_or_M, gradVp_or_M - get_velocity_gradient(model, u) * v),
        Eq(gradR, gradR - 2*grad(u) * v)
    ]
    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + gradient_update, subs=model.spacing_map,
                    name='Gradient', **kwargs)


def IlluminationOperator(model, geometry, space_order=4, **kwargs):
    """
    Construct an operator to compute the illumination (hessian diagonal).

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    """
    # Gradient symbol and wavefield symbols
    illumVp_or_M = Function(name='illum' + str.capitalize(model.vp_m_edit_mode), grid=model.grid)
    illumR = VectorFunction(name='illumR', grid=model.grid)

    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt, 
                     time_order=2, space_order=space_order, staggered=NODE)

    illum_update = [
        Inc(illumVp_or_M, get_velocity_gradient(model, u)**2),
        Inc(illumR[0], 4*u.dx**2),
        Inc(illumR[1], 4*u.dy**2)
    ]

    # Substitute spacing terms to reduce flops
    return Operator(illum_update, subs=model.spacing_map,
                    name='Illumination', **kwargs)