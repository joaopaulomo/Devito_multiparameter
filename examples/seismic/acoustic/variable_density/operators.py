from devito import *

from examples.seismic import AcquisitionGeometry


def ForwardOperator(model, geometry, space_order=4, save=False, kernel="OT2", **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium with variable density.

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
    m = model.m
    b = model.b
    rho = 1 / b

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(
        name="u",
        grid=model.grid,
        save=geometry.nt if save else None,
        time_order=2,
        space_order=space_order,
    )
    src = geometry.src
    rec = geometry.rec

    # Create the stencil

    pde = m * u.dt2 - rho * div(b * grad(u, 0.5), -0.5) + model.damp * u.dt

    stencil = Eq(u.forward, solve(pde, u.forward))

    s = model.grid.stepping_dim.spacing

    # Create the equation
    src_term = src.inject(u.forward, expr=src * s**2 / m)
    rec_term = rec.interpolate(expr=u)

    equation = [stencil] + src_term + rec_term

    return Operator(equation, subs=model.spacing_map, name="Forward", **kwargs)


def AdjointOperator(model, geometry, space_order=4, kernel="OT2", save=False, **kwargs):
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
    m = model.m
    b = model.b
    rho = 1 / b

    v = TimeFunction(
        name="v",
        grid=model.grid,
        save=geometry.nt if save else None,
        time_order=2,
        space_order=space_order,
    )

    srca = geometry.new_src(name="srca", src_type=None)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing

    eqn = m * v.dt2 - div(b * grad(rho * v, 0.5), -0.5) + model.damp * v.dt.T

    stencil = Eq(v.backward, solve(eqn, v.backward))

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Create interpolation expression for the adjoint-source

    source_a = srca.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    return Operator(
        [stencil] + receivers + source_a,
        subs=model.spacing_map,
        name="Adjoint",
        **kwargs
    )


def BornOperator(
    model, geometry: AcquisitionGeometry, space_order=4, kernel="OT2", **kwargs
):
    """
    Construct an Linearized Born operator in an acoustic medium with variable density.

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
        Type of discretization, centered or shifted.
    """
    src = geometry.src
    rec = geometry.rec

    u = TimeFunction(
        name="u", grid=model.grid, time_order=2, space_order=space_order, save=None
    )
    du = TimeFunction(
        name="du", grid=model.grid, time_order=2, space_order=space_order, save=None
    )
    dm = Function(name="dm", grid=model.grid, space_order=0)

    kappa = 1 / (model.m * model.b)
    pde_bkg = (
        u.dt2
        - kappa * div(model.b * grad(u, shift=-0.5), shift=0.5)
        + model.damp * u.dt
    )
    pde_sct = (
        du.dt2
        - kappa * div(model.b * grad(du, shift=-0.5), shift=0.5)
        + model.damp * du.dt
        + dm / model.m * u.dt2
    )

    eq_fwd = Eq(u.forward, solve(pde_bkg, u.forward))
    eq_born = Eq(du.forward, solve(pde_sct, du.forward))

    src_term = src.inject(u.forward, src * model.critical_dt**2 / model.m)
    # src_term = src.inject(u.forward, src*model.critical_dt**2)
    rec_term = rec.interpolate(expr=du)

    return Operator(
        [eq_fwd] + src_term + [eq_born] + rec_term,
        subs=model.spacing_map,
        name="Born",
        **kwargs
    )


def GradientOperator(model, geometry, space_order=4, save=True, kernel="OT2", **kwargs):
    """
    Construct a gradient operator in an acoustic media for m and b.

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
        Option to store the entire (unrolled) wavefield.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m
    b = model.b
    rho = 1 / b
    rec = geometry.rec

    grad_m = Function(name="grad_m", grid=model.grid)
    grad_rho = Function(name="grad_rho", grid=model.grid)

    u = TimeFunction(
        name="u",
        grid=model.grid,
        save=save and geometry.nt or None,
        time_order=2,
        space_order=space_order,
    )

    v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=space_order)

    pde_adj = (
        m * v.dt2 - div(b * grad(rho * v, shift=0.5), shift=-0.5) + model.damp * v.dt.T
    )
    eqn = Eq(v.backward, solve(pde_adj, v.backward))

    gradient_update_m = Inc(grad_m, -v * u.dt2)  

    grad_u = u.grad(shift=0.5) 
    grad_rho_func = rho.grad(shift=0.5) 

    term1 = (1 / rho**2) * grad_rho_func.dot(grad_u)  
    term2 = (1 / rho) * u.laplace  
    dL_drho = term1 - term2  

    gradient_update_rho = Inc(
        grad_rho, -v * dL_drho
    )  

    rec_term = rec.inject(field=v.backward, expr=rec * model.critical_dt**2 / m)

    return Operator(
        [eqn] + rec_term + [gradient_update_m, gradient_update_rho],
        subs=model.spacing_map,
        name="Gradient",
        **kwargs
    )
