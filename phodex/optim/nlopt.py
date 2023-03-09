from typing import Callable

import numpy as np
from autograd import tensor_jacobian_product
from meep.adjoint import OptimizationProblem


def get_objective(
    mpa_opt: OptimizationProblem,
    mapping: Callable,
    callback: Callable | None = None,
) -> Callable:
    def _objective(x, gd):
        f0, meep_grad = mpa_opt([mapping(x)])
        f0 = np.real(f0)

        if gd.size > 0:
            gd[:] = tensor_jacobian_product(mapping, 0)(x, np.sum(meep_grad, axis=1))

        if callback:
            callback(x, f0, meep_grad)

        return f0

    return _objective


def get_epigraph_formulation(
    mpa_opt: OptimizationProblem,
    mapping: Callable,
    callback: Callable | None = None,
) -> tuple[Callable, Callable]:
    """Helper function to create the NLopt objective and vector constraints for an
    epigraph formulation of a meep adjoint optimization problem.

    Parameters
    ----------
    mpa_opt : OptimizationProblem
        Instance of a meep adjoint optimization problem.
    mapping : function
        A Python function that maps the optimization variables to a design
        distribution. Expects the following signature::

            mapping(x: np.ndarray) -> np.ndarray

        If `mapping` takes additional arguments, you can wrap it in a lambda before
        passing it to this function.
    callback : function, optional
        A function that is called after every evaluation of the constraints, useful for
        logging during optimization.
        Expects the following signature::

            callback(t: float, v: np.ndarray, f0: np.ndarray, grad: np.ndarray) -> None

        with the epigraph dummy objective `t`, design weights `v`, the current meep
        objective values `f0` and the gradients `grad`.

    Returns
    -------
    _objective : function
        The epigraph dummy objective.
    _constraints: function
        The epigraph constraints.
    """

    def _objective(x: np.ndarray, gd: np.ndarray) -> float:
        if gd.size > 0:
            gd[0] = 1
            gd[1:] = 0
        return x[0]

    def _constraints(result: np.ndarray, x: np.ndarray, gd: np.ndarray) -> None:
        t, v = x[0], x[1:]

        f0, meep_grad = mpa_opt([mapping(v)])

        # f0 -> (obj_funs, wavelengths), grad -> (obj_funs, dofs, wavelengths)
        f0 = np.atleast_2d(f0)
        meep_grad = np.reshape(meep_grad, (f0.shape[0], -1, f0.shape[-1]))

        f0 = np.concatenate(f0, axis=0)
        meep_grad = np.concatenate(meep_grad, axis=-1)

        grad = np.zeros((v.size, meep_grad.shape[-1]))

        for k in range(grad.shape[-1]):
            grad[:, k] = tensor_jacobian_product(mapping, 0)(v, meep_grad[:, k])

        if gd.size > 0:
            gd[:, 0] = -1
            gd[:, 1:] = grad.T

        result[:] = np.real(f0) - t

        if callback:
            callback(t, v, f0, grad)

    return _objective, _constraints
