import numpy as np
import pytest

from vkoga2.data import PDE_FUNCTIONAL_DIRICHLET, PDE_FUNCTIONAL_INTERIOR
from vkoga2.kernels import GaussianKernel
from vkoga2.pde_composer import SecondOrderEllipticComposer


def _ones(X):
    return np.ones(len(X), dtype=float)


def _zeros(X):
    return np.zeros(len(X), dtype=float)


def test_second_order_elliptic_power_diagonal_for_phase1_types():
    gamma = 0.5
    kernel = GaussianKernel(gamma=gamma)
    composer = SecondOrderEllipticComposer(kernel, diffusion=_ones, reaction=_zeros)
    X = np.array([[0.0], [1.0]])
    types = np.array([PDE_FUNCTIONAL_INTERIOR, PDE_FUNCTIONAL_DIRICHLET])

    diag = composer.power_diagonal(types, X)

    np.testing.assert_allclose(diag, np.array([12.0 * gamma**2, 1.0]))


def test_second_order_elliptic_eval_candidate_dispatch_blocks():
    kernel = GaussianKernel(gamma=0.7)
    composer = SecondOrderEllipticComposer(kernel, diffusion=_ones, reaction=_zeros)
    trial_X = np.array([[0.25], [0.0], [1.0]])
    trial_types = np.array([
        PDE_FUNCTIONAL_INTERIOR,
        PDE_FUNCTIONAL_DIRICHLET,
        PDE_FUNCTIONAL_DIRICHLET,
    ])
    z_int = np.array([[0.25]])
    z_dir = np.array([[0.0]])

    interior_center = composer.eval_candidate(
        PDE_FUNCTIONAL_INTERIOR,
        trial_X,
        trial_types,
        z_int,
    )
    dirichlet_center = composer.eval_candidate(
        PDE_FUNCTIONAL_DIRICHLET,
        trial_X,
        trial_types,
        z_dir,
    )

    expected_int_center = np.array([
        kernel.bilaplacian(trial_X[:1], z_int)[0, 0],
        -kernel.laplacian_wrt_second(trial_X[1:2], z_int)[0, 0],
        -kernel.laplacian_wrt_second(trial_X[2:3], z_int)[0, 0],
    ])
    expected_dir_center = np.array([
        -kernel.laplacian_wrt_first(trial_X[:1], z_dir)[0, 0],
        kernel.eval(trial_X[1:2], z_dir)[0, 0],
        kernel.eval(trial_X[2:3], z_dir)[0, 0],
    ])
    np.testing.assert_allclose(interior_center, expected_int_center)
    np.testing.assert_allclose(dirichlet_center, expected_dir_center)


def test_second_order_elliptic_prediction_matrices():
    kernel = GaussianKernel(gamma=0.8)
    composer = SecondOrderEllipticComposer(kernel, diffusion=_ones, reaction=_zeros)
    X = np.array([[0.2], [0.7]])
    Z = np.array([[0.3], [1.0]])
    center_types = np.array([PDE_FUNCTIONAL_INTERIOR, PDE_FUNCTIONAL_DIRICHLET])

    value_matrix = composer.eval_predict(center_types, X, Z)
    operator_matrix = composer.eval_operator_predict(center_types, X, Z)

    expected_value = np.column_stack((
        -kernel.laplacian_wrt_second(X, Z[:1])[:, 0],
        kernel.eval(X, Z[1:2])[:, 0],
    ))
    expected_operator = np.column_stack((
        kernel.bilaplacian(X, Z[:1])[:, 0],
        -kernel.laplacian_wrt_first(X, Z[1:2])[:, 0],
    ))
    np.testing.assert_allclose(value_matrix, expected_value)
    np.testing.assert_allclose(operator_matrix, expected_operator)


def test_second_order_elliptic_coefficients_accept_scalar_returns():
    kernel = GaussianKernel(gamma=0.5)
    composer = SecondOrderEllipticComposer(
        kernel,
        diffusion=lambda X: 2.0,
        reaction=lambda X: 3.0,
    )
    X = np.array([[0.1], [0.2]])
    types = np.array([PDE_FUNCTIONAL_INTERIOR, PDE_FUNCTIONAL_DIRICHLET])

    diag = composer.power_diagonal(types, X)

    assert diag.shape == (2,)
    assert np.all(np.isfinite(diag))


def test_second_order_elliptic_rejects_unknown_functional_type():
    composer = SecondOrderEllipticComposer(
        GaussianKernel(gamma=0.5),
        diffusion=_ones,
        reaction=_zeros,
    )

    with pytest.raises(ValueError, match="unsupported functional types"):
        composer.power_diagonal(np.array([2]), np.array([[0.0]]))

    with pytest.raises(ValueError, match="Unsupported center_type"):
        composer.eval_candidate(2, np.array([[0.0]]), np.array([0]), np.array([[0.0]]))
