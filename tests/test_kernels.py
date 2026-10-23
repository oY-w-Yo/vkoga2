import json
import pickle

import numpy as np
import pytest

from vkoga2.distances import pairwise_distance, pairwise_euclidean_distance
from vkoga2.validation import as_valid_point_matrix
from vkoga2.kernels import (
    GaussianKernel,
    Matern32Kernel,
    PolynomialKernel,
    kernel_from_config,
)


def test_pairwise_euclidean_distance_shape_and_values():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])

    D = pairwise_euclidean_distance(X, Y)

    expected = np.array(
        [
            [0.0, 1.0, 3.0],
            [5.0, np.sqrt(20.0), 4.0],
        ]
    )
    assert D.shape == (2, 3)
    np.testing.assert_allclose(D, expected)


def test_pairwise_distance_rejects_unknown_metric():
    X = np.zeros((1, 2))
    with pytest.raises(ValueError, match="Unsupported distance metric"):
        pairwise_distance(X, X, metric="manhattan")


def test_pairwise_distance_rejects_shape_mismatch():
    X = np.zeros((2, 2))
    Y = np.zeros((3, 3))
    with pytest.raises(ValueError, match="same feature dimension"):
        pairwise_euclidean_distance(X, Y)


def test_pairwise_distance_rejects_non_matrix_input():
    with pytest.raises(ValueError, match=r"shape \(n, d\)"):
        pairwise_euclidean_distance(np.array([0.0, 1.0]), np.zeros((2, 2)))


def test_public_validation_helper_accepts_point_matrix():
    X = np.array([[1.0, 2.0]])
    out = as_valid_point_matrix(X, name="X")
    np.testing.assert_allclose(out, X)


def test_gaussian_eval_diagonal_eval_prod_params_dict_and_pickle():
    X = np.array([[0.0], [2.0]])
    Y = np.array([[1.0], [3.0]])
    gamma = 2.0
    kernel = GaussianKernel(gamma=gamma)

    r = pairwise_euclidean_distance(X, Y)
    expected = np.exp(-gamma * r * r)

    np.testing.assert_allclose(kernel.eval(X, Y), expected)
    np.testing.assert_allclose(kernel.diagonal(X), np.ones(2))

    v = np.array([2.0, -1.0])
    np.testing.assert_allclose(kernel.eval_prod(X, Y, v), expected @ v)

    rebuilt = kernel_from_config(**kernel.get_params())
    np.testing.assert_allclose(rebuilt.eval(X, Y), expected)

    metadata = kernel.to_dict()
    json.dumps(metadata)
    assert metadata["parameter_convention"] == "gamma is the coefficient in exp(-gamma * r^2)"
    assert metadata["kernel_type"] == "gaussian"
    assert set(metadata["parameters"]) == {"gamma", "distance_metric"}
    assert not hasattr(kernel, "set_params")

    restored = pickle.loads(pickle.dumps(kernel))
    np.testing.assert_allclose(restored.eval(X, Y), expected)


def test_matern32_eval_diagonal_params_dict_and_pickle():
    X = np.array([[0.0], [2.0]])
    Y = np.array([[1.0], [3.0]])
    gamma = 3.0
    kernel = Matern32Kernel(gamma=gamma)

    r = pairwise_euclidean_distance(X, Y)
    expected = (1.0 + gamma * r) * np.exp(-gamma * r)

    np.testing.assert_allclose(kernel.eval(X, Y), expected)
    np.testing.assert_allclose(kernel.diagonal(X), np.ones(2))

    rebuilt = kernel_from_config(**kernel.get_params())
    np.testing.assert_allclose(rebuilt.eval(X, Y), expected)

    metadata = kernel.to_dict()
    json.dumps(metadata)
    assert metadata["kernel_type"] == "matern32"
    assert metadata["parameter_convention"] == "gamma = inverse length scale"
    assert set(metadata["parameters"]) == {"gamma", "distance_metric"}
    assert not hasattr(kernel, "set_params")

    restored = pickle.loads(pickle.dumps(kernel))
    np.testing.assert_allclose(restored.eval(X, Y), expected)


def test_matern32_eval_prod_matches_eval_matvec_with_and_without_chunking():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(17, 3))
    Y = rng.normal(size=(11, 3))
    v = rng.normal(size=11)
    kernel = Matern32Kernel(gamma=2.7)

    expected = kernel.eval(X, Y) @ v

    np.testing.assert_allclose(kernel.eval_prod(X, Y, v), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        kernel.eval_prod(X, Y, v, chunk_size=5),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_polynomial_eval_diagonal_factory_and_pickle():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[0.5, 1.0], [2.0, 0.0]])
    kernel = PolynomialKernel(degree=3, offset=0.5)

    expected = (0.5 + X @ Y.T) ** 3
    np.testing.assert_allclose(kernel.eval(X, Y), expected)
    np.testing.assert_allclose(kernel.diagonal(X), (0.5 + np.sum(X * X, axis=1)) ** 3)

    factory_kernel = kernel_from_config("polynomial", degree=3, offset=0.5)
    np.testing.assert_allclose(factory_kernel.eval(X, Y), expected)

    metadata = kernel.to_dict()
    json.dumps(metadata)
    assert metadata["kernel_type"] == "polynomial"
    assert set(metadata["parameters"]) == {"degree", "offset"}
    assert not hasattr(kernel, "set_params")

    restored = pickle.loads(pickle.dumps(kernel))
    np.testing.assert_allclose(restored.eval(X, Y), expected)


def test_factory_supported_kernel_types():
    assert isinstance(kernel_from_config("gaussian", gamma=1.5), GaussianKernel)
    assert isinstance(kernel_from_config("matern32", gamma=2.5), Matern32Kernel)
    assert isinstance(kernel_from_config("polynomial", degree=2, offset=1.0), PolynomialKernel)


def test_factory_rejects_unknown_kernel_type():
    with pytest.raises(ValueError, match="Unsupported kernel_type"):
        kernel_from_config("not_a_kernel")


def test_kernel_objects_are_immutable():
    kernel = GaussianKernel(gamma=1.0)
    with pytest.raises(Exception):
        kernel.gamma = 2.0

    poly = PolynomialKernel(degree=2, offset=1.0)
    with pytest.raises(Exception):
        poly.degree = 3


@pytest.mark.parametrize("gamma", [-1.0, np.inf])
def test_gaussian_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        GaussianKernel(gamma=gamma)


def test_matern32_rejects_nan_gamma():
    with pytest.raises(ValueError):
        Matern32Kernel(gamma=np.nan)


@pytest.mark.parametrize("degree", [0, -1, 2.5, True])
def test_polynomial_rejects_invalid_degree(degree):
    with pytest.raises(ValueError):
        PolynomialKernel(degree=degree, offset=1.0)


@pytest.mark.parametrize("offset", [-1.0, np.inf])
def test_polynomial_rejects_invalid_offset(offset):
    with pytest.raises(ValueError):
        PolynomialKernel(degree=2, offset=offset)


def test_reconstruction_from_get_params_for_all_kernel_types():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    Y = np.array([[1.0, 0.0], [4.0, 5.0]])
    kernels = [
        GaussianKernel(gamma=2.0),
        Matern32Kernel(gamma=3.0),
        PolynomialKernel(degree=3, offset=0.5),
    ]
    for kernel in kernels:
        rebuilt = kernel_from_config(**kernel.get_params())
        np.testing.assert_allclose(rebuilt.eval(X, Y), kernel.eval(X, Y))


def test_to_dict_parameters_do_not_repeat_kernel_type():
    kernels = [
        (GaussianKernel(gamma=2.0), {"gamma", "distance_metric"}),
        (Matern32Kernel(gamma=3.0), {"gamma", "distance_metric"}),
        (PolynomialKernel(degree=2, offset=1.0), {"degree", "offset"}),
    ]
    for kernel, expected_parameter_keys in kernels:
        metadata = kernel.to_dict()
        json.dumps(metadata)
        assert "kernel_type" in metadata
        assert "kernel_type" not in metadata["parameters"]
        assert set(metadata["parameters"]) == expected_parameter_keys


def test_gamma_zero_flat_limit_for_rbf_kernels():
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    Y = np.array([[3.0, 4.0], [5.0, 6.0], [0.0, 0.0]])

    np.testing.assert_allclose(GaussianKernel(gamma=0.0).eval(X, Y), np.ones((2, 3)))
    np.testing.assert_allclose(Matern32Kernel(gamma=0.0).eval(X, Y), np.ones((2, 3)))


def test_unknown_distance_metric_raises_through_kernel_eval():
    X = np.zeros((2, 2))
    Y = np.zeros((3, 2))
    kernel = GaussianKernel(gamma=1.0, distance_metric="bad_metric")
    with pytest.raises(ValueError, match="Unsupported distance metric"):
        kernel.eval(X, Y)


def test_rbf_second_argument_derivatives_match_symmetry():
    X = np.array([[0.0, 1.0], [2.0, -1.0]])
    Y = np.array([[1.5, 0.5], [-0.5, 2.0], [0.0, 0.0]])
    kernel = GaussianKernel(gamma=0.7)

    np.testing.assert_allclose(
        kernel.gradient_wrt_second(X, Y),
        -kernel.gradient_wrt_first(X, Y),
    )
    np.testing.assert_allclose(
        kernel.laplacian_wrt_second(X, Y),
        kernel.laplacian_wrt_first(X, Y),
    )


def test_gaussian_bilaplacian_matches_closed_form_and_finite_difference():
    gamma = 0.7
    kernel = GaussianKernel(gamma=gamma)
    X = np.array([[0.0], [0.3]])
    Y = np.array([[-0.4], [1.1]])

    r = pairwise_euclidean_distance(X, Y)
    r_sq = r * r
    expected = (
        16.0 * gamma**4 * r_sq * r_sq
        - 16.0 * gamma**3 * 3.0 * r_sq
        + 12.0 * gamma**2
    ) * np.exp(-gamma * r_sq)
    np.testing.assert_allclose(kernel.bilaplacian(X, Y), expected, rtol=1e-12, atol=1e-12)

    x0 = np.array([[0.3]])
    y0 = np.array([[-0.4]])
    h = 1e-4

    def lap_y_at(x):
        return kernel.laplacian_wrt_second(np.array([[x]]), y0)[0, 0]

    finite_diff = (
        lap_y_at(x0[0, 0] + h)
        - 2.0 * lap_y_at(x0[0, 0])
        + lap_y_at(x0[0, 0] - h)
    ) / (h * h)
    np.testing.assert_allclose(
        kernel.bilaplacian(x0, y0)[0, 0],
        finite_diff,
        rtol=2e-5,
        atol=2e-5,
    )


def test_non_gaussian_bilaplacian_not_implemented_for_now():
    kernel = Matern32Kernel(gamma=1.0)

    with pytest.raises(NotImplementedError, match="bilaplacian"):
        kernel.bilaplacian(np.array([[0.0]]), np.array([[1.0]]))
