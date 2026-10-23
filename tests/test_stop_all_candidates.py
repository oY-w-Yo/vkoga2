"""Test that STOP_ALL_CANDIDATES triggers when max_centers > available candidates."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "..")

from vkoga2 import (
    PDE_STOP_ALL_CANDIDATES,
    PDE_STOP_MAX_CENTERS,
    PDEGreedySolver,
    SecondOrderEllipticComposer,
    VKOGARebuilt,
    build_pde_training_data,
    build_training_data,
)
from vkoga2.kernels import GaussianKernel
from vkoga2.point_selection import point_selection_rule_from_config
from vkoga2.solver import STOP_ALL_CANDIDATES


@pytest.fixture
def standard_data():
    """3 training points — trivial case."""
    X = np.array([[0.0], [0.5], [1.0]])
    y = np.array([0.0, 1.0, 0.0])
    return build_training_data(X, y)


@pytest.fixture
def gaussian_kernel():
    return GaussianKernel(gamma=1.0)


class TestVKOGARebuiltStopAllCandidates:
    """Solver should stop with STOP_ALL_CANDIDATES when n_trial < max_centers."""

    def test_all_candidates_selected(self, standard_data, gaussian_kernel):
        solver = VKOGARebuilt(
            kernel=gaussian_kernel,
            point_selection_rule=point_selection_rule_from_config("f_greedy"),
            max_centers=100,  # >> n_trial = 3
            power_tol=0.0,
        )
        solver.fit(standard_data.X_candidates, standard_data.y_candidates)
        assert solver.stop_reason_ == STOP_ALL_CANDIDATES
        assert solver.n_selected_full_ == 3

    def test_max_centers_honored_when_smaller(self, standard_data, gaussian_kernel):
        solver = VKOGARebuilt(
            kernel=gaussian_kernel,
            point_selection_rule=point_selection_rule_from_config("f_greedy"),
            max_centers=2,  # < n_trial = 3
        )
        solver.fit(standard_data.X_candidates, standard_data.y_candidates)
        assert solver.stop_reason_ == "max_centers"
        assert solver.n_selected_full_ == 2


@pytest.fixture
def pde_data():
    """Minimal PDE problem: 2 interior + 2 Dirichlet points."""
    X_int = np.array([[0.25], [0.75]])
    f_vals = np.ones(2)
    X_dir = np.array([[0.0], [1.0]])
    g_dir = np.array([0.0, 0.0])
    return build_pde_training_data(X_int, f_vals, X_dir, g_dir)


@pytest.fixture
def pde_composer(gaussian_kernel):
    return SecondOrderEllipticComposer(
        kernel=gaussian_kernel,
        diffusion=lambda X: np.ones(len(X)),
        reaction=lambda X: np.zeros(len(X)),
    )


class TestPDEGreedySolverStopAllCandidates:
    """Solver should stop with PDE_STOP_ALL_CANDIDATES when n_trial < max_centers."""

    def test_all_candidates_selected(self, pde_data, pde_composer):
        solver = PDEGreedySolver(
            composer=pde_composer,
            max_centers=100,  # >> n_trial = 4
            power_tol=0.0,
            residual_tol=0.0,
        )
        solver.fit(pde_data)
        assert solver.stop_reason_ == PDE_STOP_ALL_CANDIDATES
        assert solver.n_selected_full_ == 4

    def test_max_centers_honored_when_smaller(self, pde_data, pde_composer):
        solver = PDEGreedySolver(
            composer=pde_composer,
            max_centers=2,  # < n_trial = 4
            power_tol=0.0,
            residual_tol=0.0,
        )
        solver.fit(pde_data)
        assert solver.stop_reason_ == PDE_STOP_MAX_CENTERS
        assert solver.n_selected_full_ == 2
