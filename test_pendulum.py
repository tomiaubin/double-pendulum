"""Tests for pendulum.py. Run with:  pytest"""

import numpy as np
from pendulum import deriv, energy, euler, rk4, reference, lyapunov, g

Y0 = np.array([np.pi / 2, 0.0, np.pi / 2, 0.0])


def test_equilibrium_is_stationary():
    # Hanging straight down and at rest: nothing moves
    assert np.allclose(deriv([0, 0, 0, 0], 0), 0)


def test_small_oscillation_frequency():
    # For tiny angles theory gives a normal mode with omega^2 = g (2 - sqrt(2)).
    # After exactly one period the pendulum must be back where it started.
    omega = np.sqrt(g * (2 - np.sqrt(2)))
    eps = 1e-4
    y0 = [eps, 0, np.sqrt(2) * eps, 0]
    y = reference(y0, [0, 2 * np.pi / omega])
    assert abs(y[-1, 0] - eps) < 1e-3 * eps


def test_reference_conserves_energy():
    t = np.linspace(0, 20, 2000)
    E = energy(reference(Y0, t))
    assert np.max(np.abs(E - E[0])) < 1e-8


def test_euler_drifts_and_rk4_does_not():
    t = np.linspace(0, 20, 2001)
    drift_euler = abs(energy(euler(Y0, t))[-1] - energy(Y0)[0])
    drift_rk4 = abs(energy(rk4(Y0, t))[-1] - energy(Y0)[0])
    assert drift_euler > 1.0
    assert drift_rk4 < 1e-3


def test_rk4_is_fourth_order():
    # Halving the step should divide the error by about 2^4 = 16
    y0 = [0.3, 0, 0.2, 0]
    exact = reference(y0, [0, 2.0])[-1]
    err = []
    for n in (160, 320):
        err.append(np.linalg.norm(rk4(y0, np.linspace(0, 2, n + 1))[-1] - exact))
    assert 13 < err[0] / err[1] < 19


def test_lyapunov_chaotic_vs_regular():
    assert lyapunov(Y0, t_total=40) > 0.5                       # chaos
    assert abs(lyapunov([0.05, 0, 0.05, 0], t_total=40)) < 0.3  # regular motion
