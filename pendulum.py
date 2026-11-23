"""Double pendulum: comparison of numerical integrators and chaos.

State of the system: y = [theta1, omega1, theta2, omega2]
(angles measured from the downward vertical, omega = angular velocity).
Run `python pendulum.py` to reproduce the figures and the numbers in the README.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Physical parameters
g = 9.81
L1, L2 = 1.0, 1.0
m1, m2 = 1.0, 1.0


def deriv(y, t):
    """Equations of motion (from the Lagrangian): returns dy/dt."""
    theta1, z1, theta2, z2 = y
    delta = theta2 - theta1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1

    dz1 = (m2 * L1 * z1**2 * np.sin(delta) * np.cos(delta)
           + m2 * g * np.sin(theta2) * np.cos(delta)
           + m2 * L2 * z2**2 * np.sin(delta)
           - (m1 + m2) * g * np.sin(theta1)) / den1

    dz2 = (-m2 * L2 * z2**2 * np.sin(delta) * np.cos(delta)
           + (m1 + m2) * (g * np.sin(theta1) * np.cos(delta)
                          - L1 * z1**2 * np.sin(delta)
                          - g * np.sin(theta2))) / den2

    return np.array([z1, dz1, z2, dz2])


def energy(y):
    """Total energy (kinetic + potential). y can be one state or an array of states."""
    y = np.atleast_2d(y)
    theta1, z1, theta2, z2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    # Heights of the two masses
    h1 = -L1 * np.cos(theta1)
    h2 = h1 - L2 * np.cos(theta2)

    # Velocities of the two masses
    vx1, vy1 = L1 * z1 * np.cos(theta1), L1 * z1 * np.sin(theta1)
    vx2, vy2 = vx1 + L2 * z2 * np.cos(theta2), vy1 + L2 * z2 * np.sin(theta2)

    kinetic = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
    potential = g * (m1 * h1 + m2 * h2)
    return kinetic + potential


def euler(y0, t):
    """Explicit Euler: y_{n+1} = y_n + h * f(y_n). First order."""
    y = np.zeros((len(t), 4))
    y[0] = y0
    for n in range(1, len(t)):
        h = t[n] - t[n - 1]
        y[n] = y[n - 1] + h * deriv(y[n - 1], t[n - 1])
    return y


def rk4(y0, t):
    """Classical Runge-Kutta 4: four slope evaluations per step. Fourth order."""
    y = np.zeros((len(t), 4))
    y[0] = y0
    for n in range(1, len(t)):
        h = t[n] - t[n - 1]
        k1 = deriv(y[n - 1], t[n - 1])
        k2 = deriv(y[n - 1] + h / 2 * k1, t[n - 1] + h / 2)
        k3 = deriv(y[n - 1] + h / 2 * k2, t[n - 1] + h / 2)
        k4 = deriv(y[n - 1] + h * k3, t[n - 1] + h)
        y[n] = y[n - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def reference(y0, t):
    """Very accurate solution using scipy's odeint with tiny tolerances."""
    return odeint(deriv, y0, t, rtol=1e-12, atol=1e-12)


def lyapunov(y0, t_total=100.0, tau=0.5, d0=1e-8):
    """Largest Lyapunov exponent (Benettin method).

    We follow two trajectories that start a distance d0 apart. Every tau seconds
    we measure how far apart they are, record the growth, and put the second
    trajectory back at distance d0. The exponent is the average growth rate of
    log(distance). Positive means chaos.
    """
    a = np.array(y0, dtype=float)
    b = a + d0 * np.ones(4) / 2          # nearby trajectory, at distance d0
    total = 0.0
    n_steps = int(t_total / tau)
    for _ in range(n_steps):
        a = reference(a, [0, tau])[-1]
        b = reference(b, [0, tau])[-1]
        d = np.linalg.norm(b - a)
        total += np.log(d / d0)
        b = a + (b - a) * d0 / d         # renormalise the separation back to d0
    return total / (n_steps * tau)


def main():
    y0 = np.array([np.pi / 2, 0.0, np.pi / 2, 0.0])   # both rods horizontal, at rest
    scale = g * (m1 * L1 + m2 * (L1 + L2))             # energy scale (E(0) is 0 here)

    # 1. Energy conservation for each method (h = 0.01 s)
    t = np.linspace(0, 20, 2001)
    methods = {"Euler": euler(y0, t), "RK4": rk4(y0, t), "odeint (reference)": reference(y0, t)}
    E0 = energy(y0)[0]

    plt.figure(figsize=(7, 4.5))
    for name, y in methods.items():
        drift = np.abs(energy(y) - E0) / scale
        plt.semilogy(t, drift + 1e-18, label=name)
        print(f"{name}: max energy drift = {drift.max():.2e} (relative to m g L)")
    plt.xlabel("t [s]")
    plt.ylabel("|E(t) - E(0)| / (m g L)")
    plt.title("Energy conservation (h = 0.01 s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("imgs/comparacion_energia.png", dpi=150)

    # 2. Convergence order: error at t = 2 s as the step h shrinks (small angles, no chaos)
    y0_small = np.array([0.3, 0.0, 0.2, 0.0])
    exact = reference(y0_small, [0, 2.0])[-1]
    steps = np.array([25, 50, 100, 200, 400, 800])
    h = 2.0 / steps
    err_euler = [np.linalg.norm(euler(y0_small, np.linspace(0, 2, n + 1))[-1] - exact) for n in steps]
    err_rk4 = [np.linalg.norm(rk4(y0_small, np.linspace(0, 2, n + 1))[-1] - exact) for n in steps]

    # Order = slope of log(error) vs log(h) (fit on the 4 smallest steps)
    print("Euler order:", np.polyfit(np.log(h[-4:]), np.log(err_euler[-4:]), 1)[0])
    print("RK4 order:  ", np.polyfit(np.log(h[-4:]), np.log(err_rk4[-4:]), 1)[0])

    plt.figure(figsize=(6, 4.5))
    plt.loglog(h, err_euler, "o-", label="Euler")
    plt.loglog(h, err_rk4, "o-", label="RK4")
    plt.xlabel("step h [s]")
    plt.ylabel("error at t = 2 s")
    plt.legend()
    plt.tight_layout()
    plt.savefig("imgs/convergencia.png", dpi=150)

    # 3. Sensitivity to initial conditions: two trajectories 1e-8 apart
    t = np.linspace(0, 20, 4001)
    a = reference(y0, t)
    b = reference(y0 + np.array([0, 0, 1e-8, 0]), t)
    plt.figure(figsize=(6, 4.5))
    plt.semilogy(t, np.linalg.norm(b - a, axis=1))
    plt.xlabel("t [s]")
    plt.ylabel("distance between the two trajectories")
    plt.title("Initial distance 1e-8")
    plt.tight_layout()
    plt.savefig("imgs/sensibilidad_condiciones_iniciales.png", dpi=150)

    print("Lyapunov exponent (chaotic):     ", lyapunov(y0))
    print("Lyapunov exponent (small angles):", lyapunov([0.05, 0.0, 0.05, 0.0]))


if __name__ == "__main__":
    main()
