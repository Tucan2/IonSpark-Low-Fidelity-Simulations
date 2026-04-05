# Created by Ujjwal (debugged & fixed)

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, e, pi, m_e, hbar, k
from scipy.special import ellipk, ellipe


def schottky_const():
    return np.sqrt(e**3 / (4 * pi * epsilon_0))


def y_MG(F, phi):
    phi_J = phi * e
    return schottky_const() * np.sqrt(F) / phi_J


def v_MG(y):
    y = np.clip(y, 1e-12, 1 - 1e-12)  # avoid singularities

    k = np.sqrt((1 - y) / (1 + y))

    K = ellipk(k**2)
    E = ellipe(k**2)

    return np.sqrt(1 + y) * (E - y * K)


def t_MG(y):
    y = np.clip(y, 1e-12, 1 - 1e-12)

    k = np.sqrt((1 - y) / (1 + y))

    K = ellipk(k**2)
    E = ellipe(k**2)

    return (1 / np.sqrt(1 + y)) * ((1 + y) * E - y * K)


def d_MG(y, F, phi):
    phi_J = phi * e
    t = t_MG(y)
    return (2 * np.sqrt(2 * m_e) / (e * hbar)) * np.sqrt(phi_J) * t / F


def theta_T_MG(y, F, phi, T):
    d = d_MG(y, F, phi)
    x = d * pi * k * T

    # avoid division by zero
    return np.where(x == 0, 1, x / np.sin(x))


def J_murphy_good(F, phi, T):
    # constants
    a = e**3 / (15 * (pi**2) * hbar)
    b = 4 * np.sqrt(2 * m_e) / (3 * e * hbar)

    y = y_MG(F, phi)
    t = t_MG(y)
    v = v_MG(y)
    theta_T = theta_T_MG(y, F, phi, T)

    phi_J = phi * e

    return (a / (t**2)) * (F**2 / phi_J) * \
           np.exp(-b * v * (phi_J**1.5) / F) * theta_T


# -------- TEST PLOT --------
"""
phi = 4.5  # eV
T = 300

F = np.linspace(1e9, 5e9, 200)

J = J_murphy_good(F, phi, T)

plt.plot(F, J)
plt.xlabel("Electric field (V/m)")
plt.ylabel("Current density (A/m^2)")
plt.title("Murphy-Good emission")
plt.yscale("log")
plt.show()
"""