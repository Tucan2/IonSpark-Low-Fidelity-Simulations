# Created by Ujjwal

import numpy as np
from scipy.constants import hbar, m_e, pi, e
import matplotlib.pyplot as plt

def J_fowler_nordheim(F, phi):
    """
    J = a*(F^2)/phi * exp(-b * phi**1.5 / F)
    Constants:-
    a = q_e^3 / (16*pi^2*hbar)
    b = 4*sqrt(2m) / (3*q_e*hbar)

    q_e is the charge of an electron
    hbar is reduced Planck constant
    m is mass of electron

    phi is the workfunction
    F is Applied electric field
    """

    a = e**3/(15*(np.pi**2)*hbar)
    b = 4*np.sqrt(2*m_e)/(3*e*hbar)

    return a*(F**2)/phi * np.exp(-b*(phi**1.5)/F)


"""
Example usage

import numpy as np
import matplotlib.pyplot as plt

phi = 4.5  # eV
F = np.linspace(1e9, 1e10, 200)

J = J_fowler_nordheim(F, phi)

plt.plot(F, J)
plt.xlabel("Electric field (V/m)")
plt.ylabel("Current density (A/m^2)")
plt.title("Fowler-Nordheim emission")
plt.show()
"""