#Created by Yaoning

import numpy as np
from scipy.constants import h, m_e, k, e, pi
import matplotlib.pyplot as plt


def eta_thermionic(E_joules, T_kelvin):
    """
    EEDF model: η(E) = (-4 * π * me * qe / h^3) * E * exp(-E / (kB * T))
        where qe, h and kB are the elementary charge, Planck constant and Boltzmann constant, respectively, and me and E are the mass and energy of electrons emitted at temperature T.
        source: https://dx.doi.org/10.14288/1.0445206 section 1.4 equation 6
    """
    q_e = -e  
    constant = (-4 * np.pi * m_e * q_e) / (h**3)
    boltzmann_factor = np.exp(-E_joules / (k * T_kelvin))
    return constant * E_joules * boltzmann_factor

T = 2500.0 
for E_ev in np.arange(0, 2.5, 0.025):
    E_joules = E_ev * e 
    result = eta_thermionic(E_joules, T)
    plt.plot(E_ev, result, 'bo', markersize=2)


plt.xlabel('Energy (eV)')
plt.ylabel(f'η(E)@T={T}K')
plt.show()
