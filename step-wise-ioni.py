"""
    Please keep in mind that this is very preliminary. Zero clue if this is useful or correct or relavent in any way.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- 1. System Parameters & Constants ---
Te = 2.0  # Electron temperature in eV
ne = 1e16 # Electron density in m^-3 (assumed constant for simplicity)
N_gas = 3e21 # Neutral argon density (0.1 Torr at 300K)
E_ex = 11.5 # Excitation energy (ground to metastable)
E_iz = 15.76 # Ionization energy (ground state)
E_si = 4.26  # Stepwise ionization energy (15.76 - 11.5)

# --- 2. Rate Coefficients (Simplified Arrhenius-like) ---
# k = A * exp(-E/Te)
k_ex = 1e-15 * np.exp(-E_ex / Te)  # Ar + e -> Arm + e
k_si = 1e-13 * np.exp(-E_si / Te)  # Arm + e -> Ar+ + 2e (Stepwise)
k_dr = 5e-14 * (Te**-0.66)        # Recombination / Losses

# --- 3. Rate Equation Formulation ---
def plasma_dynamics(y, t, ne, N_gas, k_ex, k_si, k_dr):
    n_gs, n_m, n_i = y # Ground, Metastable, Ion
    
    # Rate equations (d/dt)
    # 1. Ground State: Losses to excited/ionized
    dn_gs_dt = -ne * n_gs * k_ex
    # 2. Metastable State: Gains from excitation, losses to ionization
    dn_m_dt = (ne * n_gs * k_ex) - (ne * n_m * k_si) - (k_dr * n_m)
    # 3. Ion Density: Stepwise ionization + others
    dn_i_dt = (ne * n_m * k_si) 
    
    return [dn_gs_dt, dn_m_dt, dn_i_dt]

# --- 4. Simulation Execution ---
y0 = [N_gas, 0.0, 1e10] # Initial densities
t = np.linspace(0, 1e-3, 1000) # Time span (seconds)

# Solve ODE
solution = odeint(plasma_dynamics, y0, t, args=(ne, N_gas, k_ex, k_si, k_dr))
n_gs, n_m, n_i = solution.T

# --- 5. Plotting Results ---
plt.figure(figsize=(10,6))
plt.plot(t*1e6, n_gs, label='Ground State Ar')
plt.plot(t*1e6, n_m, label='Metastable $Ar_m$')
plt.plot(t*1e6, n_i, label='Ion $Ar^+$')
plt.xlabel('Time ($\mu s$)')
plt.ylabel('Density ($m^{-3}$)')
plt.yscale('log')
plt.legend()
plt.title(f'Stepwise Ionization of Argon (Te={Te}eV)')
plt.grid(True)
plt.show()
