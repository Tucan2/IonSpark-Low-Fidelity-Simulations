import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- Rate Coefficients (m^3/s or m^6/s) and Parameters ---
# Te: Electron Temperature in eV
# k: Rate coefficients derived from Arrhenius-type equations (e.g.,)
def get_rates(Te):
    # Electron impact ionization Ar + e -> Ar+ + 2e (Direct)
    k_dir = 1.235e-13 * np.exp(-18.687 / Te) # cm^3/s to m^3/s
    # Excitation Ar + e -> Ar* + e (Metastable)
    k_exc = 3.712e-14 * np.exp(-15.06 / Te)
    # Stepwise Ionization Ar* + e -> Ar+ + 2e
    k_step = 2.05e-14 * np.exp(-4.95 / Te)
    # Metastable pooling Ar* + Ar* -> Ar + Ar+ + e
    k_pool = 6.2e-16 # m^3/s
    
    return k_dir*1e-6, k_exc*1e-6, k_step*1e-6, k_pool*1e-6

# --- Differential Equations ---
def odes(y, t, Te, n_e):
    Ar, Arm, Arp = y
    k_dir, k_exc, k_step, k_pool = get_rates(Te)
    
    # Rates
    r_dir = k_dir * n_e * Ar
    r_exc = k_exc * n_e * Ar
    r_step = k_step * n_e * Arm
    r_pool = k_pool * Arm**2
    
    # Balance Equations
    dAr_dt = -r_dir - r_exc + r_pool # Ground state loss
    dArm_dt = r_exc - r_step - 2*r_pool # Metastable density
    dArp_dt = r_dir + r_step + r_pool # Ion density
    
    return [dAr_dt, dArm_dt, dArp_dt]

# --- Simulation Setup ---
n_total = 3.3e22 # Density at 1 Torr, 300K (m^-3)
n_e = 1e16 # Electron density (m^-3)
Te = 2.0 # Electron Temperature (eV)
y0 = [n_total, 0, 0] # Initial conditions [Ar, Arm, Arp]
t = np.linspace(0, 1e-3, 1000) # Time span (s)

# --- Solve ODEs ---
sol = odeint(odes, y0, t, args=(Te, n_e))
Ar, Arm, Arp = sol.T

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(t*1e6, Ar, label='Ar (Ground)')
plt.plot(t*1e6, Arm, label='Ar* (Metastable)')
plt.plot(t*1e6, Arp, label='Ar+ (Ion)')
plt.xlabel('Time ($\mu$s)')
plt.ylabel('Density ($m^{-3}$)')
plt.title('Stepwise Ionization of Argon (0D Model)')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()