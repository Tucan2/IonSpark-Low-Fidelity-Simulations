#Made by Aditya
#Very very questionable
#Slightly updated, so a little bit more accurate/less questionable

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, k, m_e
from scipy.integrate import odeint

#From setup
R_pin = 0.5e-3 #in m
L_pin = 100e-3 #in m
V_applied = 5000 #in V

#Assume plasma sheath same diameter as quartz tube
Rt = 5e-3
d_sheath = 1e-2

#From: https://www.nature.com/articles/s41467-025-60607-6
k_geo = 1.680 * (R_pin/Rt+0.468)**(-1.066)

#from https://pubs.aip.org/aip/pop/article/25/4/043113/903792/A-universal-formula-for-the-field-enhancement
gamma = (2 * L_pin/ R_pin)/ (np.log((4 * L_pin)/ R_pin) - 2)
F_local = gamma * V_applied / (k_geo * R_pin) / 1e9 

phi_W = 4.55 #eV, Internet

#Textbook
B0 = 6.83089 #eV^-1.5 * nm^-1
A0 = 1.5414*10**(-6) #A/eV
t0 = 1.06131 #correction for non-triangular barrier (??)
Q = 0.35999 #eV nm

#Forbes-Deane approx (?) of Fowler–Nordheim
def get_J_FN_precise(F, phi):
    v = (2.68754/phi)**(1/2)    
    #Schottky lowering factor (?)
    coeff_term = ((phi**2)/(4 * Q))**v 
    
    pre_exp = (A0/(phi * t0**2)) * coeff_term * (F**(2 - v))

    exponent = np.exp(-(B0 * (phi**1.5))/F)
    
    J = pre_exp * exponent
    return J

J_FN = get_J_FN_precise(F_local, phi_W)
I_FN = J_FN * np.pi * R_pin**2

v_drift = 5e4 #approximate, see https://www.researchgate.net/publication/200702789_An_update_of_argon_inelastic_cross_sections_for_plasma_discharges

#Net rate of electron generation due to electron field emission
S_e = J_FN / (e * v_drift)# m^-3/s 

#Assuming ideal gas law, T=300K
n_Ar = 6.4e21
n_tungsten = 6.32*10**28 #n_e = Z * rho * A/M
from scipy.constants import e, m_e, hbar
T_e = (hbar**2 / (2 * m_e)) * (3 * np.pi**2 * n_tungsten)**(2/3)/e  #assume electrons are emitted at fermi energy, eV
print("HI")
print(T_e)
def plasma_dynamics(y, t):
    n_e, n_m, n_i, n_n = y
    n_e = max(n_e, 0)
    n_m = max(n_m, 0)
    n_i = max(n_i, 0)
    n_n = max(n_n, 0)
    
    #Basic/bad idea: Vfelt = Vpower_source + Velectrons but Velectrons actually negative so get below
    #Thus, can get proportional electron temperature; unlikely to be accurate
    A_eff = np.pi * Rt**2
    current_inst = n_e * e * v_drift * A_eff
    V_gap = max(V_applied - current_inst * 1*10**6, 200) 
    T_e_collisional = (V_gap / V_applied) * T_e

    #From https://pubs-aip-org.proxy2.library.illinois.edu/aip/pop/article/13/5/053502/1032298/On-the-multistep-ionizations-in-an-argon 
    ioniz_rate = 2.3*10**(-14)*((T_e_collisional)**(0.68)*((np.e)**(-15.76/T_e_collisional))) * n_n * n_e * 10**(-6) #rate constant in cm^-3
    exc_rate = 1.4*10**(-14)*((T_e_collisional)**(0.71))*((np.e)**(-13.2/T_e_collisional)) * n_n * n_e * 10**(-6)
    step_rate = 1.8*10**(-13)*((T_e_collisional)**(0.61))*((np.e)**(-2.61/T_e_collisional)) * n_m * n_e * 10**(-6)
    recomb_rate = 3.9*10**(-13)*((T_e_collisional)**(0.71)) * n_e * n_i * 10**(-6)
    
    #from some questionable calculations based on following:
    #ui = uo * no/nAr, uo = https://iopscience.iop.org/article/10.1088/0370-1328/80/3/307
    #ambipolar coeff D = ui * Te 
    #diffusion length 1/l^2 = 1/(R/2.405)^2 + 1/(L/pi)^2
    #time scale T = l^2/D; assumed loss same as time scale
    #loss rate probably needs to be relaculated, but it seems to be correct order of magnitude with 10^-6
    loss = 3.19*10**(-3) 
    
    dn_e_dt = S_e + ioniz_rate - recomb_rate + step_rate - n_e/loss 
    dn_m_dt = exc_rate - step_rate - n_m/loss
    dn_i_dt = ioniz_rate+step_rate - recomb_rate - n_i/loss
    dn_n_dt = -1*ioniz_rate-1*exc_rate+recomb_rate+n_i/loss+n_m/loss
    
    return [dn_e_dt, dn_m_dt, dn_i_dt, dn_n_dt]

y0 = [0.1, 0.1, 0.1, n_Ar]  #Initial, simply want (not n_Ar) nonzero for numerics reasons; that being said, questionable
t = np.linspace(0, 1*10**(-6), 5000)  # 1 us total

sol = odeint(plasma_dynamics, y0, t, rtol=1e-4, atol=1e-4)

n_e = sol[:,0]
n_m = sol[:,1]
n_i = sol[:,2]
n_n = sol[:,3]

#Bohm velocity (ion speed at sheath or something to that effect)
m_Ar = 39.948 * 1.6605e-27  # kg
v_B = np.sqrt(T_e*e/m_Ar)

#Ignores some sheath effects, possibly; approximated to be 0.5
I_plasma_t = 0.5*n_e * v_B * np.pi * Rt**2

print(f"I_FN = {I_FN*1e3:.3f} mA")

plt.figure(figsize=(10,6))
plt.plot(t*10**6, I_FN*1e3*np.ones_like(t), 'k--', lw=2, label=f'I_FN')
plt.plot(t*10**6, I_plasma_t*1e3, 'm-', lw=3, label='Plasma current')
plt.xlabel('Time (us)')
plt.ylabel('Current (mA)')
plt.legend()
plt.grid()
plt.title('Current versus Time')

plt.tight_layout()
plt.show()
