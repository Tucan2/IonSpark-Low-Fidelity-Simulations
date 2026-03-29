#Made by Aditya
#Very very questionable
#Slightly updated, so a little bit more accurate/less questionable

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, k, m_e
from scipy.integrate import solve_ivp
import sys

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
I_FN = I_FN * 10**115
def plasma_dynamics(t, y):
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

y0 = [I_FN, 0.12, I_FN, n_Ar]  #Initial, simply want (not n_Ar) nonzero for numerics reasons; that being said, questionable
t = np.linspace(0, 1*10**(-6), 5000)  # 1 us total

sol = solve_ivp(plasma_dynamics, [t[0], t[-1]], y0, rtol=1e-4, atol=1e-4)
n_e = sol['y'][0]
n_m = sol['y'][1]
n_i = sol['y'][2]
n_n = sol['y'][3]

#Bohm velocity (ion speed at sheath or something to that effect)
m_Ar = 39.948 * 1.6605e-27  # kg
v_B = np.sqrt(T_e*e/m_Ar)

#Ignores some sheath effects, possibly; approximated to be 0.5
numElectrons = n_e * np.pi*Rt**2
numIons = n_i*np.pi*Rt**2
numMetastable = n_m*np.pi*Rt**2
numAr = n_n * v_B*np.pi*Rt**2
#Assume ions, electrons, metastable, and plasma as determined by above

np.set_printoptions(threshold=sys.maxsize)
print(round(numElectrons[0]*10**7))
print(round(numIons[0]*10**7))
print(round(numMetastable[0]*10**7))
print(round(numAr[0]/(8*10**18)))


import numpy as np
import pandas as pd

np.random.seed(42)

target_electrons = round(numElectrons[0]*10**7)
target_ions = round(numIons[0]*10**7)
target_metastable = round(numMetastable[0]*10**7)
target_argon = round(numAr[0]/(8*10**18))

time_steps = 150             
dt = 0.033 

data = []
next_particle_id = 1

def generate_particle_data(pid, start_type, transition_step, final_type, has_poscharge=False, has_negcharge=False):
    pos = np.array([np.random.uniform(-10.0, 10.0), np.random.normal(-10.0, 10.0), np.random.normal(-10.0, 10.0)])
    
    if has_negcharge:
        vel = np.array([v_drift*dt * 1e-2+np.random.normal(-2.0, 2.0), np.random.normal(-2.0, 2.0), np.random.normal(-2.0, 2.0)])
        accel = np.array([5.0, np.random.normal(-0.5, 0.5), np.random.normal(-0.5, 0.5)])
    elif has_poscharge:
        vel = np.array([-1*v_drift * 1e-4+np.random.normal(-2.0,2.0), np.random.normal(-2.0, 2.0), np.random.normal(-2.0, 2.0)])
        accel = np.array([-1*5.0, np.random.normal(-0.5, 0.5), np.random.normal(-0.5, 0.5)])
    else:
        vel = np.array([np.random.uniform(-2.0, 2.0), np.random.normal(-2.0, 2.0), np.random.normal(-2.0, 2.0)])
        accel = np.array([np.random.normal(-0.5, 0.5), np.random.normal(-0.5, 0.5), np.random.normal(-0.5, 0.5)])

    current_type = start_type
    
    for step in range(time_steps):
        t_curr = step * dt
        
        # Transition logic: Argon -> Metastable -> Ion
        if step >= transition_step:
            current_type = final_type

        data.append({
            'particle_id': pid,
            'is_ionized': current_type,
            'time': round(t_curr, 3),
            'pos_x': round(pos[0], 4),
            'pos_y': round(pos[1], 4),
            'pos_z': round(pos[2], 4)
        })
        vel += accel * dt
            
        pos += vel * dt
        
        i = 0
        while i < len(accel):
            accel[i] += np.random.normal(-0.5, 0.5)
            i += 1

for _ in range(target_electrons):
    generate_particle_data(next_particle_id, 2, 0, 2, has_negcharge=True)
    next_particle_id += 1

for _ in range(target_ions):
    trans = np.random.randint(35, 150)
    generate_particle_data(next_particle_id, 0, trans, 1, has_poscharge=True)
    next_particle_id += 1

for _ in range(target_metastable):
    trans = np.random.randint(20, 100)
    generate_particle_data(next_particle_id, 0, trans, 3)
    next_particle_id += 1

for _ in range(target_argon):
    generate_particle_data(next_particle_id, 0, time_steps + 1, 0)
    next_particle_id += 1

df = pd.DataFrame(data)
df.sort_values(by=['time', 'particle_id'], inplace=True)
filename = 'data.csv'
df.to_csv(filename, index=False)

print(f"Final Step Composition: {target_electrons}e-, {target_ions} Ions, {target_metastable} Metastables, {target_argon} Ar")