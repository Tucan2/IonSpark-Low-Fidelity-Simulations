import jax
import numpy as np
from pycharge import Charge, potentials_and_fields
from scipy.constants import e, m_e, pi
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eedf.Sommerfield import eta_thermionic
jax.config.update("jax_enable_x64", True)

R = 0.001
L = 0.012
n_e = 50
q_e = -e
T = 2500.0 
ev_max = 10.0

theta = np.random.uniform(0, 2*pi, n_e)
r = R * np.sqrt(np.random.uniform(0, 1, n_e))
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.random.uniform(0, L, n_e)

energies = []
p = eta_thermionic(np.linspace(0, ev_max, 1000) * e, T)
p /= np.sum(p)
#print(p)
for _ in range(n_e):
    energies.append(np.random.choice(np.linspace(0, ev_max, 1000), p=p))
energies = np.array(energies)

speeds = np.sqrt(2 * energies * e / m_e)
a = np.random.uniform(0, 2*pi, n_e)
b = np.random.uniform(0, pi, n_e)
vx = speeds * np.sin(b) * np.cos(a)
vy = speeds * np.sin(b) * np.sin(a)
vz = speeds * np.cos(b)

def trajectory(p0, v0):
    return lambda t: p0 + v0 * t
charges = []
for i in range(n_e):
    p0 = jax.numpy.array([x[i], y[i], z[i]])
    v0 = jax.numpy.array([vx[i], vy[i], vz[i]])
    charges.append(Charge(trajectory(p0, v0), q_e))
#field = potentials_and_fields(charges)(0, 0, L/2, 0)
#print(field.electric)
def field(x,y,z,t):
    return jax.jit(potentials_and_fields(charges))(x,y,z,t)
#print(field(0, 0, L/2, 0).electric)
