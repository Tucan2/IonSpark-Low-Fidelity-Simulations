import numpy as np

def tip_field(V,r,R):
    # E ~ 2V / [r * ln(2 * R_tube / r)]
    E = (2 * V) / (r * np.log((2 * R) / r))
    return E

V = 10000.0        
r_pin = 20e-6    
R_tube = 0.003   
print(tip_field(V,r_pin,R_tube))
