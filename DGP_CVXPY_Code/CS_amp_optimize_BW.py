import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Constants and assumptions for 1um technology
L = 1e-6  # Channel length (m)
VTO = 0.5  # Threshold voltage (Volts)
KP = 50e-6  # Transconductance parameter (A/V^2)
Cox = 2.3e-3  # Oxide capacitance per unit area (F/m^2)
mu_n = KP / Cox  # Mobility (m^2/Vs)

#Design Variablea
RS = 10e3 # Source resistance (Ω)
Vov = 0.2 # Overdrive voltage (Volts)
CL = 1e-12  # Load capacitance (F)
gain = 4

# Define the optimization variable
W = cp.Variable(pos=True)  

# Define expressions  
gm = mu_n * Cox * (W/L) * Vov
Rout = gain/gm
Cgs = 2/3 * Cox * W * L
f3db = 1 / (2 * np.pi * (Cgs * RS + CL * Rout))
Id = 0.5 * mu_n * Cox * (W/L) * Vov**2

# Define the objective function
objective_fn = f3db

# Define constraints
constraints = [
    W >= 1e-6,   
    W <= 500e-6, 
]

# Set up and solve the problem
assert objective_fn.is_log_log_concave()
assert all(constraint.is_dgp() for constraint in constraints)
problem = cp.Problem(cp.Maximize(objective_fn), constraints)
print("Is this problem DGP?", problem.is_dgp())
problem.solve(gp=True)  # Solve with geometric programming
print("Status:", problem.status)

# Print results
print(f"Optimal W: {W.value*1e6:.2f} µm")
print(f"Optimal gm: {gm.value:.2e} S")
print(f"f3db: {f3db.value/1e6:.2f} MHz")
print(f"Gain: {gm.value*Rout.value:.2f} ")
print(f"Id: {Id.value*1e3:.3f} mA")
print(f"Rout: {Rout.value*1e-3:.3f} kOhm")
print(f"Tout: {CL*Rout.value*1e9:.3f} ns")
print(f"Tin: {RS*Cgs.value*1e9:.3f} ns")
