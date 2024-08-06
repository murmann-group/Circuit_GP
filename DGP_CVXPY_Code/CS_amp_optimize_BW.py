import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Constants and assumptions for 1um technology
L = 1e-6  # Channel length (m)
CL = 1e-12  # Load capacitance (F)

VTO = 0.5  # Threshold voltage in Volts
KP = 50e-6  # Transconductance parameter in A/V^2
Cox = 2.3e-3  # Oxide capacitance per unit area in F/m^2
mu_n = KP / Cox  # Mobility in m^2/Vs
RS = 10e3 # Source resistance resistance (Ω)
VoV = 0.2


# Define the optimization variable
W = cp.Variable(pos=True)

# Define expressions for our metrics
gain = 4
gm = mu_n * Cox * (W/L) * VoV
Rout = gain/gm
Cgs = 2/3 * Cox * W * L
f3db = 1 / (2 * np.pi * (Cgs*RS + CL*Rout))
Id = 0.5 * mu_n * Cox * (W/L) * VoV**2
# Id = gm * VoV / 2

#Contraint on the minimum f3db
f3db_min = 10e6

# Define the objective function
objective_fn = f3db

# Define constraints
constraints = [
    W >= 1e-6,  # Minimum width of 1 µm
    W <= 500e-6,  # Maximum width of 800 µm
    f3db >= f3db_min,  # Minimum f3db 

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
print(f"Gain: {20*np.log10(gm.value*Rout.value):.2f} dB")
print(f"Id: {Id.value*1e3:.2f} mA")
print(f"Rout: {Rout.value*1e-3:.2f} kOhm")
print(f"Tout: {CL*Rout.value*1e12:.2f} ns")
print(f"Tin: {RS*Cgs.value*1e12:.2f} ns")
