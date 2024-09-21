import cvxpy as cp
import numpy as np

#Define Constants
L = 1e-6  # Channel length (m)
VTO = 0.5  # Threshold voltage (Volts)
KP = 50e-6  # Transconductance parameter (A/V^2)
Cox = 2.3e-3  # Oxide capacitance per unit area (F/m^2)
mu_n = KP / Cox  # Mobility (m^2/Vs)

RS = 10e3  # Source resistance (Ohms)
CL = 1e-12  # Load capacitance (F)
gain = 4  # Amplifier gain
f3db = 80e6  # 3dB frequency (Hz)
Omega_3db = 2 * np.pi * f3db  # Angular frequency (rad/s)

Rout = 5e3  # Output resistance (Ohms)
 
Vov = cp.Variable(pos=True) # Overdrive voltage (Volts)
t = cp.Variable(pos = True) # Variable for transforming non-convex function to a convex form

Varx = (1/2) * Omega_3db * gain * CL 
Vary = (2/3) * (L**2) * gain * Omega_3db * RS * (1 / (mu_n  * Rout))

Id = Varx * Vov / (1 - (Vary / Vov))
# Id = (1/2 * Omega_3db * gain * Vov.value * CL) / ((1 - (2/3)  * cp.square(L) * gain * Omega_3db * RS * (1/(mu_n * Vov.value * Rout))) )

# Define the objective function
objective_fn = t

# Define constraints
constraints = [
    Vov >= 0.125,   
    Vov <= 2,
    (Vary)/Vov + Varx * Vov/t <= 1 # contsraint after transforming Varx*Vov/(1 - Vary/Vov) to a convex form
]

# Check the curvature of the problem
print("Curvature of Objective", objective_fn.log_log_curvature)
# Set up and solve the problem
assert objective_fn.is_log_log_convex()
assert all(constraint.is_dgp() for constraint in constraints)
problem = cp.Problem(cp.Minimize(objective_fn), constraints)
print("Is this problem DGP?", problem.is_dgp())
problem.solve(gp=True)  # Solve with geometric programming
print("Status:", problem.status)

print(f"Problem_Value: {problem.value*1e3:.3f} mA")
print(f"Id: {(Id.value)*1e3:.3f} mA")
print(f"Vov: {Vov.value*1e3:.3f} mV")
