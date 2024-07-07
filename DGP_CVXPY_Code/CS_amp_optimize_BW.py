import cvxpy as cp
import numpy as np

# Constants
VTO = 0.5 # Threshold voltage
KP = 50e-6   # Transconductance parameter (μnCox)
Cox = 2.3e-3
mu_n = KP/Cox #Mobility in m^2/Vs

# Given parameters
Rs = 10e3  # Source resistance in Ohms
RD = 5e3  # Drain resistance in Ohms
CL = 1e-12  # Load capacitance in Farads


# Create optimization variables
L = cp.Variable(pos=True)  # Width of the transistor
VOV = cp.Variable(pos=True) # Gate-source voltage

# Specification 
ID = 500e-6
AV = 4

W = AV * L  / (RD * mu_n * VOV * Cox)
Rout = RD
omega_3dB = 1 / (((2/3)*cp.square(L)*AV*Rs/(RD*mu_n*VOV)) + AV*CL*VOV/(2*ID))

# Calculate open-circuit time constants (OCT)

objective_fn = ((omega_3dB))

constraints = [
    L >= 1e-6,  
    L <= 4e-6,
    VOV >= 100e-3, 
    VOV <= 0.9,
]

# Define the problem and solve
assert objective_fn.is_log_log_concave()

assert all(constraint.is_dgp() for constraint in constraints)

problem = cp.Problem(cp.Maximize(objective_fn), constraints)


print(problem)
print("Is this problem DGP?", problem.is_dgp())

problem.solve(gp=True) # Solve with DCP
print("Status:", problem.status)



# Print results
print("Status:", problem.status)
print("Optimal omega_3dB:", problem.value)
print("Optimal W:", W.value)
print("Optimal L:", L.value)
print("Optimal VOV:", VOV.value)
print("Optimal gm:", 2*ID/VOV.value)
