import cvxpy as cp
import numpy as np

# Constants
mu = 600  # A/V^2, mobility
Cox = 1.72653e-6 # oxide capacitance per unit area (F/m^2) Eo.Er/tox
Vth = 0.7        # V, threshold voltage
mu_Cox = mu * Cox # A/V^2, mobility capacitance product
Lambda = 0.03       # Channel length modulation
LD=0.2e-6 # m, Lateral diffusion
CJ=271e-6 #        ; Zero-bias bulk junction capacitance (F/m2)
CJSW=600e-12 #      ; Zero-bias sidewall bulk junction capacitance (F/m)
Phi=0.688       #  ; Surface inversion potential
Lmin = 0.8e-6       # m, minimum channel length
Wmin = 2e-6         # m, minimum channel width
Cl = 10e-12      # F, load capacitance
VDD = 5          # V, supply voltage

#Initial values fro variables
ID = 5e-6 # A, drain current
W = 2e-6 # m, width of the transistor
L = 0.8e-6 # m, length of the transistor
vgs = 2.5 # V, gate-source voltage

# Derived expressions
#ID = mu_Cox * (W / (2 * L)) * cp.square(vgs - Vth)
gm = cp.sqrt(2 * mu_Cox * (W / L) * ID) # Transconductance
vdsat = cp.sqrt(2 * ID /(mu_Cox * (W / L) )) # Vdsat
vdsat = vgs - Vth  # Vgs - Vth = Vdsat   # Vdsat

Csbo = CJ*LD*W + CJSW*(2*LD + W) # zero bia Source bulk capacitance
Csb = Csbo / cp.sqrt(1-vdsat/Phi) # Source bulk capacitance
Cdb = Csb # Drain bulk capacitance
Cgs = Cox * W * L # Gate-source capacitance
Cd = 2 * Cox * W * LD # Drain-source and Drain Gate capacitance


Av = cp.sqrt(2 * mu_Cox * (W / L) * 1 / (ID * cp.square(Lambda)))  # Voltage gain
BW = Lambda * ID / (2 * np.pi * (Cl + Cd))  # Bandwidth
power = VDD * ID


# Create optimization variables
W = cp.Variable(pos=True)  # Width of the transistor
L = cp.Variable(pos=True) # Length of the transistor
ID = cp.Variable(pos=True) # Drain current
vgs = cp.Variable(pos=True) # Gate-source voltage
vdsat = cp.Variable(pos=True) # drain - source voltage

# GBW calculation
GBW = gm / (2 * np.pi * (Cl+Cd))

# Constraints for gm
constraints = [
    W >= Lmin,  # Width must be positive and reasonable
    L >= Wmin,  # Length must be positive and reasonable
    W >= L, # Width must be greater than length
    L <= 2e-6, # Length must be less than 10um
    W <= 500e-6, # Width must be greater than length
    vgs >= Vth + 200e-3, # Gate-source voltage must be greater than threshold voltage
    vgs <= VDD, # Gate-source voltage must be less than supply voltage
    ID >= 1e-6,  # Drain current must be positive and reasonable
    ID <= 10e-3,
    cp.sqrt(2 * mu_Cox * (W / L) * 1 / (ID * cp.square(Lambda))) >= 10,  # Voltage gain must be at least 10
    cp.sqrt(2 * mu_Cox * (W / L) * 1 / (ID * cp.square(Lambda))) <= 100,  # Voltage gain must be at max 100
    Lambda * ID / (2 * np.pi * (Cl + Cd)) >= 100000,  # Bandwidth must be at least 1 KHz
    vdsat >= 200e-3 # Vdsat must be greater than 200mV

]


# Calculations  of params
# Objective: maximize GBW
objective_fn = (gm / (2 * np.pi * (Cl+Cd)))

assert objective_fn.is_log_log_concave()
# Define the problem and solve
assert all(constraint.is_dgp() for constraint in constraints)


problem = cp.Problem(cp.Maximize(objective_fn), constraints)


print(problem)
print("Is this problem DGP?", problem.is_dgp())

problem.solve(gp=True) # Solve with DCP

# Re-Calculations  of params
gm = cp.sqrt(2 * mu_Cox * (W / L) * ID) # Transconductance
gds = Lambda * ID # Output conductance
GBW = gm / (2 * np.pi * (Cl+Cd))
Av = cp.sqrt(2 * mu_Cox * (W / L) * 1 / (ID * cp.square(Lambda)))  # Voltage gain
Av_d = gm/gds  # Voltage gain
BW = Lambda * ID / (2 * np.pi * (Cl + Cd))  # Bandwidth
BW_d = gds /  (2 * np.pi * (Cl + Cd)) # Bandwidth
vdsat = cp.sqrt(2 * ID /(mu_Cox * (W / L) )) # Vdsat
GBW_d = Av * BW # Gain Bandwidth product
# Print results
print("Status:", problem.status)
print("Optimal GBW:", problem.value)
print("Optimal GBW_d:", GBW_d.value)
print("Optimal gm:", gm.value)
print("Optimal gds:", gds.value)
print("Optimal W:", W.value)
print("Optimal L:", L.value)
print("Optimal Av:", Av.value)
print("Optimal Av_d:", Av_d.value)
print("Optimal BW:", BW.value)
print("Optimal BW_d:", BW_d.value)
print("Optimal ID:", ID.value)
print("Optimal vgs:", vgs.value)
print("Optimal vdsat:", vdsat.value)
print("Optimal power:", VDD * ID.value)



