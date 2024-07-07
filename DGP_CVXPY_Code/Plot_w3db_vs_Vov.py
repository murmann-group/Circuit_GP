import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
CL = 1e-12          # Capacitance in Farads
A_v0 = 4           # Gain
RD = 5e3            # Drain resistance in ohms
Rs = 10e3           # Source resistance in ohms
KP = 50e-6   # Transconductance parameter (μnCox)
Cox = 2.3e-3
mu_n = KP/Cox 
ID = 500e-6         # Drain current in Amps

# Define the range for VOV
VOV = np.linspace(0.01, 1, 100)  # Overdrive voltage range from 150mV to 800mV

# Define the function for omega_3dB
def omega_3dB(L, VOV):
    term1 = (2 / 3) * (L**2 * A_v0 * Rs / (RD * mu_n * VOV))
    term2 = (A_v0 * CL * VOV / (2 * ID))
    return 1 / (term1 + term2)

# Calculate omega_3dB for L = 1um and L = 1.5um
omega_3dB_L1 = omega_3dB(1e-6, VOV)
omega_3dB_L1_5 = omega_3dB(1.5e-6, VOV)

# Plot the function for omega_3dB
plt.figure(figsize=(10, 6))
plt.plot(VOV, omega_3dB_L1, label='ω_3dB vs V_OV for L=1μm')
plt.plot(VOV, omega_3dB_L1_5, label='ω_3dB vs V_OV for L=1.5μm')
plt.xlabel('V_OV (V)')
plt.ylabel('ω_3dB (rad/s)')
plt.title('ω_3dB as a function of Overdrive Voltage (V_OV)')
plt.legend()
plt.grid(True)
# plt.show()

# Using numerical differentiation
domega_3dB_dVOV_L1 = np.gradient(omega_3dB_L1, VOV)
domega_3dB_dVOV_L1_5 = np.gradient(omega_3dB_L1_5, VOV)

# Plotting the derivatives
plt.figure(figsize=(10, 6))
plt.plot(VOV, domega_3dB_dVOV_L1, label='L = 1 µm')
plt.plot(VOV, domega_3dB_dVOV_L1_5, label='L = 1.5 µm')
plt.xlabel('$V_{OV}$ (V)')
plt.ylabel('d$\omega_{3dB}$/d$V_{OV}$ (rad/s/V)')
plt.title('Derivative of 3dB Bandwidth vs $V_{OV}$')
plt.legend()
plt.grid(True)
plt.show()