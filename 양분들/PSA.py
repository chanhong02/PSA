import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#  Model parameter (main Table 3)
Jcp = 5e-5
kt = 0.0153  # Torque constant [Nm/A]
kv = 0.0153  # Back EMF constant [V/(rad/s)]
Rmm = 0.82  # Motor resistance [Ohm]
eta_mm = 0.985  # Motor mechanical efficiency
eta_cp = 0.8  # Isentropic efficiency

# Map parameter (Appendix Table 2)
R_air = 286.9               # J/(kg∙K)
rho_air = 1.23  # Density of air [kg/m3]      
d_cp = 0.2286  # Compressor diameter [m]
gamma = 1.4  # Adiabatic constant
Psm = 750000  # Supply pressure [Pa]

a0 = 2.21195e-3
a1 = -4.63685e-5
a2 = -5.36235e-4
a3 = 2.70399e-4
a4 = -3.69906e-5
b0 = 2.44419
b1 = -1.34837
b2 = 1.76567
c0 = 0.43331
c1 = -0.68344
c2 = 0.80121
c3 = -0.42937
c4 = 0.10581
c5 = -9.78755e-3

Cp_air = gamma * R_air / (gamma - 1)              # C_p- C_v = R

# --- Simulation initial value  ---
Tatm = 298.15  # Ambient temperature [K]
Patm = 101325  # Atmospheric pressure [Pa]

Vmm = 380    # Main Table.5  PSA submotor voltage
omega0 = 0  # Initial speed [rad/s]


def compressor_torque(omega):
    U_cp = omega * d_cp / 2
    P_ratio = Psm / Patm
    term = (P_ratio ** ((gamma - 1) / gamma)) - 1
    Wcp = (np.pi * d_cp * rho_air**2 * U_cp) / 4
    torque = Cp_air * omega * Tatm / eta_cp * term * Wcp
    return torque, Wcp

def motor_torque(omega):
    return eta_mm * kt / Rmm * (Vmm - kv * omega)

# Differential equation
def d_omega_dt(t, omega):
    tau_mm = motor_torque(omega[0])
    tau_cp, _ = compressor_torque(omega[0])
    return [(tau_mm - tau_cp) / Jcp]

# Time span for simulation
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve ODE
sol = solve_ivp(d_omega_dt, t_span, [omega0], t_eval=t_eval, method='RK45')

# Calculate Wcp for each omega
Wcp_vals = [compressor_torque(omega)[1] for omega in sol.y[0]]


