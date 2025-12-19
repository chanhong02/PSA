import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Model & Map parameters ---
Jcp = 5e-5
kt = 0.0153
kv = 0.0153
Rmm = 0.82
eta_mm = 0.985
eta_cp = 0.8

R_air = 286.9
rho_air = 1.23
d_cp = 0.2286
gamma = 1.4
Psm = Pout = 750000

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

Cp_air = gamma * R_air / (gamma - 1)

# --- Initial conditions ---
Tatm = 298.15
Patm = 101325
Vmm = 380
omega0 = 0

# --- Map functions ---

def psi_value(Cp_air, Tatm, Pout, Patm, U_cp, gamma):
    return (Cp_air * Tatm / (0.5*(U_cp**2))) * (((Pout / Patm)**((gamma - 1) / gamma)) - 1)

def mach_number(U_cp, Tatm):
    return U_cp / np.sqrt(gamma * R_air * Tatm)

def phi_max(M):
    return a4*M**4 + a3*M**3 + a2*M**2 + a1*M + a0

def beta_poly(M):
    return b2*M**2 + b1*M + b0

def psi_max(M):
    return c5*M**5 + c4*M**4 + c3*M**3 + c2*M**2 + c1*M + c0

def phi_value(U_cp, Tatm, psi):
    M = mach_number(U_cp, Tatm)
    phi_m = phi_max(M)
    beta_m = beta_poly(M)
    psi_m = psi_max(M)
    
    # psi 제한
    psi = min(psi, psi_m)

    exponent = beta_m * (psi / psi_m - 1)  # overflow 방지
    
    return phi_m * (1 - np.exp(exponent))

# --- Compressor torque ---
def compressor_torque(omega):
    # Clamp omega to avoid divergence
    # omega = np.clip(omega, 1e-3, 1000)
    U_cp = omega * d_cp / 2
    P_ratio = Psm / Patm
    term = (P_ratio ** ((gamma - 1) / gamma)) - 1
    psi = psi_value(Cp_air, Tatm, Pout, Patm, U_cp, gamma)
    phi = phi_value(U_cp, Tatm, psi)
    Wcp = (np.pi * d_cp * rho_air**2 * U_cp * phi) / 4
    torque = Cp_air * omega * Tatm / eta_cp * term * Wcp
    return torque, Wcp

# --- Motor torque ---
def motor_torque(omega):
    return eta_mm * kt / Rmm * (Vmm - kv * omega)

# --- Differential equation ---
def d_omega_dt(t, omega):
    tau_mm = motor_torque(omega[0])
    tau_cp, _ = compressor_torque(omega[0])
    return [(tau_mm - tau_cp) / Jcp]

# --- Simulation ---
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(d_omega_dt, t_span, [omega0], t_eval=t_eval, method='RK45', max_step=0.01)

# --- Post-processing ---
Wcp_vals = [compressor_torque(omega)[1] for omega in sol.y[0]]

# --- Plotting ---
fig, axs = plt.subplots(2, 1, figsize=(12, 6))

# Speed plot
axs[0].plot(sol.t, sol.y[0] * 60 / (2 * np.pi))
axs[0].set_title("Compressor Speed vs Time")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Speed [rad/s]")
axs[0].grid(True)

# Mass flow rate plot
axs[1].plot(sol.t, Wcp_vals)
axs[1].set_title("Compressor Outlet Mass Flow Rate vs Time")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Mass Flow Rate [kg/s]")
axs[1].grid(True)

plt.tight_layout()
plt.show()
