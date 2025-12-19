import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#  Model parameter (main Table 3)
J_cp = 5e-5
k_v = 0.0153
k_t = 0.0153
R_mm = 0.82
eta_mm = 0.985
eta_cp = 0.80

# Map parameter (Appendix Table 2)
R_air = 286.9               # J/(kg∙K)
rho_air_std = 1.23          # kg/m³
d_cp = 0.2286 * 2
gamma_air = 1.4
P_out = 101325*2.7  # main pg 9. (3.5) working pressure of PSA  (750 Kpa)

# Regression coeffs (Appendix Table 2)
A_coeffs = np.array([-3.69906e-5, 2.70399e-4, -5.36235e-4, -4.63685e-5, 2.21195e-3])
B_coeffs = np.array([1.76567,-1.34837, 2.44419])
C_coeffs = np.array([-9.78755e-3, 0.10581, -0.42937, 0.80121, -0.68344, 0.43331])

Cp_air = gamma_air * R_air / (gamma_air - 1)              # C_p- C_v = R

# --- Simulation initial value  ---
T_in = 300.15
P_in = 101325

V_mm_base = 380    # Main Table.5  PSA submotor voltage






# --- CP MAP 1 BLOCK (Appendix Eq. 21-27) ---
def get_static_performance(omega_cp, T_in, P_in, P_out):

    T_std = 288         # appendix >> 288
    P_std = 101325
    theta = T_in / T_std
    delta = P_in / P_std
    N_cp = omega_cp * 60 / (2 * np.pi)             #  (rad/s >> r/min)


    N_cr = N_cp / np.sqrt(theta)

    U_cp = omega_cp*d_cp/2                #np.pi * N_cr * d_cp / 60    # Blade tip speed

    M = U_cp / np.sqrt(gamma_air * R_air * T_in)                           # 부록 26번


    Pi_max = np.dot(A_coeffs, np.array([M**4, M**3, M**2, M, 1]))
    Beta = np.dot(B_coeffs, np.array([M**2, M, 1]))
    Psi_max = 14 * np.dot(C_coeffs, np.array([M**5, M**4, M**3, M**2, M, 1]))


    Psi = (Cp_air * T_in / (0.5*(U_cp**2))) * (((P_out / P_in)**((gamma_air - 1) / gamma_air)) - 1)   # Appendix Eq 22
    if Psi > Psi_max :
      Psi = Psi_max
    Pi= Pi_max * (1 - np.exp(Beta*(Psi / Psi_max - 1)))    # normalized mass flow rate  Eq, 24  for Nondimensional Number

    W_cr = rho_air_std * np.pi/4 * (d_cp**2) * U_cp * Pi      # Appendix Eq.27

    W_cp = W_cr * delta / np.sqrt(theta)

    T_out = T_in + T_in / eta_cp * (((P_out / P_in)**((gamma_air - 1) / gamma_air)) - 1)



    print("Psi", Psi,"Pi",Pi, "Psi_max",Psi_max,"omega",omega_cp,"U_cp", U_cp, "W_cp",W_cp,"Pi_max","M",M)
    return W_cp, T_out              #T_out







# --- BLOCK 2 Dynamic model (Main Eq. 21) ---
def compressor_ode(t, y, V_mm, T_in, P_in, P_out):

    omega_cp = y[0]
    tau_mm = eta_mm * (k_t / R_mm) * (V_mm - (k_v * omega_cp))
    W_cp=get_static_performance(omega_cp, T_in, P_in, P_out)[0]
    tau_cp = (Cp_air*T_in) / (omega_cp * eta_cp)*((P_out/P_in)**((gamma_air-1)/gamma_air)-1)* W_cp
    d_omega_dt = (tau_mm - tau_cp) / J_cp
    print("tau_mm",tau_mm,"tau_cp",tau_cp)
    return [d_omega_dt]



def get_voltage_profile(t):        # Voltage Profile (t)

    if t < 0.3: Voltage = V_mm_base * 0.9
    elif t < 0.5: Voltage = V_mm_base * 0
    elif t < 0.7: Voltage = V_mm_base * 1.3
    else: Voltage = V_mm_base * 0
    return Voltage

t_span = [0, 1]
t_eval = np.linspace(t_span[0], t_span[1], 900)
V_profile = np.array([get_voltage_profile(t) for t in t_eval])   #   W_cp_res = pd.read_csv('file path.csv')

V_interp1d = interp1d(t_eval, V_profile, kind='zero')

initial_omega = [1000]

# --- differentail solver ---
solution = solve_ivp(
    fun=lambda t, y: compressor_ode(t,y, V_interp1d(t) , T_in, P_in, P_out),
    t_span=t_span,
    y0=initial_omega,
    t_eval=t_eval,
    method='BDF'
)

# --- result % simulation ---
t_res = solution.t
omega_res = solution.y[0]
W_cp_res = np.array([get_static_performance(w, T_in, P_in, P_out)[0] for w in omega_res])
T_out_res = np.array([get_static_performance(w, T_in, P_in, P_out)[1] for w in omega_res])
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('PSA compressor 1', fontsize=16)

ax1.plot(t_res, V_profile, 'r-', label='(V)')
ax1.set_ylabel('Driving Voltage (V)')
ax1.set_title('Input: Time vs Voltage')
ax1.grid(True)
ax1.legend()

ax2_twin = ax2.twinx()
p1, = ax2.plot(t_res, omega_res, 'b-', label='omega (rad/s)')
p2, = ax2_twin.plot(t_res, W_cp_res, 'g--', label='inlet massflow rate (kg/s)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Compressor Speed (rad/s)', color='b')
ax2_twin.set_ylabel('Outlet Mass Flow (kg/s)', color='g')
ax2.set_title('Output:compressor speed vs Mass flow rate')
ax2.grid(True)
ax2.legend(handles=[p1, p2], loc='best')

ax3.plot(t_res, T_out_res , 'c-', label='T_out (K))')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Outlet Temperature (K))')
ax3.set_title('Output: Outlet Temperature')
ax3.grid(True)
ax3.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()