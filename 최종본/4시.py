import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Model & Map parameters
inertia_compressor_motor = 5e-5 # kg * m2
back_electromtive_force_constant = 0.0225 # V/(rad/s)
Torque_constant = 0.0153 # Nm/A
motor_resistance = 1.2 # ohm
motor_efficiency = 0.985 
compressor_efficiency = 0.8

air_constant = 286.9 # J/(kg*K)
deinsity_air = 1.23 # kg/m3
compressor_diameter = 0.2286 # m
adiabatic_constant = 1.4
supply_pressure =250000 
out_pressure = 270000 # pa

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

specific_heat_air = adiabatic_constant * air_constant / (adiabatic_constant - 1)
# almost 1004J/(mol*K)

# Initial conditions
atmospheric_temperature = 298.15 # K
atmospheric_pressure = 101325 # pa
initial_compressor_speed = 8400  # 초기값 8400 0에서 시작해도 큰 상관 없는데 결과값 자세히 보기위함

out_compressor_temperature = (atmospheric_temperature 
                              + (atmospheric_temperature / compressor_efficiency) 
                              * (((out_pressure / atmospheric_pressure)
                                  **((adiabatic_constant - 1) / adiabatic_constant)) - 1))
# Voltage 대신 voltage_profile 함수 사용

# Voltage profile (2.5초마다 변화)
def voltage_profile(t):
    if t < 0.5:
        return 160
    elif t < 1.0:
        return 170
    elif t < 1.5:
        return 150
    else:
        return 165

# --- Map functions ---
def psi_value(U_cp):
    return ((specific_heat_air * atmospheric_temperature / (0.5*(U_cp**2))) 
            * (((out_pressure / atmospheric_pressure)**((adiabatic_constant - 1) 
                                                        / adiabatic_constant)) - 1))

def mach_number(U_cp):
    return U_cp / np.sqrt(adiabatic_constant * air_constant * atmospheric_temperature)

def phi_max(M):
    return a4*M**4 + a3*M**3 + a2*M**2 + a1*M + a0

def beta_poly(M):
    return b2*M**2 + b1*M + b0

def psi_max(M):
    return c5*M**5 + c4*M**4 + c3*M**3 + c2*M**2 + c1*M + c0

def air_density():
    return out_pressure / (air_constant * out_compressor_temperature)

def phi_value(U_cp):
    M = mach_number(U_cp)
    phi_m = phi_max(M)
    beta_m = beta_poly(M)
    psi_m = psi_max(M)
    psi = psi_value(U_cp)
    psi_e = min(psi, psi_m)
    exponent = beta_m * (psi_e / psi_m - 1)
    return phi_m * (1 - np.exp(exponent))

# Compressor torque
def compressor_torque(omega, V):
    U_cp = omega * compressor_diameter / 2
    phi = phi_value(U_cp)
    airdensity = air_density()
    Wcp = (np.pi * compressor_diameter**2 * airdensity * U_cp * phi) / 4
    term = ((supply_pressure / atmospheric_pressure) ** ((adiabatic_constant - 1) / adiabatic_constant)) - 1
    tau_cp = specific_heat_air / omega * atmospheric_temperature / compressor_efficiency * term * Wcp
    return tau_cp, Wcp

# Motor torque
def motor_torque(omega, V):
    return (motor_efficiency * back_electromtive_force_constant / motor_resistance 
            * (V - Torque_constant * omega))

# Differential equation (전압 profile 반영)
def d_omega_dt(t, omega):
    V = voltage_profile(t)
    tau_mm = motor_torque(omega[0], V)
    tau_cp, _ = compressor_torque(omega[0], V)
    return [(tau_mm - tau_cp) / inertia_compressor_motor]

# 시간 범위 설정
t_span = (0, 2)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# 미분방정식 풀기
sol = solve_ivp(d_omega_dt, t_span, [initial_compressor_speed],method= "LSODA", 
                t_eval=t_eval, vectorized=False)

# Wcp 값 계산
Wcp_list = []
for t, omega in zip(sol.t, sol.y[0]):
    _, Wcp = compressor_torque(omega, voltage_profile(t))
    Wcp_list.append(Wcp)
Wcp_array = np.array(Wcp_list)

# --- 결과 시각화 ---
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(sol.t, sol.y[0], 'b-', label='Angular Velocity (rad/s)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular Velocity (rad/s)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(sol.t, Wcp_array, 'g-', label='Wcp (kg/s)')
ax2.set_ylabel('Wcp (kg/s)', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.legend(loc='upper right')
ax2.set_ylim(bottom=0.00)


plt.title('Compressor Angular Velocity and Wcp over Time (with Step Voltage)')
plt.grid(True)
plt.tight_layout()
plt.show()