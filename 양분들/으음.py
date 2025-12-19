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
Vmm = 300  # Initial voltage value (it will change)
omega0 = 100

# --- Map functions ---
def psi_value(U_cp):
    return (Cp_air * Tatm / (0.5*(U_cp**2))) * (((Pout / Patm)**((gamma - 1) / gamma)) - 1)

def mach_number(U_cp):
    return U_cp / np.sqrt(gamma * R_air * Tatm)

def phi_max(M):
    return a4*M**4 + a3*M**3 + a2*M**2 + a1*M + a0

def beta_poly(M):
    return b2*M**2 + b1*M + b0

def psi_max(M):
    return c5*M**5 + c4*M**4 + c3*M**3 + c2*M**2 + c1*M + c0

def phi_value(U_cp):
    M = mach_number(U_cp)
    phi_m = phi_max(M)
    beta_m = beta_poly(M)
    psi_m = psi_max(M)
    psi = psi_value(U_cp)

    psi_e = min(psi, psi_m)
    exponent = beta_m * (psi_e / psi_m - 1)

    return phi_m * (1 - np.exp(exponent))

# --- Compressor torque ---
def compressor_torque(omega):
    U_cp = omega * d_cp / 2
    phi = phi_value(U_cp)
    Wcp = (np.pi * d_cp**2 * rho_air * U_cp * phi) / 4
    term = ((Psm / Patm) ** ((gamma - 1) / gamma)) - 1
    tau_cp = Cp_air / omega * Tatm / eta_cp * term * Wcp
    return tau_cp

# --- Motor torque ---
def motor_torque(omega):
    tau_mm = eta_mm * kt / Rmm * (Vmm - kv * omega)
    return tau_mm

# Differential equation (기존 방정식을 사용하여 코드 구현)
def d_omega_dt(t, omega):
    tau_mm = motor_torque(omega[0])  # 모터 토크
    tau_cp = compressor_torque(omega[0])  # 압축기 토크
    return [(tau_mm - tau_cp) / Jcp]  # dω/dt

# --- 시간 범위 설정 --- (100초로 설정)
t_span = (0, 100)  # 100초까지 시뮬레이션
t_eval = np.linspace(t_span[0], t_span[1], 500)  # 평가할 시간 값 생성 (100초 동안)

# --- solve_ivp로 미분방정식 풀기 ---
sol = solve_ivp(d_omega_dt, t_span, [omega0], t_eval=t_eval, vectorized=True)

# --- Wcp 값을 구하고 플로팅 ---
Wcp_values = np.array([compressor_torque(omega) for omega in sol.y[0]])

# --- 결과 시각화 ---
fig, ax1 = plt.subplots(figsize=(8, 6))

# 각속도 그래프
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular Velocity (rad/s)', color='b')
ax1.plot(sol.t, sol.y[0], label=r'$\omega(t)$', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Wcp 그래프
ax2 = ax1.twinx()  # 두 번째 y축을 공유하는 그래프
ax2.set_ylabel('Compressor Power Output $W_{cp}$ (kg/s)', color='g')
ax2.plot(sol.t, Wcp_values, label=r'$W_{cp}(t)$', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Wcp y-axis 범위 조정 (자동 설정에서 범위 확대)
ax2.set_ylim(0, max(Wcp_values) * 1.2)  # Wcp 값이 더 잘 보이도록 범위 설정

fig.tight_layout()
plt.title('Compressor Angular Velocity and Power Output Over Time with Voltage Changes (100 seconds)')
plt.grid(True)
plt.show()