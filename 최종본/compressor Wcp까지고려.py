import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Model & Map parameters ---
Jcp = 5e-5
kt = 0.0225
kv = 0.0153
Rmm = 1.2
eta_mm = 0.985
eta_cp = 0.8

R_air = 286.9
d_cp = 0.2286
gamma = 1.4

# 맵 계수
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
omega0 = 8400

# ====== 2단 직렬 세팅 ======
Pin1  = Patm                 # 1단 입구압                # <<< MOD
Pout1 = 270000               # 1단 출구압 = 2.7 bar      # <<< MOD
Pin2  = Pout1                # 2단 입구압 = 1단 출구압    # <<< MOD
Pout2 = 750000               # 2단 출구압 = 7.5 bar      # <<< MOD

# --- Voltage profiles (스테이지별) ---
def voltage_profile1(t):     # 1단 전압 프로파일 (네가 준 스텝 그대로)           # <<< MOD
    if t < 0.5: return 160
    elif t < 1.0: return 170
    elif t < 1.5: return 150
    else: return 165

def voltage_profile2(t):     # 2단 전압 프로파일(원하면 1단과 다르게 튜닝 가능)  # <<< MOD
    if t < 0.5: return 165
    elif t < 1.0: return 175
    elif t < 1.5: return 155
    else: return 170

# ===== 스테이지-파라미터화 유틸 =====
def mach_number(U_cp):
    return U_cp / np.sqrt(gamma * R_air * Tatm)

def phi_max(M):   return a4*M**4 + a3*M**3 + a2*M**2 + a1*M + a0
def beta_poly(M): return b2*M**2 + b1*M + b0
def psi_max(M):   return c5*M**5 + c4*M**4 + c3*M**3 + c2*M**2 + c1*M + c0

def T_out_stage(Pin, Pout):                                        # <<< MOD
    # 등엔트로피 압축 + 효율 반영(간단 모델)
    return Tatm + (Tatm / eta_cp) * (((Pout / Pin)**((gamma - 1) / gamma)) - 1)

def air_density_out_stage(Pin, Pout):                               # <<< MOD
    Tout = T_out_stage(Pin, Pout)
    return Pout / (R_air * Tout)

def psi_value_stage(U_cp, Pin, Pout):                               # <<< MOD
    # 스테이지별 압력비 사용
    return (Cp_air * Tatm / (0.5*(U_cp**2))) * (((Pout / Pin)**((gamma - 1) / gamma)) - 1)

def phi_value_stage(U_cp, Pin, Pout):                               # <<< MOD
    M = mach_number(U_cp)
    phi_m = phi_max(M)
    beta_m = beta_poly(M)
    psi_m = psi_max(M)
    psi = psi_value_stage(U_cp, Pin, Pout)
    psi_e = min(psi, psi_m)
    exponent = beta_m * (psi_e / psi_m - 1)
    return phi_m * (1 - np.exp(exponent))

def compressor_torque_stage(omega, V, Pin, Pout):                   # <<< MOD
    U_cp = omega * d_cp / 2
    phi = phi_value_stage(U_cp, Pin, Pout)
    airdensity = air_density_out_stage(Pin, Pout)
    Wcp = (np.pi * d_cp**2 * airdensity * U_cp * phi) / 4  # kg/s
    term = ((Pout / Pin) ** ((gamma - 1) / gamma)) - 1
    omega_eff = max(omega, 1e-6)
    tau_cp = Cp_air / omega_eff * Tatm / eta_cp * term * Wcp
    return tau_cp, Wcp

def motor_torque(omega, V):
    return eta_mm * kt / Rmm * (V - kv * omega)

def d_omega_dt_stage(t, omega, Pin, Pout, vfunc):                   # <<< MOD
    V = vfunc(t)
    tau_mm = motor_torque(omega[0], V)
    tau_cp, _ = compressor_torque_stage(omega[0], V, Pin, Pout)
    return [(tau_mm - tau_cp) / Jcp]

# --- 시간 범위 설정 ---
t_span = (0, 2)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# ===== 1단 해석 =====
sol1 = solve_ivp(lambda t, omg: d_omega_dt_stage(t, omg, Pin1, Pout1, voltage_profile1),  # <<< MOD
                 t_span, [omega0], method="LSODA", t_eval=t_eval, vectorized=False)

Wcp1 = []
for t, omega in zip(sol1.t, sol1.y[0]):
    _, w = compressor_torque_stage(omega, voltage_profile1(t), Pin1, Pout1)               # <<< MOD
    Wcp1.append(w)
Wcp1 = np.array(Wcp1)


# ===== 2단 해석 =====
# 2단도 동일 시간축 사용(필요시 별도 설정 가능)
sol2 = solve_ivp(lambda t, omg: d_omega_dt_stage(t, omg, Pin2, Pout2, voltage_profile2),  # <<< MOD
                 t_span, [omega0], method="LSODA", t_eval=t_eval, vectorized=False)

Wcp2 = []
for t, omega in zip(sol2.t, sol2.y[0]):
    _, w = compressor_torque_stage(omega, voltage_profile2(t), Pin2, Pout2)               # <<< MOD
    Wcp2.append(w)
Wcp2 = np.array(Wcp2)

# --- 결과 시각화 ---
fig, (axw, axm) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 각 스테이지 회전속도
axw.plot(sol1.t, sol1.y[0], 'b-', label='Stage-1 ω (rad/s)')
axw.plot(sol2.t, sol2.y[0], 'c--', label='Stage-2 ω (rad/s)')
axw.set_ylabel('Angular Velocity (rad/s)')
axw.legend(loc='upper left')
axw.grid(True)

# 각 스테이지 질량유량
axm.plot(sol1.t, Wcp1, 'g-', label='Stage-1 Wcp (kg/s)')
axm.plot(sol2.t, Wcp2, 'm--', label='Stage-2 Wcp (kg/s)')
axm.set_xlabel('Time (s)')
axm.set_ylabel('Wcp (kg/s)')
axm.legend(loc='upper right')
axm.grid(True)

plt.suptitle('Two-Stage Compressor (Pin1=1.013 bar → Pout1=2.7 bar → Pout2=7.5 bar)')
plt.tight_layout()
plt.show()