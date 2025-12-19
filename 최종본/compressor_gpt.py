import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Model & Map parameters (원본 그대로) ---
Jcp = 5e-5
kt = 0.0225
kv = 0.0153
Rmm = 1.2
eta_mm = 0.985
eta_cp = 0.8

R_air = 286.9
d_cp = 0.2286
gamma = 1.4

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

# --- Initial & boundary conditions ---
Tatm = 298.15
Patm = 101325
omega0 = 8400

#!!!!! 직렬 시스템 파라미터
Pin1   = Patm          # 1단 입구: 대기
Pout2  = 750000        # 2단 최종 배압(다운스트림에 의해 고정된다고 가정)
P12_0  = 270000        # 초기 인터스테이지 압력(초기화만)
V12    = 1.0e-3        # 인터스테이지 용적 [m^3] (튜닝 파라미터)
Pmin   = 1.05*Patm     # 수치 안정화용 하한
Pmax   = 0.995*Pout2   # 수치 안정화용 상한

# --- Voltage profiles (원본 유지 가능, 필요시 조정) ---
def voltage_profile1(t):
    if t < 0.5: return 160
    elif t < 1.0: return 170
    elif t < 1.5: return 150
    else: return 165

def voltage_profile2(t):
    if t < 0.5: return 165
    elif t < 1.0: return 175
    elif t < 1.5: return 155
    else: return 170

# ===== 맵 유틸 =====
def mach_number(U_cp):
    return U_cp / np.sqrt(gamma * R_air * Tatm)

def phi_max(M):   return a4*M**4 + a3*M**3 + a2*M**2 + a1*M + a0
def beta_poly(M): return b2*M**2 + b1*M + b0
def psi_max(M):   return c5*M**5 + c4*M**4 + c3*M**3 + c2*M**2 + c1*M + c0

def T_out_stage(Pin, Pout):
    return Tatm + (Tatm / eta_cp) * (((Pout / Pin)**((gamma - 1) / gamma)) - 1)

def air_density_out_stage(Pin, Pout):
    Tout = T_out_stage(Pin, Pout)
    return Pout / (R_air * Tout)

def psi_value_stage(U_cp, Pin, Pout):
    return (Cp_air * Tatm / (0.5*(U_cp**2))) * (((Pout / Pin)**((gamma - 1) / gamma)) - 1)

def phi_value_stage(U_cp, Pin, Pout):
    M = mach_number(U_cp)
    phi_m = phi_max(M)
    beta_m = beta_poly(M)
    psi_m = psi_max(M)
    psi = psi_value_stage(U_cp, Pin, Pout)
    psi_e = min(psi, psi_m)
    exponent = beta_m * (psi_e / psi_m - 1)
    return phi_m * (1 - np.exp(exponent))

def compressor_torque_and_flow(omega, Pin, Pout):
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

#!!!!! 연속(직렬) 결합 ODE: x = [omega1, omega2, P12]
def rhs(t, x):
    omega1, omega2, P12 = x
    # 수치 안정화된 인터스테이지 압력
    P12_eff = np.clip(P12, Pmin, Pmax)

    # 토크 & 유량 (동일 맵, 각 스테이지에 현재 Pin/Pout 적용)
    V1 = voltage_profile1(t)
    V2 = voltage_profile2(t)

    tau_mm1 = motor_torque(omega1, V1)
    tau_cp1, W1 = compressor_torque_and_flow(omega1, Pin1, P12_eff)

    tau_mm2 = motor_torque(omega2, V2)
    tau_cp2, W2 = compressor_torque_and_flow(omega2, P12_eff, Pout2)

    domega1 = (tau_mm1 - tau_cp1) / Jcp
    domega2 = (tau_mm2 - tau_cp2) / Jcp

    #!!!!! 인터스테이지 압력 동역학 (이상기체 + 준정온 가정)
    # plenum 온도는 1단 출구 온도로 근사(다른 가정 가능: 고정값/등온/1차 완화 등)
    T12 = T_out_stage(Pin1, P12_eff)
    dP12 = (R_air * T12 / V12) * (W1 - W2)

    return [domega1, domega2, dP12]

# --- 시뮬레이션 ---
t_span = (0, 2)
t_eval = np.linspace(*t_span, 1000)

x0 = [omega0, omega0, P12_0]   # 초기 회전속도 동일, 인터스테이지 압력은 270 kPa
sol = solve_ivp(rhs, t_span, x0, method="LSODA", t_eval=t_eval, vectorized=False)

omega1_traj, omega2_traj, P12_traj = sol.y

# 포스트: 시간축에서 유량 재계산(플롯용)
W1_hist, W2_hist = [], []
for t, P12 in zip(sol.t, P12_traj):
    P12_eff = np.clip(P12, Pmin, Pmax)
    _, W1 = compressor_torque_and_flow(omega1_traj[np.searchsorted(sol.t, t)], Pin1, P12_eff)
    _, W2 = compressor_torque_and_flow(omega2_traj[np.searchsorted(sol.t, t)], P12_eff, Pout2)
    W1_hist.append(W1); W2_hist.append(W2)
W1_hist = np.array(W1_hist); W2_hist = np.array(W2_hist)

# --- 결과 시각화 ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].plot(sol.t, omega1_traj, label='Stage-1 ω (rad/s)')
axes[0].plot(sol.t, omega2_traj, '--', label='Stage-2 ω (rad/s)')
axes[0].set_ylabel('Angular Velocity')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(sol.t, P12_traj/1e5)
axes[1].set_ylabel('Interstage Pressure (bar)')
axes[1].grid(True)

axes[2].plot(sol.t, W1_hist, label='W1 (1→12)')
axes[2].plot(sol.t, W2_hist, '--', label='W2 (12→out)')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Mass Flow (kg/s)')
axes[2].legend(); axes[2].grid(True)

plt.suptitle('Two-Stage Compressor: Fully Coupled Dynamics via Interstage Plenum')
plt.tight_layout()
plt.show()