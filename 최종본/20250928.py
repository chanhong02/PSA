import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#!!!!!! 루트파인딩 추가
from scipy.optimize import brentq

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

#!!!!!! 직렬 시스템 (부피 없음 → P12는 즉시 해로 결정)
Pin1  = Patm
Pout2 = 750000

# --- Voltage profiles ---
def voltage_profile1(t):
    if t < 2: return 206
    return 206  #!!!!!! (항상 값 반환)

def voltage_profile2(t):
    if t < 2: return 206
    return 206  #!!!!!! (항상 값 반환)

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
    M = max(mach_number(U_cp), 1e-6)                         #!!!!!!
    phi_m = phi_max(M)
    beta_m = beta_poly(M)
    psi_m = psi_max(M)
    psi = psi_value_stage(U_cp, Pin, Pout)
    psi_e = min(psi, psi_m)
    exponent = beta_m * (psi_e / max(psi_m, 1e-9) - 1)       #!!!!!!
    return max(phi_m * (1 - np.exp(exponent)), 0.0)          #!!!!!!

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

#!!!!!! 부피=0 연속조건: W1(ω1,Pin1,P12) - W2(ω2,P12,Pout2) = 0 만족하는 P12 계산
def solve_P12(omega1, omega2):
    Pmin = 1.02*Pin1           # 흡입보다 살짝 높은 하한
    Pmax = 0.98*Pout2          # 최종압보다 살짝 낮은 상한

    def g(P12):
        _, W1 = compressor_torque_and_flow(omega1, Pin1, P12)
        _, W2 = compressor_torque_and_flow(omega2, P12, Pout2)
        return W1 - W2

    try:
        return brentq(g, Pmin, Pmax, maxiter=50)
    except ValueError:
        # 부호 변화가 없을 때 스윕으로 근사
        Ps = np.linspace(Pmin, Pmax, 60)
        vals = np.abs([g(p) for p in Ps])
        return Ps[int(np.argmin(vals))]

#!!!!!! 상태공간 ODE (상태 = [ω1, ω2])
def rhs(t, y):
    omega1, omega2 = y

    # 즉시 연속으로 P12(t) 도출
    P12 = solve_P12(omega1, omega2)

    # Stage-1 dynamics
    tau_mm1 = motor_torque(omega1, voltage_profile1(t))
    tau_cp1, _ = compressor_torque_and_flow(omega1, Pin1, P12)
    domega1 = (tau_mm1 - tau_cp1) / Jcp

    # Stage-2 dynamics
    tau_mm2 = motor_torque(omega2, voltage_profile2(t))
    tau_cp2, _ = compressor_torque_and_flow(omega2, P12, Pout2)
    domega2 = (tau_mm2 - tau_cp2) / Jcp

    return [domega1, domega2]

# --- 시뮬레이션 ---
t_span = (0, 2)
t_eval = np.linspace(*t_span, 1000)

#!!!!!! 두 스테이지 동시 적분 (부피 없음)
y0 = [omega0, omega0]
sol = solve_ivp(rhs, t_span, y0, method="LSODA", t_eval=t_eval, vectorized=False)

#!!!!!! 포스트: P12, 유량 히스토리 계산
omega1_traj, omega2_traj = sol.y
P12_hist, W1_hist, W2_hist = [], [], []
for w1, w2 in zip(omega1_traj, omega2_traj):
    P12 = solve_P12(w1, w2)
    P12_hist.append(P12)
    _, W1 = compressor_torque_and_flow(w1, Pin1, P12)
    _, W2 = compressor_torque_and_flow(w2, P12, Pout2)
    W1_hist.append(W1); W2_hist.append(W2)

P12_hist = np.array(P12_hist)
W1_hist  = np.array(W1_hist)
W2_hist  = np.array(W2_hist)

# --- 결과 시각화 ---
fig, (axw, axp, axm) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axw.plot(sol.t, omega1_traj, label='Stage-1 ω (rad/s)')
axw.plot(sol.t, omega2_traj, '--', label='Stage-2 ω (rad/s)')
axw.set_ylabel('Angular Velocity'); axw.legend(); axw.grid(True)

axp.plot(sol.t, P12_hist/1e5)
axp.set_ylabel('P12 (bar)'); axp.grid(True)

axm.plot(sol.t, W1_hist, label='W1: 1→12')
axm.plot(sol.t, W2_hist, '--', label='W2: 12→out')
axm.set_xlabel('Time (s)'); axm.set_ylabel('Mass Flow (kg/s)')
axm.legend(); axm.grid(True)

plt.suptitle('Two-Stage Compressor (Zero-Volume Coupling: Instantaneous W1=W2)')
plt.tight_layout()
plt.show()