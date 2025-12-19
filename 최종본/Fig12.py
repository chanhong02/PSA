import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- (파라미터/행렬 등 기존코드 생략) ---
# 1. 시스템 파라미터 및 행렬/초기값
Rg = 8.314
T = 300.15
eps = 0.32
DL = 9e-7
KA = 9.35
kA = 0.0832
KB = 7.35
kB = 0.00468
PH = 750000    # Pa, High pressure (예시: 750kPa)
PL = 101325    # Pa, Low pressure (예시: 1 atm)
tfp = 40      # sec, pressure swing duration
column_length = 2.0  # m
a = 2.0

N = 11
delta_z = column_length / (N - 1)

Matrix_A = np.array([
    [-9.1602, 14.96, -10.336, 8.3222, -7.0296, 6.0402, -5.1923, 4.3967, -3.5791, 2.632, -1.0543],
    [-2.9225, -4.0248, 10.913, -6.9352, 5.3886, -4.4499, 3.7429, -3.1292, 2.528, -1.8512, 0.74031],
    [0.88657, -4.7918, -2.6072, 9.9843, -5.9667, 4.4664, -3.5791, 2.9127, -2.3168, 1.6823, -0.67052],
    [-0.43165, 1.8414, -6.0373, -1.9568, 9.9317, -5.6682, 4.0959, -3.1667, 2.4498, -1.753, 0.69475],
    [0.26641, -1.0454, 2.6363, -7.257, -1.5918, 10.418, -5.7238, 3.9871, -2.933, 2.0468, -0.80363],
    [-0.19229, 0.72519, -1.6577, 3.479, -8.7513, -1.365, 11.448, -6.0685, 4.0333, -2.6903, 1.0394],
    [0.15681, -0.57866, 1.2602, -2.3849, 4.5612, -10.86, -1.2163, 13.232, -6.7274, 4.0901, -1.5331],
    [-0.14276, 0.52014, -1.1026, 1.9825, -3.4161, 6.1897, -14.227, -1.1174, 16.312, -7.6987, 2.7006],
    [0.14633, -0.52909, 1.1043, -1.9311, 3.1641, -5.1796, 9.1072, -20.538, -1.0531, 22.03, -6.3204],
    [-0.17802, 0.64096, -1.3265, 2.2859, -3.6528, 5.7158, -9.16, 16.036, -36.445, -1.0154, 27.099],
    [0, 0, 0, 0, 0, -0, 0, 0, 0, 1, -1]
])


Matrix_B = np.array([
    [-23.43, -5.8982, 61.295, -61.652, 56.372, -50.304, 44.148, -37.841, 31.029, -22.91, 9.1916],
    [70.069, -144.95, 83.106, -6.6484, -5.6013, 8.5995, -9.0083, 8.3874, -7.1837, 5.4251, -2.1954],
    [-12.372, 98.598, -177.83, 108.28, -21.426, 5.9422, -1.0519, -0.7598, 1.33, -1.2495, 0.54203],
    [4.3848, -22.477, 120.02, -211.98, 132.66, -30.953, 12.171, -5.7949, 3.0242, -1.6252, 0.56268],
    [-2.1733, 9.5551, -30.959, 148.17, -262.28, 167.64, -42.018, 18.31, -9.8658, 5.6814, -2.0566],
    [1.3362, -5.4638, 14.731, -41.745, 192.42, -345.15, 225.38, -59.019, 26.879, -14.526, 5.1562],
    [-0.96735, 3.8021, -9.3693, 21.81, -58.871, 269.78, -495.39, 331.94, -89.812, 40.772, -13.694],
    [0.80723, -3.1, 7.2688, -15.363, 33.914, -90.774, 423.24, -808.55, 561.48, -154.48, 45.56],
    [-0.96735, 2.9497, -6.7215, 13.485, -27.033, 59.36, -162.81, 796.07, -1639.6, 1202.6, -237.59],
    [0.91276, -3.431, 7.6977, -15.026, 28.721, -57.778, 132.03, -390.11, 2140.3, -5370.9, 3527.5],
    [0, 0, 0, 0, 0, -0, 0, 0, 0, 1, -1]
])  # 네 코드에서 복사

# 2. 초기조건 (yB는 필요없음)
yA0 = np.full(N, 0.21)   # 산소 몰분율 (초기)
qA0 = np.zeros(N)
qB0 = np.zeros(N)
y_init = np.concatenate([yA0, qA0, qB0])  # 상태벡터 (yA, qA, qB)

# 3. 압력 시간함수 정의
def P_time(t, PH, PL, tfp):
    return PH + (PL - PH) * ((t / tfp) - 1)**2

def differ_pressure(t):
    return 2 * (PL - PH) * (t / tfp - 1) / tfp

# 4. 경계조건 함수
# def apply_boundary_conditions(yA):
#     yA[-1] = yA[-2]
#     return yA

def vg_inlet_time(t):
    if 0 <= t < 5:
        return 0.11
    elif 5 <= t < 15:
        return 0.14
    elif 15 <= t < 30:
        return 0.08
    elif 30 <= t <= 40:
        return 0.11
    else:
        return 0.11

def psa_odes(t, y, PH, PL, tfp):
    yA = y[0:N]
    qA = y[N:2*N]
    qB = y[2*N:3*N]
    yB = 1.0 - yA

    dydt = np.zeros_like(y)
    P = P_time(t, PH, PL, tfp)
    dP_dt = differ_pressure(t)

    dqA_dt = kA * (KA * yA * P / (Rg * T) - qA)
    dqB_dt = kB * (KB * yB * P / (Rg * T) - qB)

    vg_new = np.zeros(N)
    vg_new[0] = vg_inlet_time(t)
    for j in range(N-1):
        dvg_dz = - ((1-eps)/eps) * (Rg*T/P) * (dqA_dt[j] + dqB_dt[j]) - (1/P) * dP_dt
        vg_new[j+1] = vg_new[j] + dvg_dz * delta_z

    dyA_dz = Matrix_A @ yA
    d2yA_dz2 = Matrix_B @ yA

    # yA = apply_boundary_conditions(yA)
    yA[0] = 0.21
    dydt[0] = 0

    for i in range(0, N):
        convection_A = -vg_new[i] / a * dyA_dz[i]
        diffusion_A = DL / (a**2) * d2yA_dz2[i]
        adsorption_A = ((1-eps)/eps) * (Rg*T/P) * ((yA[i]-1)*dqA_dt[i] + yA[i]*dqB_dt[i])
        pressure_A = (yA[i]/P) * dP_dt
        dydt[i] = diffusion_A + convection_A + adsorption_A + pressure_A

    for i in range(N):
        dydt[N+i] = dqA_dt[i]
        dydt[2*N+i] = dqB_dt[i]
    return dydt

# ---- solve_ivp ----
t_span = (0, 40)
t_eval = np.linspace(0, 40, 201)
sol = solve_ivp(
    lambda t,y: psa_odes(t,y,PH,PL,tfp),
    t_span, y_init,
    method='LSODA',
    t_eval=t_eval,
    rtol=1e-6, atol=1e-8
)

yA_sol = sol.y[0:N, :]
yB_sol = 1.0 - yA_sol

first = yB_sol[0]
fifth = yB_sol[4]
tenth = yB_sol[9]

# --- inlet air velocity profile for plot (논문 스타일 점선) ---
inlet_velocity = np.array([vg_inlet_time(t) for t in sol.t])

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(8,4.5))

ax1.plot(sol.t, first,  label='First point of bed', marker='s', color='#27A39B', markevery=20, lw=2, mfc='white', mec='#27A39B')
ax1.plot(sol.t, fifth,  label='Fifth point of bed', marker='^', color='#F5A623', markevery=20, lw=2, mfc='white', mec='#F5A623')
ax1.plot(sol.t, tenth,  label='Tenth point of bed', marker='*', color='#1B4FDE', markevery=20, lw=2, mfc='white', mec='#1B4FDE')

ax1.set_ylabel(r'N$_2$ purity', fontsize=13)
ax1.set_xlabel('t (s)', fontsize=13)
ax1.set_ylim(0.78, 1.01)
ax1.grid(True, linestyle=':', alpha=0.5)

# --- 논문 스타일: 속도 점선(---), y2축 ---
ax2 = ax1.twinx()
ax2.plot(sol.t, inlet_velocity, 'r--', label='Inlet air velocity of adsorption tower', lw=2)
ax2.set_ylabel('Velocity (m/s)', fontsize=13)
ax2.set_ylim(0.07, 0.15)

# --- 범례 꾸미기 (두 축 모두 legend) ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='lower right', fontsize=12)

plt.tight_layout()
plt.show()