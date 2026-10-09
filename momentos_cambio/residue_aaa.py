"""
Reproduce los resultados numericos de la seccion de tesis sobre el sistema
autoconsistente con momentos de segundo orden (E=1, m=0, caida marginal).

Genera, en este orden, los datos de cada cuadro:
  Cuadro 1  barrido en mom_max            -> lambda_c converge, r_c no
  Cuadro 2  barrido en rtol               -> lambda_c estable, r_c cambia
  Cuadro 3  AAA con ventana automatica sobre p_r(r)
  Cuadro 4  AAA con ventana automatica sobre G11(r), G02(r), G20(r)
  Cuadro 5  chequeo directo (r-r_p)*p_r sobre datos crudos

No se usa en ningun paso la posicion conocida del horizonte (r=2) para
localizar el polo; rs solo define el sistema.

Requiere: numpy, scipy>=1.15, sympy. Unidades geometrizadas c=G=M=1.
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import AAA

RS = 2.0            # horizonte; SOLO define el sistema, nunca localiza el polo
E_val, m_val = 1.0, 0.0


# =====================================================================
# 1. SISTEMA (momentos G de 2do orden, ecuaciones con signo invertido)
# =====================================================================
r, p_r = sp.symbols('r p_r', real=True)
G20, G11, G02 = sp.symbols('G20 G11 G02', real=True)
c, G, M, E = sp.symbols('c G M E', positive=True)

f = 1 - 2*G*M/(c**2 * r)
f1 = sp.diff(f, r); f2 = sp.diff(f, r, 2); f3 = sp.diff(f, r, 3)
W = E**2 / (f * c**2)
W1 = sp.diff(W, r); W2 = sp.diff(W, r, 2); W3 = sp.diff(W, r, 3)

Rat = sp.Rational
r_dot  = p_r*f + f1*G11 + Rat(1, 2)*p_r*G20*f2
pr_dot = (Rat(1, 2)*W1 - Rat(1, 2)*f1*p_r**2 - Rat(1, 2)*f1*G02
          - f2*p_r*G11 + Rat(1, 4)*W3*G20 - Rat(1, 4)*f3*p_r**2*G20)
G11_dot = -(-f*G02 - Rat(1, 2)*W2*G20 + Rat(1, 2)*f2*p_r**2*G20)
G02_dot = -(2*p_r*f1*G02 - W2*G11 + f2*p_r**2*G11)
G20_dot = -(-2*f*G11 - 2*p_r*f1*G20)

subs = {c: 1.0, G: 1.0, M: 1.0, E: E_val}
state = (r, p_r, G20, G11, G02)
rhs = [sp.lambdify(state, e.subs(subs), 'numpy')
       for e in (r_dot, pr_dot, G20_dot, G11_dot, G02_dot)]

def sistema(lam, y):
    return [fn(*y) for fn in rhs]

def pr_onshell(r0):
    f0 = 1 - RS/r0
    arg = (E_val**2 - m_val**2 * f0) / f0**2
    return -np.sqrt(arg)

G0 = 1e-2
y0 = [10.0, pr_onshell(10.0), G0, 0.0, G0]

def horizonte(lam, y):
    return y[0] - 1.00001*RS
horizonte.terminal = True
horizonte.direction = -1

def integrar(mom_max, rtol=1e-13, atol=1e-17, max_step=1e-3):
    def validez(lam, y):
        return mom_max - max(abs(y[2]), abs(y[3]), abs(y[4]))
    validez.terminal = True
    validez.direction = -1
    return solve_ivp(sistema, (0, 80), y0, events=[horizonte, validez],
                     method='DOP853', rtol=rtol, atol=atol, max_step=max_step)

print(f"p_r inicial = {y0[1]:.4f}")


# =====================================================================
# CUADRO 1 -- barrido en mom_max
# =====================================================================
print("\n=== Cuadro 1: barrido en mom_max (rtol=1e-13) ===")
print(f"{'mom_max':>10}{'lambda_c':>18}{'r_c':>12}{'motivo':>10}")
for mm in [1e6, 1e10, 1e14, 1e20, 1e30]:
    s = integrar(mm)
    motivo = "corte" if s.status == 1 else "float64"
    print(f"{mm:10.0e}{s.t[-1]:18.10f}{s.y[0][-1]:12.6f}{motivo:>10}")


# =====================================================================
# CUADRO 2 -- barrido en rtol (mom_max = 1e30)
# =====================================================================
print("\n=== Cuadro 2: barrido en rtol (mom_max=1e30) ===")
print(f"{'rtol':>10}{'lambda_c':>18}{'r_c':>12}")
for rt in [1e-10, 1e-12, 1e-13]:
    s = integrar(1e30, rtol=rt)
    print(f"{rt:10.0e}{s.t[-1]:18.10f}{s.y[0][-1]:12.6f}")


# =====================================================================
# CORRIDA DE REFERENCIA (alta resolucion) para los cuadros 3, 4, 5
# =====================================================================
sol = integrar(1e30, rtol=3e-14, atol=1e-18, max_step=5e-4)
r_s, pr_s, G20_s, G11_s, G02_s = sol.y
lam_s = sol.t
print(f"\nCorrida de referencia: {len(lam_s)} puntos, "
      f"lambda_c={lam_s[-1]:.12f}, r_min={r_s.min():.6f}")

# ordenar por r creciente, limpiar no-finitos y r duplicados
def limpiar(r, *series):
    m = np.isfinite(r)
    for s in series:
        m &= np.isfinite(s)
    r = r[m]; series = [s[m] for s in series]
    idx = np.argsort(r); r = r[idx]; series = [s[idx] for s in series]
    r, ui = np.unique(r, return_index=True); series = [s[ui] for s in series]
    return (r, *series)

rr, P, A20, A11, A02 = limpiar(r_s, pr_s, G20_s, G11_s, G02_s)


# =====================================================================
# polo dominante: casi real (|Im|<1e-4), de mayor |residuo|
# =====================================================================
def polo_dominante(fit, tol_imag=1e-4):
    pol, res = fit.poles(), fit.residues()
    casi = np.abs(pol.imag) < tol_imag
    if not casi.any():
        return None
    i = np.argmax(np.abs(res[casi]))
    return pol[casi][i].real, res[casi][i].real


# =====================================================================
# CUADRO 3 -- AAA con ventana automatica sobre p_r(r)
# =====================================================================
# ventana = los N puntos con menor r (mas cercanos al punto de paro),
# sin usar RS. r_max es el mayor radio incluido.
print("\n=== Cuadro 3: AAA sobre p_r(r) ===")
print(f"{'N':>6}{'r_max':>12}{'m':>5}{'r_p':>14}{'Res':>16}")
for N in [100, 200, 300, 500, 700, 1000, 1500, 2000]:
    fit = AAA(rr[:N], P[:N], rtol=1e-10)
    dom = polo_dominante(fit)
    if dom is None:
        print(f"{N:6d}{rr[N-1]:12.5f}{fit.errors.size:5d}   sin polo casi real")
    else:
        print(f"{N:6d}{rr[N-1]:12.5f}{fit.errors.size:5d}{dom[0]:14.6f}{dom[1]:16.4e}")


# =====================================================================
# CUADRO 4 -- AAA sobre los momentos G
# =====================================================================
print("\n=== Cuadro 4: AAA sobre G11(r), G02(r), G20(r) ===")
for nombre, serie in [('G11', A11), ('G02', A02), ('G20', A20)]:
    print(f"-- {nombre} --")
    for N in [100, 300, 500, 700, 1000, 1500]:
        fit = AAA(rr[:N], serie[:N], rtol=1e-10)
        dom = polo_dominante(fit)
        if dom is None:
            print(f"   N={N:5d}   sin polo casi real")
        else:
            print(f"   N={N:5d}   r_p={dom[0]:12.6f}   Res={dom[1]:.4e}")


# =====================================================================
# CUADRO 5 -- chequeo directo (r-r_p)*p_r sobre datos crudos
# =====================================================================
# r_p = 2 (horizonte), 2.0948 (polo de p_r segun AAA, N=1000),
#       2.1005 (polo de G02 segun AAA, N=1000).
print("\n=== Cuadro 5: (r-r_p)*p_r sobre datos crudos ===")
candidatos = [2.0, 2.0948, 2.1005]
# elegir ~6 radios repartidos entre el rango medio y el punto de paro
radios_idx = [np.argmin(np.abs(rr - rv)) for rv in
              [2.904, 2.417, 2.1435, 2.1175, 2.1094, rr[0]]]
print(f"{'r':>12}" + "".join(f"{f'r_p={c}':>16}" for c in candidatos))
for i in radios_idx:
    fila = "".join(f"{(rr[i]-c)*P[i]:16.4e}" for c in candidatos)
    print(f"{rr[i]:12.6f}{fila}")

print("\n(en el caso clasico este producto es constante en -1.99999997;")
print(" aqui crece sin estabilizarse para cualquier candidato de r_p)")