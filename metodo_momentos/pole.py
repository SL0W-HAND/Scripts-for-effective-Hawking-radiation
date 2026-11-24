"""
=============================================================================
 POLO Y RESIDUO DE p_r(r) CON AAA — Schwarzschild SIN correcciones de momentos
=============================================================================
 Caída radial clásica en Schwarzschild, variables canónicas (r, p_r):

     dr/dλ   = p_r f(r)
     dp_r/dλ = 1/2 W'(r) - 1/2 f'(r) p_r^2 ,   W = E^2/f

 Se resuelven dos casos (tablas tab:sch_masivo y tab:sch_m0 de la tesis):
     - masivo : E = 0.97, kappa = 1
     - fotón  : E = 1,    kappa = 0     (p_r = -E r/(r - r_s), exacto)

 Procedimiento (no se usa la posición conocida del horizonte):
   1. La integración se detiene cuando |p_r| supera P_MAX (cerca del polo),
      antes de que el paso adaptativo colapse; así el error del integrador
      se mantiene pequeño.
   2. Los datos se ordenan por r y se submuestrean de forma logarítmica en
      r - r_min (los puntos se concentran cerca del polo), lo que hace que
      AAA sea rápido.
   3. Entre los polos casi reales de la aproximación racional situados justo
      por debajo del menor radio integrado se toma el de mayor |residuo|.

 Unidades geometrizadas c = G = M = 1  =>  r_s = 2.
=============================================================================
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import AAA

RS = 2.0          # define el sistema; NO se usa para localizar el polo
R0 = 10.0
P_MAX = 1e6       # paro: |p_r| > P_MAX  (equivale a r - r_s ~ 1e-6)
N_MUESTRA = 100   # puntos para AAA (submuestreo logarítmico)
TOL_IMAG = 1e-4   # polos "casi reales"
VENTANA = 0.5     # se buscan polos en [r_min - VENTANA, r_min]

CASOS = [("masivo", 0.97, 1.0),   # (nombre, E, kappa^2)
         ("fotón",  1.0,  0.0)]


def f(r):
    return 1.0 - RS / r


def f1(r):
    return RS / r**2


def resolver(E, kappa2):
    def sistema(lam, y):
        r, p = y
        W1 = -E**2 * f1(r) / f(r)**2
        return [p * f(r), 0.5 * W1 - 0.5 * f1(r) * p**2]

    def momento_grande(lam, y):
        return P_MAX - abs(y[1])
    momento_grande.terminal = True

    f0 = f(R0)
    p0 = -np.sqrt((E**2 - kappa2 * f0) / f0**2)       # rama entrante
    return solve_ivp(sistema, (0.0, 60.0), [R0, p0], events=momento_grande,
                     method='DOP853', rtol=1e-13, atol=1e-15, max_step=0.05)


def polo_residuo(r, p):
    """AAA sobre p_r(r) submuestreado; devuelve (r_p, Res, m, cociente)."""
    i = np.argsort(r)
    r, p = r[i], p[i]
    r, u = np.unique(r, return_index=True)
    p = p[u]
    d = r - r[0]
    idx = np.unique(np.searchsorted(d, np.geomspace(d[1], d[-1], N_MUESTRA))
                    .clip(0, len(r) - 1))
    fit = AAA(r[idx], p[idx], rtol=1e-10)
    pol, res = fit.poles(), fit.residues()
    k = (np.abs(pol.imag) < TOL_IMAG) & (pol.real < r[0]) & (pol.real > r[0] - VENTANA)
    pk, rk = pol[k].real, res[k].real
    j = np.argmax(np.abs(rk))
    # diagnóstico: mayor residuo restante en la ventana, relativo al dominante
    resto = np.delete(np.abs(rk), j)
    cociente = resto.max() / abs(rk[j]) if resto.size else 0.0
    return pk[j], rk[j], fit.support_points.size, cociente, r[0]


for nombre, E, kappa2 in CASOS:
    sol = resolver(E, kappa2)
    r_p, res, m, cociente, r_min = polo_residuo(sol.y[0], sol.y[1])
    print(f"===== Caso {nombre}: E={E}, kappa^2={kappa2} =====")
    print(f"  puntos integrados       : {sol.t.size}   (r_min - r_s = {r_min - RS:.2e})")
    if kappa2 == 0:
        exacta = -E * sol.y[0] / (sol.y[0] - RS)
        print(f"  error rel. vs exacta    : {np.max(np.abs(sol.y[1] / exacta - 1)):.1e}")
    print(f"  puntos soporte AAA (m)  : {m}")
    print(f"  polo (AAA)              : r_p = {r_p:.12f}")
    print(f"  residuo (AAA)           : Res = {res:.10f}")
    print(f"  residuo analítico -E r_s: {-E * RS:.10f}")
    print(f"  resto de residuos       : {cociente:.1e} del dominante")
    print(f"  Im(S) = pi|Res|         : {np.pi * abs(res):.6f}   (analítico {np.pi * E * RS:.6f})\n")
