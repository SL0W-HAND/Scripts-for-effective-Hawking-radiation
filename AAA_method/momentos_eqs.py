r"""
Utilidades comunes para los scripts de AAA_method.

- `sistema_segundo_orden(H)`: a partir de un Hamiltoniano clásico H(r, p_r)
  construye H_Q a segundo orden y devuelve el lado derecho numérico de
      dr/dl = dH_Q/dp_r,  dp_r/dl = -dH_Q/dr,
      dG^{a,b}/dl = sum_{c,d} h_{c,d} {G^{a,b}, G^{c,d}},
  con los corchetes en la convención posición-primero, obtenidos por
  intercambio de índices de la fórmula de Bojowald et al. (2011):
      {G^{a,b}, G^{c,d}} = swap({G_B^{b,a}, G_B^{d,c}})   (G_poisson_brackets.pb)
- `H_schwarzschild`, `H_ef`: Hamiltonianos clásicos de caída radial.
- `polo_residuo`: selección de polo de AAA igual que metodo_momentos/pole.py
  (submuestreo logarítmico en r - r_min y polo casi real de mayor |Res|
  justo por debajo del menor radio integrado).
- `suma_residuos`: suma de los residuos de TODOS los polos de la aproximación
  AAA a distancia < radio del menor radio integrado, es decir,
  (1/2 pi i) \oint R(z) dz sobre esa circunferencia. Con momentos el polo del
  horizonte se reparte entre varios polos cercanos, y la suma es la cantidad
  estable.

Unidades c = G = M = 1  =>  r_s = 2.
"""
import os
import sys

import numpy as np
import sympy as sp
from scipy.interpolate import AAA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metodo_momentos'))
from G_poisson_brackets import pb, G as GM, reducir_momentos  # noqa: E402

RS = 2.0
r, p_r = sp.symbols('r p_r', real=True)
G20, G11, G02 = sp.symbols('G20 G11 G02', real=True)
ESTADO = (r, p_r, G20, G11, G02)
MOMENTOS_2 = [(2, 0), (1, 1), (0, 2)]
SIMBOLO_G = {(2, 0): G20, (1, 1): G11, (0, 2): G02}
f = 1 - RS / r


def H_schwarzschild(E):
    """H = 1/2 (f p_r^2 - E^2/f)  (caída radial, c = 1)."""
    return sp.Rational(1, 2) * (f * p_r**2 - E**2 / f)


def H_ef(E, signo=-1):
    """H = 1/2 f p_r^2 + signo*E p_r;  signo=-1 avanzada, +1 retardada (c = 1)."""
    return sp.Rational(1, 2) * f * p_r**2 + signo * E * p_r


def sistema_segundo_orden(H):
    """Devuelve (rhs, H_Q_num): rhs(l, y) con y = [r, p_r, G20, G11, G02]."""
    h = {(i, j): sp.diff(H, r, i, p_r, j) / (sp.factorial(i) * sp.factorial(j))
         for (i, j) in MOMENTOS_2}
    H_Q = H + sum(h[k] * SIMBOLO_G[k] for k in MOMENTOS_2)
    a_simbolos = {GM(*k): SIMBOLO_G[k] for k in MOMENTOS_2}
    dG = []
    for (a, b) in MOMENTOS_2:
        lado = sum(h[(c, d)] * reducir_momentos(pb(a, b, c, d)) for (c, d) in MOMENTOS_2)
        dG.append(sp.sympify(lado).subs(a_simbolos))
    exprs = [sp.diff(H_Q, p_r), -sp.diff(H_Q, r)] + dG
    funcs = [sp.lambdify(ESTADO, e, 'numpy') for e in exprs]
    H_Q_num = sp.lambdify(ESTADO, H_Q, 'numpy')

    def rhs(lam, y):
        return [fn(*y) for fn in funcs]
    return rhs, H_Q_num


def _aaa(r_dat, p_dat, n_muestra=100, rtol=1e-10):
    i = np.argsort(r_dat)
    rr, pp = np.asarray(r_dat)[i], np.asarray(p_dat)[i]
    rr, u = np.unique(rr, return_index=True)
    pp = pp[u]
    d = rr - rr[0]
    idx = np.unique(np.searchsorted(d, np.geomspace(max(d[1], 1e-15), d[-1], n_muestra))
                    .clip(0, rr.size - 1))
    return rr[0], AAA(rr[idx], pp[idx], rtol=rtol)


def suma_residuos(r_dat, p_dat, radio=0.5, **kw):
    """Suma de residuos de los polos de AAA con |z - r_min| < radio."""
    r_min, fit = _aaa(r_dat, p_dat, **kw)
    pol, res = fit.poles(), fit.residues()
    return res[np.abs(pol - r_min) < radio].sum().real


def polo_residuo(r_dat, p_dat, n_muestra=100, tol_imag=1e-4, ventana=0.5, rtol=1e-10):
    """Polo casi real de mayor |Res| en [r_min - ventana, r_min]; (nan, nan) si no hay."""
    i = np.argsort(r_dat)
    rr, pp = np.asarray(r_dat)[i], np.asarray(p_dat)[i]
    rr, u = np.unique(rr, return_index=True)
    pp = pp[u]
    if rr.size < 10:
        return np.nan, np.nan
    d = rr - rr[0]
    idx = np.unique(np.searchsorted(d, np.geomspace(max(d[1], 1e-15), d[-1], n_muestra))
                    .clip(0, rr.size - 1))
    fit = AAA(rr[idx], pp[idx], rtol=rtol)
    pol, res = fit.poles(), fit.residues()
    k = (np.abs(pol.imag) < tol_imag) & (pol.real < rr[0]) & (pol.real > rr[0] - ventana)
    if not k.any():
        return np.nan, np.nan
    j = int(np.argmax(np.abs(res[k])))
    return pol[k][j].real, res[k][j].real
