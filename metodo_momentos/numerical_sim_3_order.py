"""
Sistema AUTOCONSISTENTE a ORDEN 3 (9 variables): r, p_r, y los momentos
de 2º orden (G20, G11, G02) y 3er orden (G30, G21, G12, G03).
Ecuaciones verificadas simbólicamente con el álgebra de Poisson
(ver orden3_deriv.py / G_poisson_brackets.py).
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

c = 1.0
E = 0.97
rs = 2.0


def f(r):
    return 1 - rs / r


def f1(r):
    return rs / r**2


def f2(r):
    return -2 * rs / r**3


def f3(r):
    return 6 * rs / r**4


def f4(r):
    return -24 * rs / r**5


# W = E^2 / (f c^2) y sus derivadas, precompiladas como funciones rápidas
_r = sp.symbols('r')
_W = E**2 / ((1 - rs / _r) * c**2)
W1 = sp.lambdify(_r, sp.diff(_W, _r, 1), 'numpy')
W2 = sp.lambdify(_r, sp.diff(_W, _r, 2), 'numpy')
W3 = sp.lambdify(_r, sp.diff(_W, _r, 3), 'numpy')
W4 = sp.lambdify(_r, sp.diff(_W, _r, 4), 'numpy')


def rhs(lam, y):
    r, p_r, G20, G11, G02, G30, G21, G12, G03 = y
    f0, f1_, f2_, f3_, f4_ = f(r), f1(r), f2(r), f3(r), f4(r)
    W1_, W2_, W3_, W4_ = W1(r), W2(r), W3(r), W4(r)

    # --- posición y momento (orden 3) ---
    dr = (p_r * f0 + f1_ * G11
          + 0.5 * p_r * G20 * f2_
          + 0.5 * f2_ * G21
          + (1 / 6) * f3_ * p_r * G30)
    dp_r = (0.5 * W1_ - 0.5 * f1_ * p_r**2
            - 0.5 * f1_ * G02 - f2_ * p_r * G11
            + 0.25 * W3_ * G20 - 0.25 * f3_ * p_r**2 * G20
            - 0.5 * f2_ * G12 - 0.5 * f3_ * p_r * G21
            + (1 / 12) * (W4_ - f4_ * p_r**2) * G30)

    # --- momentos de 2º orden (con acople a los de 3er orden) ---
    dG20 = -2 * f0 * G11 - 2 * f1_ * p_r * G20 - 2 * f1_ * G21 - f2_ * p_r * G30
    dG11 = (0.5 * (f2_ * p_r**2 - W2_) * G20 - f0 * G02
            + 0.25 * (f3_ * p_r**2 - W3_) * G30
            + 0.5 * f2_ * p_r * G21 - 0.5 * f1_ * G12)
    dG02 = ((f2_ * p_r**2 - W2_) * G11 + 2 * f1_ * p_r * G02
            + 0.5 * (f3_ * p_r**2 - W3_) * G21
            + 2 * f2_ * p_r * G12 + f1_ * G03)

    # --- momentos de 3er orden (subsistema homogéneo) ---
    dG30 = -3 * f1_ * p_r * G30 - 3 * f0 * G21
    dG21 = 0.5 * (f2_ * p_r**2 - W2_) * G30 - f1_ * p_r * G21 - 2 * f0 * G12
    dG12 = (f2_ * p_r**2 - W2_) * G21 + f1_ * p_r * G12 - f0 * G03
    dG03 = 1.5 * (f2_ * p_r**2 - W2_) * G12 + 3 * f1_ * p_r * G03

    return [dr, dp_r, dG20, dG11, dG02, dG30, dG21, dG12, dG03]


def correr(G3_0, G2_0=1e-2, r0=10.0):
    """Integra desde r0 hasta el rebote o el horizonte.
    G3_0: valor inicial común de los momentos de 3er orden (0 = paquete gaussiano).
    G2_0: valor inicial común de G20 y G02 (G11_0 = 0)."""
    f0 = 1 - rs / r0
    p_r0 = -np.sqrt((E**2 - f0) / f0**2)   # m = 1, rama entrante, on-shell
    y0 = [r0, p_r0, G2_0, 0.0, G2_0, G3_0, 0.0, 0.0, G3_0]

    def llega_al_horizonte(lam, y):
        return y[0] - 1.00001 * rs
    llega_al_horizonte.terminal = True
    llega_al_horizonte.direction = -1

    def sale_de_validez(lam, y):
        return 1e14 - max(abs(v) for v in y[2:])
    sale_de_validez.terminal = True
    sale_de_validez.direction = -1

    sol = solve_ivp(rhs, (0, 80), y0, t_eval=np.linspace(0, 80, 20000),
                     events=[llega_al_horizonte, sale_de_validez],
                     method='DOP853', rtol=1e-11, atol=1e-14)

    r_sol = sol.y[0]
    i_rebote = int(np.argmin(r_sol))
    return sol, r_sol[i_rebote], r_sol.min()


if __name__ == '__main__':
    print("PRUEBA 1 — paquete Gaussiano (momentos 3er orden = 0):")
    sol, r_min, _ = correr(G3_0=0.0)
    print(f"   r_min = {r_min:.4f}   (orden 2 daba 3.2031)")
    max_g3 = np.max(np.abs(sol.y[5:]))
    print(f"   ¿G^(3) se mantienen en cero? max|G30..G03| = {max_g3:.2e}")

    print("\nPRUEBA 2 — paquete NO Gaussiano (momentos 3er orden != 0):")
    for g3_0 in [1e-4, 1e-3, 1e-2]:
        _, r_min, _ = correr(G3_0=g3_0)
        print(f"   G^(3)_0={g3_0:.0e}  ->  r_min = {r_min:.4f}")
