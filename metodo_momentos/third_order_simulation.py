"""
Sistema AUTOCONSISTENTE a ORDEN 3 (9 variables): r, p_r, y los momentos
de 2º orden (G20, G11, G02) y 3er orden (G30, G21, G12, G03).
Ecuaciones verificadas simbólicamente con el álgebra de Poisson en la
convención posición-primero (ver third_order_derivation.py / G_poisson_brackets.py).
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
    # Corchetes en la convención posición-primero (G_poisson_brackets.pb,
    # intercambio de índices de la ec. 5 de Bojowald 2011).
    dG20 = 2 * f0 * G11 + 2 * f1_ * p_r * G20 + 2 * f1_ * G21 + f2_ * p_r * G30
    dG11 = (-0.5 * (f2_ * p_r**2 - W2_) * G20 + f0 * G02
            - 0.25 * (f3_ * p_r**2 - W3_) * G30
            - 0.5 * f2_ * p_r * G21 + 0.5 * f1_ * G12)
    dG02 = (-(f2_ * p_r**2 - W2_) * G11 - 2 * f1_ * p_r * G02
            - 0.5 * (f3_ * p_r**2 - W3_) * G21
            - 2 * f2_ * p_r * G12 - f1_ * G03)

    # --- momentos de 3er orden (subsistema homogéneo) ---
    dG30 = 3 * f1_ * p_r * G30 + 3 * f0 * G21
    dG21 = -0.5 * (f2_ * p_r**2 - W2_) * G30 + f1_ * p_r * G21 + 2 * f0 * G12
    dG12 = -(f2_ * p_r**2 - W2_) * G21 - f1_ * p_r * G12 + f0 * G03
    dG03 = -1.5 * (f2_ * p_r**2 - W2_) * G12 - 3 * f1_ * p_r * G03

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

    # t_eval muestrea cada 0.004 en lambda; se añade el estado exacto del evento
    # terminal para que r_min y lambda_fin correspondan al punto de paro real.
    for t_ev, y_ev in zip(sol.t_events, sol.y_events):
        if len(t_ev):
            sol.t = np.append(sol.t, t_ev[-1])
            sol.y = np.hstack([sol.y, y_ev[-1][:, None]])
    r_sol = sol.y[0]
    i_rebote = int(np.argmin(r_sol))
    return sol, r_sol[i_rebote], r_sol.min()


def resumen(sol):
    r_sol = sol.y[0]
    i = int(np.argmin(r_sol))
    rebota = i < r_sol.size - 1
    motivo = {1: 'divergen los momentos', -1: 'el paso del integrador colapsa',
              0: 'fin del intervalo'}[sol.status]
    if sol.status == 1 and len(sol.t_events[0]):
        motivo = 'llega al horizonte'
    return r_sol[i], sol.t[i], rebota, sol.t[-1], motivo


if __name__ == '__main__':
    print("PRUEBA 1 — paquete gaussiano (momentos de 3er orden = 0):")
    sol, _, _ = correr(G3_0=0.0)
    r_min, lam_min, rebota, lam_fin, motivo = resumen(sol)
    print(f"   r_min = {r_min:.4f} (lambda={lam_min:.4f}), rebote: {rebota}, paro en lambda={lam_fin:.4f} ({motivo})")
    print(f"   ¿G^(3) se mantienen en cero? max|G30..G03| = {np.max(np.abs(sol.y[5:])):.2e}")

    print("\nPRUEBA 2 — paquete NO gaussiano (momentos de 3er orden != 0):")
    for g3_0 in [1e-4, 1e-3, 1e-2]:
        sol, _, _ = correr(G3_0=g3_0)
        r_min, lam_min, rebota, lam_fin, motivo = resumen(sol)
        print(f"   G^(3)_0={g3_0:.0e} -> r_min = {r_min:.4f} (lambda={lam_min:.4f}), "
              f"rebote: {rebota}, paro en lambda={lam_fin:.4f} ({motivo})")

    # ---------------- FIGURA: trayectoria y cota de incertidumbre ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HBAR = 2 * 1e-2                      # hbar = 2 G0  (estado de mínima incertidumbre)
    COTA = HBAR**2 / 4
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.25})
    fig, (ax_r, ax_U) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colores = {0.0: 'k', 1e-4: '#1f77b4', 1e-3: '#ff7f0e', 1e-2: '#d62728'}
    for g3_0, color in colores.items():
        sol, _, _ = correr(G3_0=g3_0)
        lam, y = sol.t, sol.y
        m = lam <= 16
        etiqueta = r'$G^{(3)}_0=0$ (gaussiano)' if g3_0 == 0 else fr'$G^{{(3)}}_0=10^{{{int(np.log10(g3_0))}}}$'
        ax_r.plot(lam[m], y[0][m], color=color, lw=2.2, zorder=3, label=etiqueta)
        U = (y[2]*y[4] - y[3]**2) / COTA
        ax_U.plot(lam[m], U[m], color=color, lw=2.2, zorder=3, label=etiqueta)
        neg = np.where(((y[2] < 0) | (y[4] < 0)) & m)[0]
        if neg.size:                     # primera varianza negativa
            k = neg[0]
            ax_r.plot(lam[k], y[0][k], 'X', color=color, ms=11, zorder=6)
            ax_U.plot(lam[k], U[k], 'X', color=color, ms=11, zorder=6)
    ax_r.axhline(rs, color='gray', ls=':', lw=1.4)
    ax_r.text(0.3, rs + 0.15, 'horizonte $2m$', fontsize=9, color='gray')
    ax_r.set_xlim(0, 16); ax_r.set_ylim(0, 11)
    ax_r.set_xlabel(r'$\lambda$'); ax_r.set_ylabel(r'$r$')
    ax_r.set_title(r'Trayectoria $r(\lambda)$ (aspa: primera varianza negativa)')
    ax_r.legend(fontsize=9, loc='upper right')
    ax_U.axhline(1.0, color='crimson', ls=(0, (4, 4)), lw=1.0, zorder=1, label=r'cota $U=\hbar^2/4$')
    ax_U.set_yscale('symlog', linthresh=1.0)
    ax_U.set_ylim(-3e10, 10)
    ax_U.axhspan(-3e10, 1.0, color='crimson', alpha=0.06)
    ax_U.set_xlim(0, 16)
    ax_U.set_xlabel(r'$\lambda$'); ax_U.set_ylabel(r'$U/(\hbar^2/4)$  (symlog)')
    ax_U.set_title(r'$U=G^{2,0}G^{0,2}-(G^{1,1})^2$ a tercer orden')
    ax_U.legend(fontsize=9, loc='lower left')
    fig.suptitle('Tercer orden: solo el paquete gaussiano respeta la cota de incertidumbre', y=1.00, fontsize=12.5)
    fig.tight_layout()
    fig.savefig('third_order_cota.png', dpi=140, bbox_inches='tight')
    print("\nFigura guardada: third_order_cota.png")

