"""
Residuo de p_r(r) con AAA en la carta AVANZADA de Eddington--Finkelstein.

Rama saliente no regular del fotón, p_r = 2E/(c f) (la que contiene el polo
del horizonte y, por tanto, Im S). Se inicializa fuera, en campo débil
(r0 = 6), y se integra HACIA ATRÁS en lambda, hacia el horizonte. Clásicamente

    p_r = 2E r/(r - r_s)  =>  Res = 2E r_s = 4,   Im S = pi |Res| = 4 pi.

Se compara el caso sin momentos con el sistema de segundo orden para dos
paquetes iniciales G20 = G02 = G0, G11 = 0. La integración se detiene cuando
|p_r| > 1e6 o algún momento supera 1e12.
"""
import numpy as np
from scipy.integrate import solve_ivp

from momentos_eqs import RS, H_ef, sistema_segundo_orden, polo_residuo

E = 1.0
R0 = 6.0
P0 = 2 * E / (1 - RS / R0)
RHS, _ = sistema_segundo_orden(H_ef(E, signo=-1))


def rhs_clasico(lam, y):
    d = RHS(lam, [y[0], y[1], 0.0, 0.0, 0.0])
    return [d[0], d[1], 0.0, 0.0, 0.0]


def integrar(G0, con_momentos=True):
    def momento_grande(lam, y):
        return 1e6 - abs(y[1])
    momento_grande.terminal = True

    def momentos_divergen(lam, y):
        return 1e12 - max(abs(y[2]), abs(y[3]), abs(y[4]))
    momentos_divergen.terminal = True

    return solve_ivp(RHS if con_momentos else rhs_clasico, (0.0, -60.0), [R0, P0, G0, 0.0, G0],
                     events=[momento_grande, momentos_divergen], method='DOP853',
                     rtol=1e-12, atol=1e-16, max_step=0.01)


def chequeo_directo(s, G0):
    """Producto (r - r_p) p_r sobre los datos crudos para varios candidatos de r_p:
    el horizonte, el cero de f_eff = f + f''G20/2 con los G iniciales
    (r_p ~ r_s + G0/2) y el polo que asigna AAA. Si hubiera un polo simple en
    r_p, el producto tendería al residuo (clásicamente 4)."""
    r_dat, p_dat = s.y[0], s.y[1]
    r_aaa, _ = polo_residuo(r_dat, p_dat)
    candidatos = {'horizonte': RS, 'f_eff=0': RS + G0 / 2, 'AAA': r_aaa}
    idx = [int(np.argmin(np.abs(r_dat - 2.9))), int(np.argmin(np.abs(r_dat - 2.4)))]
    idx += list(np.unique(np.linspace(0.97 * r_dat.size, r_dat.size - 1, 5).astype(int)))
    print("   Chequeo directo (r - r_p) p_r:  " + "  ".join(f"{k}: r_p={v:.6f}" for k, v in candidatos.items()))
    for i in idx:
        print(f"      r={r_dat[i]:.6f}  p_r={p_dat[i]:.4g}  " +
              "  ".join(f"{(r_dat[i]-v)*p_dat[i]:12.4g}" for v in candidatos.values()))


if __name__ == '__main__':
    print(f"Carta avanzada, rama saliente no regular: r0={R0}, p_r(0)={P0}, E={E}")
    print(f"Clásico: Res = 2E r_s = {2*E*RS}, Im S = {np.pi*2*E*RS:.6f}\n")
    for G0, con in [(0.0, False), (1e-2, True), (1e-4, True)]:
        s = integrar(G0, con)
        r_p, res = polo_residuo(s.y[0], s.y[1])
        sig = np.sqrt(abs(s.y[2, -1]))
        print(f"G0={G0:.0e} momentos={con}: paro en r={s.y[0,-1]:.6f} (lambda={s.t[-1]:.4f}), "
              f"sigma_r/(r-r_s)={sig/(s.y[0,-1]-RS):.3f}")
        print(f"   G11={s.y[3,-1]:.3e}  G02={s.y[4,-1]:.3e}")
        print(f"   AAA: r_p={r_p:.8f}  Res={res:.6g}  Im S=pi|Res|={np.pi*abs(res):.6g}")
        if con:
            chequeo_directo(s, G0)
        print()
