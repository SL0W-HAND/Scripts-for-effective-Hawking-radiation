"""
Barrido en el paquete inicial G0 del residuo que estima AAA en las dos cartas.

- Schwarzschild: fotón (E=1) que cae desde r0=10, p_r = -E/f (Res exacto -2).
- EF avanzada: rama saliente no regular desde r0=6 hacia atrás en lambda,
  p_r = 2E/f (Res exacto +4).

Para cada G0 se estima el residuo con todos los datos integrados y con la
ventana donde el desarrollo en momentos sigue siendo válido,
q = sigma_r/(r - r_s) < 0.5, tomando el polo de mayor |Res| (polo_residuo).
Además se da la suma de los residuos de todos los polos a distancia < R del
menor radio integrado (suma_residuos) para R = 0.2, 0.5 y 1: con momentos el
polo del horizonte se reparte entre varios polos (p. ej. uno en r_s - delta y
otro junto al punto de divergencia), y el de mayor |Res| solo lleva una parte.
Para medir la robustez de la suma se da su rango al variar los parámetros de
AAA (n_muestra = 60, 100, 150, 200 y rtol = 1e-8, 1e-10, 1e-12) con R = 0.5.
G0 = 0 es el caso clásico (los momentos permanecen nulos): un solo polo.

Al final, para G0 = 1e-10 y 1e-6 y tres juegos de parámetros de AAA, se listan
los polos cercanos al horizonte (posición en unidades de d = r_paro - r_s) con
sus residuos, y los momentos de la distribución de polos
    M_n = sum_k Res_k (z_k - r_s)^n,
que son los coeficientes del desarrollo lejano
    R(z) = M_0/(z-r_s) + M_1/(z-r_s)^2 + M_2/(z-r_s)^3 + ...
M_0 es la suma de residuos.
"""
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from momentos_eqs import RS, H_schwarzschild, H_ef, sistema_segundo_orden, polo_residuo, suma_residuos, _aaa

warnings.filterwarnings('ignore')
E = 1.0
Q_MAX = 0.5
G0S = [0.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]   # G0 = 0: caso clásico
CASOS = [
    ('Schwarzschild (fotón que cae, r0=10), Res exacto = -2',
     sistema_segundo_orden(H_schwarzschild(E))[0], 10.0, lambda r0: -E / (1 - RS / r0), (0, 60)),
    ('EF avanzada (saliente no regular, hacia atrás desde r0=6), Res exacto = +4',
     sistema_segundo_orden(H_ef(E, -1))[0], 6.0, lambda r0: 2 * E / (1 - RS / r0), (0, -60)),
]


def rango_suma(r_dat, p_dat, radio=0.5):
    v = [suma_residuos(r_dat, p_dat, radio, n_muestra=n, rtol=t)
         for n in (60, 100, 150, 200) for t in (1e-8, 1e-10, 1e-12)]
    return min(v), max(v)


def integrar(rhs, r0, p0, G0, tramo):
    def momento_grande(lam, y):
        return 1e6 - abs(y[1])
    momento_grande.terminal = True

    def momentos_divergen(lam, y):
        return 1e12 - max(abs(y[2]), abs(y[3]), abs(y[4]))
    momentos_divergen.terminal = True
    return solve_ivp(rhs, tramo, [r0, p0, G0, 0.0, G0], events=[momento_grande, momentos_divergen],
                     method='DOP853', rtol=1e-12, atol=1e-16, max_step=0.01)


if __name__ == '__main__':
    for nombre, rhs, r0, p_inicial, tramo in CASOS:
        print(f"\n## {nombre}")
        print(f"{'G0':>7} {'r_paro':>10} {'Res(todo)':>11} {'Res(q<0.5)':>11} {'r_min(q<0.5)':>13}"
              f" {'suma R=0.2':>11} {'suma R=0.5':>11} {'suma R=1':>10} {'rango (param. AAA)':>22}")
        for G0 in G0S:
            s = integrar(rhs, r0, p_inicial(r0), G0, tramo)
            r_dat, p_dat = s.y[0], s.y[1]
            q = np.sqrt(np.abs(s.y[2])) / (r_dat - RS)
            _, res_todo = polo_residuo(r_dat, p_dat)
            valido = q < Q_MAX
            _, res_valido = polo_residuo(r_dat[valido], p_dat[valido])
            print(f"{G0:7.0e} {r_dat[-1]:10.6f} {res_todo:11.4g} {res_valido:11.5g} "
                  f"{r_dat[valido].min():13.6f} " +
                  " ".join(f"{suma_residuos(r_dat, p_dat, R):11.5f}" for R in (0.2, 0.5, 1.0)) +
                  "   [{:+.4f}, {:+.4f}]".format(*rango_suma(r_dat, p_dat)))

    for G0 in (1e-10, 1e-6):
        for nombre, rhs, r0, p_inicial, tramo in CASOS:
            s = integrar(rhs, r0, p_inicial(r0), G0, tramo)
            r_dat, p_dat = s.y[0], s.y[1]
            d = r_dat[-1] - RS
            print(f"\n## Polos cercanos al horizonte, G0={G0:.0e}: {nombre}")
            print(f"   d = r_paro - r_s = {d:.3e},  sigma_r en el paro = {np.sqrt(abs(s.y[2][-1])):.3e}")
            for n, t in ((60, 1e-8), (100, 1e-10), (200, 1e-12)):
                _, fit = _aaa(r_dat, p_dat, n_muestra=n, rtol=t)
                pol, res = fit.poles(), fit.residues()
                c = (np.abs(pol - RS) < 50 * d) & (np.abs(res) > 1e-2)
                z, cr = pol[c] - RS, res[c]
                polos = "  ".join(f"({(zz / d).real:+.2f}{(zz / d).imag:+.2f}i: {cc.real:+.3f})"
                                  for zz, cc in sorted(zip(z, cr), key=lambda x: (x[0].real, x[0].imag)))
                M = [np.sum(cr * z**k).real for k in range(3)]
                print(f"   n={n:3d}, rtol={t:.0e}: {polos}")
                print(f"      M0 = {M[0]:+.5f}   M1/d = {M[1] / d:+.1e}   M2/d^2 = {M[2] / d**2:+.2f}")
