"""
Validez del desarrollo y carácter de la divergencia en Schwarzschild
(caída radial masiva, E = 0.97, r0 = 10, G20 = G02 = G0, G11 = 0).

1. Criterio de validez q = sigma_r/(r - r_s): como H contiene E^2/(2 f), sus
   derivadas d^n H/dr^n ~ (r - r_s)^{-(n+1)} y la serie en momentos solo tiene
   sentido mientras sigma_r << r - r_s.  Se reporta dónde q cruza 1 y su valor
   al divergir los momentos, para cada G0.
2. Exponentes cerca de lambda_c: ajuste log-log de |p_r|, |G02|, |G11| y
   |dr/dl| frente a (lambda_c - lambda), con varias ventanas de ajuste y dos
   cortes de seguridad, para medir la robustez.
3. Dependencia del punto de paro con el corte: r y G20 siguen bajando al subir
   el corte, pero q se mantiene cerca de 1 y f + f''G20/2 no se anula (la
   divergencia no ocurre en un "horizonte efectivo" f + f''G20/2 = 0).
"""
import numpy as np
from scipy.integrate import solve_ivp

from momentos_eqs import RS, H_schwarzschild, sistema_segundo_orden

E = 0.97
R0 = 10.0
F0 = 1 - RS / R0
P0 = -np.sqrt((E**2 - F0) / F0**2)          # capa de masa clásica, rama entrante
RHS, H_Q = sistema_segundo_orden(H_schwarzschild(E))


def integrar(G0, corte=1e14, rtol=1e-12):
    def momentos_divergen(lam, y):
        return corte - max(abs(y[2]), abs(y[3]), abs(y[4]))
    momentos_divergen.terminal = True
    return solve_ivp(RHS, (0, 80), [R0, P0, G0, 0.0, G0], events=momentos_divergen,
                     method='DOP853', rtol=rtol, atol=1e-15, max_step=0.01)


if __name__ == '__main__':
    print("== 1. Criterio q = sigma_r/(r - r_s) ==")
    print(f"{'G0':>7} {'lambda(q=1)':>12} {'r(q=1)':>8} {'lambda_paro':>12} {'q al paro':>10}")
    for G0 in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        s = integrar(G0)
        q = np.sqrt(np.abs(s.y[2])) / (s.y[0] - RS)
        k = int(np.argmax(q >= 1)) if np.any(q >= 1) else None
        txt = f"{s.t[k]:12.4f} {s.y[0,k]:8.4f}" if k is not None else f"{'---':>12} {'---':>8}"
        print(f"{G0:7.0e} {txt} {s.t[-1]:12.4f} {q[-1]:10.3f}")
    s = integrar(1e-2)
    i = int(np.argmin(np.abs(s.y[0] - 3.3646)))
    print(f"en r*=3.36 (G0=1e-2): q = {np.sqrt(s.y[2,i])/(s.y[0,i]-RS):.3f}")
    print(f"Ligadura inicial: H(0) = -0.5, H_Q(0) = {H_Q(R0, P0, 1e-2, 0.0, 1e-2):.6f}")

    print("\n== 2. Exponentes en (lambda_c - lambda), G0 = 1e-2 ==")
    ventanas = [(1e-7, 1e-3), (1e-6, 1e-3), (1e-6, 1e-4), (1e-5, 1e-2)]
    for corte in [1e20, 1e30]:
        s = integrar(1e-2, corte=corte, rtol=1e-13)
        lc = s.t[-1]
        dl = lc - s.t
        rdot = np.abs(np.array([RHS(0, y)[0] for y in s.y.T]))
        cantidades = {'|p_r|': np.abs(s.y[1]), '|G02|': np.abs(s.y[4]),
                      '|G11|': np.abs(s.y[3]), '|dr/dl|': rdot,
                      'G20': s.y[2], 'r-r_s': s.y[0] - RS}
        print(f"corte={corte:.0e}, lambda_c={lc:.8f}")
        for nombre, v in cantidades.items():
            exps = []
            for a, b in ventanas:
                m = (dl > a) & (dl < b)
                exps.append(np.polyfit(np.log(dl[m]), np.log(v[m]), 1)[0] if m.sum() > 5 else np.nan)
            print(f"   {nombre:>8}: " + "  ".join(f"[{a:.0e},{b:.0e}]: {e:+.3f}"
                                               for (a, b), e in zip(ventanas, exps)))
        print(f"   U fijo: G11^2/(G20 G02) en el paro = {s.y[3,-1]**2/(s.y[2,-1]*s.y[4,-1]):.8f}")

    print("\n== 3. q, r y G20 en el paro según el corte (G0 = 1e-2, 1e-4, 1e-6) ==")
    for G0 in [1e-2, 1e-4, 1e-6]:
        for corte in [1e6, 1e14, 1e30]:
            s = integrar(G0, corte=corte, rtol=1e-13)
            r, p, G20 = s.y[0, -1], s.y[1, -1], s.y[2, -1]
            f = 1 - RS / r
            feff = f - RS / r**3 * G20      # f + f''G20/2
            print(f"{G0:7.0e} corte={corte:.0e} lambda={s.t[-1]:.10f} r={r:.4f} "
                  f"G20={G20:.4f} q={np.sqrt(G20)/(r-RS):.3f} f+f''G20/2={feff:.3f}")
