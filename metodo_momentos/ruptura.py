"""
=============================================================================
 Ruptura del desacople  G ≈ 0  cerca del horizonte de Schwarzschild
 y contraste con el ansatz WKB-gaussiano.
=============================================================================

 Reproduce tres resultados:
   (R1) Los G divergen como potencias de f = 1 - 2m/r, con exponentes que
        salen de una ecuación indicial:  a² + 4a - 16 = 0  →  a = 2√5 - 2.
   (R2) El desacople "fondo clásico + G ≈ 0" se autofalsifica: las correcciones
        con G superan a los términos clásicos MUCHO antes del horizonte.
   (R3) El ansatz WKB-gaussiano con σ constante es inconsistente con la
        dinámica cerca del horizonte (predice a = 0, la dinámica exige a ≈ 2.47).

 Unidades geometrizadas c = G = 1, horizonte en r_s = 2m_BH = 2. Caso m = 0.
=============================================================================
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# =========================================================================
# ETAPA 0 — CONSTANTES
# =========================================================================
E_val  = 1.10      # energía conservada
m_val  = 0.0       # masa de la partícula de prueba (0 = geodésica nula)
rs_val = 2.0       # radio de Schwarzschild


# =========================================================================
# ETAPA 1 — INGREDIENTES SIMBÓLICOS
# =========================================================================
r = sp.symbols('r', positive=True)
E, m, rs = sp.symbols('E m r_s', positive=True)

f = 1 - rs/r
W = E**2/f                                   # potencial E²/(f c²)
f1, f2, f3 = [sp.diff(f, r, k) for k in (1, 2, 3)]
W1, W2, W3 = [sp.diff(W, r, k) for k in (1, 2, 3)]

pr  = -sp.sqrt(E**2 - m**2*f)/f              # p_r on-shell (rama entrante)
Spp = sp.diff(pr, r)                         # S''(q) = dp/dr : curvatura de la fase WKB

Hpp = f                                      # ∂²H/∂p²
Hrp = f1*pr                                  # ∂²H/∂r∂p
Hrr = sp.Rational(1, 2)*(f2*pr**2 - W2)      # ∂²H/∂r²

subs = {E: E_val, m: m_val, rs: rs_val}
N = lambda e: sp.lambdify((r,), e.subs(subs), 'numpy')
f_, f1_, f2_, f3_ = N(f), N(f1), N(f2), N(f3)
W1_, W3_ = N(W1), N(W3)
pr_, Spp_ = N(pr), N(Spp)
Hpp_, Hrp_, Hrr_ = N(Hpp), N(Hrp), N(Hrr)


# =========================================================================
# ETAPA 2 — MOMENTOS SOBRE FONDO CLÁSICO, CON r COMO VARIABLE INDEPENDIENTE
# -------------------------------------------------------------------------
#   dG/dr = (dG/dλ) / (dr/dλ),   dr/dλ = f p_r   (= -E si m = 0)
# =========================================================================
def dG_dr(rq, G):
    G20, G11, G02 = G
    drdl = f_(rq)*pr_(rq)
    dG20 = -2*Hrp_(rq)*G20 - 2*Hpp_(rq)*G11
    dG11 =    Hrr_(rq)*G20 -   Hpp_(rq)*G02
    dG02 =  2*Hrr_(rq)*G11 + 2*Hrp_(rq)*G02
    return [dG20/drdl, dG11/drdl, dG02/drdl]


# ---- Condiciones iniciales (paquete casi clásico en r0 = 6) -------------
r0 = 6.0
G20_0, G11_0, G02_0 = 1e-2, 0.0, 1e-2
# Invariante de incertidumbre U = G20 G02 - G11²  =  (ħ_eff/2)²
U0 = G20_0*G02_0 - G11_0**2
hbar_eff = 2*np.sqrt(U0)

sol = solve_ivp(dG_dr, (r0, rs_val + 1e-8), [G20_0, G11_0, G02_0],
                method='Radau', rtol=1e-11, atol=1e-18,
                dense_output=True, max_step=0.005)


# =========================================================================
# (R1) EXPONENTES DE LA DIVERGENCIA
# -------------------------------------------------------------------------
# Ansatz  G20 ~ A f^-a,  G11 ~ B f^-(a+2),  G02 ~ C f^-(a+4)
# sustituido en las ecuaciones cerca del horizonte da  (a+2)(a²+4a-16) = 0.
# =========================================================================
a_exact = 2*np.sqrt(5) - 2
print("=" * 70)
print("(R1)  EXPONENTES DE DIVERGENCIA")
print("=" * 70)
print(f"  Ecuación indicial: a² + 4a - 16 = 0   →   a = 2√5 - 2 = {a_exact:.4f}")
print(f"  Predicción:  G20 ~ f^-{a_exact:.3f},  G11 ~ f^-{a_exact+2:.3f},  "
      f"G02 ~ f^-{a_exact+4:.3f},  p_r ~ f^-1")

rr = sol.t
G20s, G11s, G02s = sol.y
mask = (rr < 2.02) & (rr > rs_val + 1e-7)
log_f = np.log(f_(rr[mask]))
print("\n  Ajuste numérico  log|X| = n·log f + cte   (región r < 2.02):")
for name, series in [('p_r', pr_(rr[mask])), ('G20', G20s[mask]),
                     ('G11', G11s[mask]), ('G02', G02s[mask])]:
    n = np.polyfit(log_f, np.log(np.abs(series)), 1)[0]
    print(f"     |{name}| ~ f^({n:+.3f})")


# =========================================================================
# (R2) RUPTURA DEL DESACOPLE:  |corrección con G| / |término clásico|
# -------------------------------------------------------------------------
# En  ṙ  : clásico = p_r f ;  corrección = f' G11 + ½ p_r G20 f''
# En  ṗ_r: clásico = ½W' - ½f' p_r² ;
#          corrección = -½f' G02 - f'' p_r G11 + ¼W''' G20 - ¼f''' p_r² G20
# =========================================================================
def ratio_r(rq):
    G20, G11, G02 = sol.sol(rq)
    clas = abs(pr_(rq)*f_(rq))
    quan = abs(f1_(rq)*G11 + 0.5*pr_(rq)*G20*f2_(rq))
    return quan/clas

def ratio_p(rq):
    G20, G11, G02 = sol.sol(rq)
    clas = abs(0.5*W1_(rq) - 0.5*f1_(rq)*pr_(rq)**2)
    quan = abs(-0.5*f1_(rq)*G02 - f2_(rq)*pr_(rq)*G11
               + 0.25*W3_(rq)*G20 - 0.25*f3_(rq)*pr_(rq)**2*G20)
    return quan/clas

print("\n" + "=" * 70)
print("(R2)  RUPTURA DEL DESACOPLE  G ≈ 0")
print("=" * 70)
print(f"{'r':>7} {'f':>10} {'ratio en ṙ':>14} {'ratio en ṗ_r':>15}")
for rq in [5.0, 4.0, 3.0, 2.5, 2.3, 2.2, 2.1, 2.05, 2.01]:
    print(f"{rq:7.3f} {f_(rq):10.2e} {ratio_r(rq):14.3e} {ratio_p(rq):15.3e}")

r_break_p = brentq(lambda x: ratio_p(x) - 1.0, 2.001, 3.5)
r_break_r = brentq(lambda x: ratio_r(x) - 1.0, 2.001, 3.5)
print(f"\n  La aproximación se rompe (cociente = 1):")
print(f"     ecuación de ṗ_r :  r = {r_break_p:.4f}   (f = {f_(r_break_p):.3e})")
print(f"     ecuación de ṙ   :  r = {r_break_r:.4f}   (f = {f_(r_break_r):.3e})")
print(f"     → ambos MUY por fuera del horizonte r_s = {rs_val}")


# =========================================================================
# (R3) CONTRASTE CON EL ANSATZ WKB-GAUSSIANO
# -------------------------------------------------------------------------
#   Ψ(q) = (2πσ²)^(-1/4) |p(q)|^(-1/2) exp[-(q-q0)²/(4σ²) + i S(q)/ħ]
# Expandiendo S(q) a orden cuadrático (S' = p, S'' = dp/dr) se obtiene
#   G20 = σ²,   G11 = σ² S'',   G02 = ħ²/(4σ²) + σ² S''²
# de modo que el ANSATZ predice   G11 / (G20 · S'') = 1.
# La DINÁMICA, en cambio, exige   G11 / (G20 · S'') = (2 - a)/2 ≈ -0.236.
# =========================================================================
print("\n" + "=" * 70)
print("(R3)  ¿ES CONSISTENTE EL ANSATZ WKB-GAUSSIANO?")
print("=" * 70)
print(f"  Ansatz (σ constante) predice:  G11/(G20·S'') = +1.000")
print(f"  Dinámica exige:               G11/(G20·S'') = (2-a)/2 = {(2-a_exact)/2:+.4f}")
print(f"\n{'r':>7} {'f':>10} {'G11/(G20·S″)':>15} {'U = G20G02-G11²':>18}")
for rq in [3.0, 2.5, 2.2, 2.1, 2.05, 2.02]:
    G20, G11, G02 = sol.sol(rq)
    print(f"{rq:7.3f} {f_(rq):10.2e} {G11/(G20*Spp_(rq)):15.5f} {G20*G02 - G11**2:18.6e}")

print(f"\n  → El cociente converge a {(2-a_exact)/2:.4f}, NO a +1.")
print(f"  → σ no puede ser constante:  σ = √G20 ~ f^-(√5-1) = f^-{np.sqrt(5)-1:.4f}")
print(f"\n  Invariante de incertidumbre U = (ħ_eff/2)² = {U0:.3e}  "
      f"(ħ_eff = {hbar_eff:.3f})")
print("  U se conserva exactamente en las ecuaciones; su pérdida numérica")
print("  cerca de r≈2.02 es cancelación catastrófica (G20·G02 ~ 1e15 vs U ~ 1e-4),")
print("  y sirve como diagnóstico del límite de la doble precisión.")


# =========================================================================
# (R4) JERARQUÍA DE TÉRMINOS cerca del horizonte  (¿cuáles dominan?)
# =========================================================================
print("\n" + "=" * 70)
print("(R4)  JERARQUÍA DE TÉRMINOS  (potencia de f, a = 2√5-2)")
print("=" * 70)
print(f"  En ṙ:")
print(f"     p_r f              ~ f^0        (clásico)")
print(f"     f' G11             ~ f^-{a_exact+2:.2f}   ← corrección dominante")
print(f"     ½ p_r G20 f''      ~ f^-{a_exact+1:.2f}")
print(f"  En ṗ_r:")
print(f"     ½W' , ½f' p_r²     ~ f^-2       (clásico)")
print(f"     ½ f' G02           ~ f^-{a_exact+4:.2f}   ← corrección dominante")
print(f"     ¼ W''' G20         ~ f^-{a_exact+4:.2f}   ← corrección dominante")
print(f"     f'' p_r G11        ~ f^-{a_exact+3:.2f}")
print(f"     ¼ f''' p_r² G20    ~ f^-{a_exact+2:.2f}")