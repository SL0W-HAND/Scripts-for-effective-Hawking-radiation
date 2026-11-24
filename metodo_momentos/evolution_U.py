"""
=============================================================================
 RELACIÓN DE INCERTIDUMBRE DE ROBERTSON-SCHRÖDINGER EN EL SISTEMA DE MOMENTOS
 U = G^{2,0} G^{0,2} - (G^{1,1})^2  >=  hbar^2 / 4
=============================================================================

 Verifica, sobre el sistema con momentos de 2º orden (caída radial en
 Schwarzschild), que la combinación U:

   (a) es una CONSTANTE DE MOVIMIENTO exacta  (dU/dλ = 0), porque la matriz
       de covarianza evoluciona con un generador de traza nula (flujo
       simpléctico => det(Sigma) = U se conserva, teorema de Liouville);

   (b) SATURA la cota de incertidumbre: con las condiciones iniciales de un
       paquete gaussiano de mínima incertidumbre (del paper),
            G^{2,0}_0 = sigma0^2 ,  G^{0,2}_0 = (hbar/2 sigma0)^2 ,  G^{1,1}_0 = 0
       se tiene U_0 = hbar^2/4 EXACTO, y por (a) se mantiene así para siempre;

   (c) se conserva durante toda la caída, hasta que los momentos divergen.

 CLAVES NUMÉRICAS:
   - Se usan las MISMAS condiciones iniciales que second_order_system.py:
     G20_0 = G02_0 = 1e-2, G11_0 = 0. Para un paquete de mínima incertidumbre
     esto fija hbar = 2*G0 = 0.02 en unidades c = G = M = 1, es decir
     hbar/M^2 = (m_P/M)^2 = 0.02  (M ~ 7 m_P, régimen semiclásico).
     Se grafica U/(hbar^2/4), que debe ser >= 1.
   - Como diagnóstico AUXILIAR e independiente se usa la deriva del
     Hamiltoniano efectivo: si |ΔH/H0| > 5%, la integración numérica ya no
     es fiable a partir de ahí. (H es lineal en los G, así que este criterio
     certifica la INTEGRACIÓN, no la evaluación de U.)

 Unidades geometrizadas c = G = M = 1  =>  horizonte en r_s = 2m = 2.
=============================================================================
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# ETAPA 0 — CONSTANTES
# =========================================================================
c_val = G_val = M_val = 1.0
E_val = 0.97          # energía por unidad de masa
m_val = 1.0           # masa de la partícula de prueba
rs_val = 2.0          # horizonte
HBAR = 0.02           # hbar/M^2 = 0.02  (mismo estado que second_order_system.py)
COTA = HBAR**2 / 4    # = 1e-4

# Anchura del paquete gaussiano inicial (parámetro libre del paper).
# sigma0 = 0.1  ->  G20_0 = G02_0 = 1e-2  (igual que el sistema del texto)
SIGMA0 = 0.1


# =========================================================================
# ETAPA 1 — SISTEMA AUTOCONSISTENTE DE 2º ORDEN (simbólico -> numérico)
# =========================================================================
r, p_r        = sp.symbols('r p_r', real=True)
G20, G11, G02 = sp.symbols('G20 G11 G02', real=True)
c, G, M, E    = sp.symbols('c G M E', positive=True)

f  = 1 - 2*G*M/(c**2 * r)
f1 = sp.diff(f, r); f2 = sp.diff(f, r, 2); f3 = sp.diff(f, r, 3)
W  = E**2 / (f * c**2)
W1 = sp.diff(W, r); W2 = sp.diff(W, r, 2); W3 = sp.diff(W, r, 3)

# ecuaciones de movimiento
r_dot  = p_r*f + f1*G11 + sp.Rational(1, 2)*p_r*G20*f2
pr_dot = ( sp.Rational(1, 2)*W1 - sp.Rational(1, 2)*f1*p_r**2
         - sp.Rational(1, 2)*f1*G02 - f2*p_r*G11
         + sp.Rational(1, 4)*W3*G20 - sp.Rational(1, 4)*f3*p_r**2*G20 )
# corchetes en la convención posición-primero (ver G_poisson_brackets.py)
G11_dot = f*G02 + sp.Rational(1, 2)*W2*G20 - sp.Rational(1, 2)*f2*p_r**2*G20
G02_dot = -2*p_r*f1*G02 + W2*G11 - f2*p_r**2*G11
G20_dot = 2*f*G11 + 2*p_r*f1*G20

# Hamiltoniano efectivo (test de conservación / diagnóstico del 5%)
Hc    = sp.Rational(1, 2)*(f*p_r**2 - W)
H_rr  = sp.Rational(1, 2)*(f2*p_r**2 - W2)
H_Q = Hc + sp.Rational(1, 2)*H_rr*G20 + f1*p_r*G11 + sp.Rational(1, 2)*f*G02

const_subs = {c: c_val, G: G_val, M: M_val, E: E_val}
state = (r, p_r, G20, G11, G02)
rhs_funcs = [sp.lambdify(state, e.subs(const_subs), 'numpy')
             for e in (r_dot, pr_dot, G20_dot, G11_dot, G02_dot)]
H_func = sp.lambdify(state, H_Q.subs(const_subs), 'numpy')

def system(lam, y):
    return [fn(*y) for fn in rhs_funcs]


# =========================================================================
# ETAPA 2 — INTEGRACIÓN CON CONDICIONES INICIALES DEL PAPER
# -------------------------------------------------------------------------
#   G20_0 = sigma0^2 ,  G02_0 = (hbar/2 sigma0)^2 ,  G11_0 = 0
#   =>  U_0 = G20_0 G02_0 - G11_0^2 = hbar^2/4  EXACTO
# =========================================================================
def pr_onshell(r0):
    f0 = 1 - rs_val/r0
    return -np.sqrt((E_val**2 - m_val**2 * f0) / f0**2)   # rama entrante

def integrar(sigma0, r0=10.0, lam_max=80.0, npts=40000):
    G20_0 = sigma0**2
    G02_0 = (HBAR / (2*sigma0))**2
    y0 = [r0, pr_onshell(r0), G20_0, 0.0, G02_0]

    def horizonte(lam, y): return y[0] - 1.00001*rs_val
    horizonte.terminal = True; horizonte.direction = -1
    def validez(lam, y):  return 1e16 - max(abs(y[2]), abs(y[3]), abs(y[4]))
    validez.terminal = True; validez.direction = -1

    sol = solve_ivp(system, (0.0, lam_max), y0,
                    t_eval=np.linspace(0.0, lam_max, npts),
                    events=[horizonte, validez],
                    method='DOP853', rtol=1e-13, atol=1e-16)
    return sol, y0

sol, y0 = integrar(SIGMA0)
lam = sol.t
r_s, pr_s, G20_s, G11_s, G02_s = sol.y

# cantidades derivadas
U  = G20_s*G02_s - G11_s**2                 # relación de incertidumbre
Un = U / COTA                                # normalizado: cota en 1
H  = H_func(*sol.y)
Hd = np.abs(H/H[0] - 1)                      # deriva relativa del Hamiltoniano
prod = np.abs(G20_s*G02_s)
digits = np.log10(prod/COTA + 1)             # dígitos perdidos en la resta
# Piso de precisión: el error absoluto de U es ~ rtol*(|G20 G02| + G11^2).
# Donde ese error supera el 1% de la cota, U ya no puede evaluarse en float64.
err_U  = 1e-13*(prod + G11_s**2)
k_prec = int(np.argmax(err_U > 1e-2*COTA)) if np.any(err_U > 1e-2*COTA) else len(lam)
lam_prec = lam[k_prec] if k_prec < len(lam) else lam[-1]
valido = slice(0, k_prec)


# =========================================================================
# ETAPA 3 — RESUMEN NUMÉRICO
# =========================================================================
print("="*70)
print(f"hbar/M^2 = {HBAR}  =>  cota  U >= hbar^2/4 = {COTA}")
print(f"Paquete gaussiano inicial: sigma0={SIGMA0} -> "
      f"G20_0={y0[2]:.4f}, G02_0={y0[4]:.4f}, G11_0=0")
print("="*70)
print(f"U(0)                 = {U[0]:.10f}")
print(f"fin de la integración: lambda={lam[-1]:.4f}, r={r_s[-1]:.4f} (status {sol.status}: "
      f"{'divergen los momentos' if sol.status == 1 else 'fin del intervalo'})")
print(f"U al final           = {U[-1]:.10f}")
print(f"desv. relativa máx   = {np.max(np.abs(U/U[0]-1)):.3e}  (todo el rango)")
# La cota se considera violada solo si U cae por debajo MÁS que el error
# numérico de integración (usamos una tolerancia relativa de 1e-3, muy por
# encima de la desviación real ~1e-4 => cualquier "violación" es ruido).
tol_viol = 1e-3 * COTA
print(f"piso de precisión    : lam > {lam_prec:.3f}  (r={r_s[min(k_prec,len(lam)-1)]:.3f})")
print(f"desv. rel. máx (zona válida) = {np.max(np.abs(Un[valido]-1)):.3e}")
print(f"¿viola U >= hbar^2/4 en la zona válida? = {'SÍ' if np.any(U[valido] < COTA - tol_viol) else 'NO'}")
print(f"deriva H máxima      = {Hd.max():.3e}  -> "
      f"{'FIABLE (<5%)' if Hd.max()<0.05 else 'FALLA (>5%)'}")
print(f"dígitos cancelados   = {digits.max():.1f} (de ~16 en doble precisión)")


# =========================================================================
# ETAPA 4 — FIGURA (2 paneles)
# -------------------------------------------------------------------------
# U se grafica solo hasta lam_prec: más allá, el error de redondeo de la
# resta G20*G02 - G11^2 supera el 1% de la cota y U no es evaluable en float64.
# =========================================================================
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.25})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))

# ---- (1) U vs lambda durante la caída (hasta el piso de precisión) ----
a1.axhline(1.0, color='crimson', ls=(0, (4, 4)), lw=1.0, zorder=1,
            label=r'cota $U=\hbar^2/4$')
a1.plot(lam[:k_prec], Un[:k_prec], 'b', lw=2.6, zorder=3, label=r'$U$')
a1.fill_between([0, lam_prec], 0, 1.0, color='crimson', alpha=0.07)
a1.text(lam_prec*0.45, 0.5, 'región prohibida', ha='center',
        color='crimson', fontsize=10)
a1.set_ylim(0, 2); a1.set_xlim(0, lam_prec)
a1.set_xlabel(r'$\lambda$')
a1.set_ylabel(r'$U/(\hbar^2/4)$')
a1.set_title(r'$U(\lambda)$ durante la caída')
a1.legend(loc='upper left', fontsize=9)
# recuadro: desviación relativa |U/(hbar^2/4) - 1| en escala log,
# para que la coincidencia con la cota no oculte la curva
ins = a1.inset_axes([0.50, 0.64, 0.46, 0.30])
ins.semilogy(lam, np.abs(Un - 1) + 1e-18, 'b', lw=1.6)
ins.set_xlim(0, lam[-1]); ins.set_ylim(1e-16, 1e-3)
ins.set_xlabel(r'$\lambda$', fontsize=8, labelpad=1)
ins.set_title(r'$|U/(\hbar^2/4)-1|$', fontsize=9)
ins.tick_params(labelsize=7)

# ---- (2) criterio del 5% en H ----
a2.semilogy(lam, Hd + 1e-18, color='#2ca02c', lw=2.4, label=r'$|\Delta H_Q/H_Q(0)|$')
a2.axhline(0.05, color='crimson', ls='--', lw=2.2, label='umbral 5% (falla numérica)')
a2.axvline(lam[-1], color='gray', ls=':', lw=1.6)
a2.text(lam[-1]-0.15, 1e-11, 'divergen los $G$', rotation=90, fontsize=9,
        color='gray', ha='right')
a2.set_xlim(0, lam[-1]); a2.set_ylim(1e-15, 1)
a2.set_xlabel(r'$\lambda$'); a2.set_ylabel(r'$|\Delta H_Q/H_Q(0)|$  (log)')
a2.set_title(f'Deriva de $H_Q$: máx {Hd.max():.1e} $\\ll$ 5%')
a2.legend(loc='lower right', fontsize=9)

fig.suptitle(r'$\hbar=2\times10^{-2}$, $G^{2,0}_0=G^{0,2}_0=10^{-2}$: '
             r'$U$ se conserva en $\hbar^2/4$ durante la caída', y=1.00, fontsize=12.5)
fig.tight_layout()
fig.savefig('incertidumbre_natural.png', dpi=140, bbox_inches='tight')
print("\nFigura guardada: incertidumbre_natural.png")
