"""
=============================================================================
 SCRIPT MAESTRO COMPLETO — Sistema semiclásico de momentos de 2º orden CON
 RETROALIMENTACIÓN (autoconsistente) en caída radial de Schwarzschild.
=============================================================================

 Reúne todo el análisis de la caída radial descrita por una expansión en
 momentos truncada a segundo orden, en su versión autoconsistente (los momentos
 G retroalimentan a la posición r y al momento p_r).

 Produce:
   FIG 1 : momento total |p_r| y momentos |G| en función de r (rama de caída)
   FIG 2 : ruptura de la aproximación  (r* donde la corrección = clásico)
   FIG 3 : boomerang  (r(λ) rebota; descomposición de dr/dλ)
   FIG 4 : desacople momento-velocidad + comparación con la geodésica clásica
   FIG 5 : el rebote como PLIEGUE al graficar la trayectoria completa vs r
   TABLA : acercamiento máximo a 2m (r_min) y ruptura (r*) según el paquete G0

 Todos los números que se citan en el informe se imprimen por pantalla.

 Unidades geometrizadas c = G = 1  =>  horizonte en r_s = 2m = 2.
=============================================================================
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# ETAPA 0 — CONSTANTES FÍSICAS
# =========================================================================
c_val = 1.0
G_val = 1.0
M_val = 1.0
E_val = 0.97          # energía por unidad de masa
m_val = 1.0           # masa de la partícula de prueba
rs_val = 2.0 * G_val * M_val / c_val**2     # horizonte en r = 2


# =========================================================================
# ETAPA 1 — SÍMBOLOS Y MÉTRICA f(r)
# =========================================================================
r, p_r        = sp.symbols('r p_r', real=True)
G20, G11, G02 = sp.symbols('G20 G11 G02', real=True)
c, G, M, E    = sp.symbols('c G M E', positive=True)

f = 1 - 2*G*M/(c**2 * r)


# =========================================================================
# ETAPA 2 — LADO DERECHO DEL SISTEMA (autoconsistente)
# =========================================================================
f1 = sp.diff(f, r)
f2 = sp.diff(f, r, 2)
f3 = sp.diff(f, r, 3)

W  = E**2 / (f * c**2)
W1 = sp.diff(W, r)
W2 = sp.diff(W, r, 2)
W3 = sp.diff(W, r, 3)

# ecuación de la posición  (los G entran aquí -> retroalimentación)
r_dot = p_r*f + f1*G11 + sp.Rational(1, 2)*p_r*G20*f2

# ecuación del momento (completa)
pr_dot = ( sp.Rational(1, 2)*W1
         - sp.Rational(1, 2)*f1*p_r**2
         - sp.Rational(1, 2)*f1*G02
         - f2*p_r*G11
         + sp.Rational(1, 4)*W3*G20
         - sp.Rational(1, 4)*f3*p_r**2*G20 )

# ecuaciones de los momentos de 2º orden
G11_dot = -f*G02 - sp.Rational(1, 2)*W2*G20 + sp.Rational(1, 2)*f2*p_r**2*G20
G02_dot = 2*p_r*f1*G02 - W2*G11 + f2*p_r**2*G11
G20_dot = -2*f*G11 - 2*p_r*f1*G20

# Hamiltoniano efectivo (test de conservación)
Hc   = sp.Rational(1, 2)*(f*p_r**2 - W)
H_rr = sp.Rational(1, 2)*(f2*p_r**2 - W2)
H_rp = f1*p_r
H_pp = f
H_eff = Hc + sp.Rational(1, 2)*H_rr*G20 + H_rp*G11 + sp.Rational(1, 2)*H_pp*G02


# =========================================================================
# ETAPA 3 — SUSTITUCIÓN Y LAMBDIFY
# =========================================================================
const_subs = {c: c_val, G: G_val, M: M_val, E: E_val}
state = (r, p_r, G20, G11, G02)

rhs_funcs = [sp.lambdify(state, e.subs(const_subs), 'numpy')
             for e in (r_dot, pr_dot, G20_dot, G11_dot, G02_dot)]
H_func = sp.lambdify(state, H_eff.subs(const_subs), 'numpy')

# Piezas de dp_r/dλ:  término clásico  vs  aportación de cada G
prclas_f = sp.lambdify(state, (sp.Rational(1,2)*W1 - sp.Rational(1,2)*f1*p_r**2).subs(const_subs), 'numpy')
aG20_f   = sp.lambdify(state, (sp.Rational(1,4)*W3*G20 - sp.Rational(1,4)*f3*p_r**2*G20).subs(const_subs), 'numpy')
aG11_f   = sp.lambdify(state, (-f2*p_r*G11).subs(const_subs), 'numpy')
aG02_f   = sp.lambdify(state, (-sp.Rational(1,2)*f1*G02).subs(const_subs), 'numpy')

# Piezas de dr/dλ:  término clásico  vs  corrección de los G
rclas_f = sp.lambdify(state, (p_r*f).subs(const_subs), 'numpy')
rcorr_f = sp.lambdify(state, (f1*G11 + sp.Rational(1,2)*p_r*G20*f2).subs(const_subs), 'numpy')

def system(lam, y):
    return [fn(*y) for fn in rhs_funcs]


# =========================================================================
# ETAPA 4 — INTEGRADOR REUTILIZABLE
# -------------------------------------------------------------------------
# Integra el sistema para un paquete inicial G0 dado y devuelve la solución
# junto con los diagnósticos clave (r*, r_min, etc.).
# =========================================================================
def pr_onshell(r0):
    f0 = 1 - rs_val/r0
    arg = (E_val**2 - m_val**2 * c_val**4 * f0) / (f0**2 * c_val**2)
    return -np.sqrt(arg) if arg > 0 else 0.0        # rama entrante

def integrar(G0, r0=10.0, lam_max=80.0, npts=20000,
             rtol=1e-12, atol=1e-15, mom_max=1e14):
    """Integra el sistema autoconsistente con G20(0)=G02(0)=G0, G11(0)=0."""
    y0 = [r0, pr_onshell(r0), G0, 0.0, G0]

    def horizonte(lam, y):
        return y[0] - 1.00001*rs_val
    horizonte.terminal = True
    horizonte.direction = -1

    def validez(lam, y):
        return mom_max - max(abs(y[2]), abs(y[3]), abs(y[4]))
    validez.terminal = True
    validez.direction = -1

    sol = solve_ivp(system, (0.0, lam_max), y0,
                    t_eval=np.linspace(0.0, lam_max, npts),
                    events=[horizonte, validez],
                    method='DOP853', rtol=rtol, atol=atol)

    lam = sol.t
    r_s, pr_s, G20_s, G11_s, G02_s = sol.y

    # diagnósticos
    clas = np.abs(prclas_f(*sol.y))
    aG20 = np.abs(aG20_f(*sol.y)); aG11 = np.abs(aG11_f(*sol.y)); aG02 = np.abs(aG02_f(*sol.y))
    corr_max = np.maximum.reduce([aG20, aG11, aG02])
    ratio = corr_max / clas
    idx = int(np.argmax(ratio >= 1.0))
    if ratio[idx] >= 1.0 and idx > 0:      # interpola r*
        y1, y2 = ratio[idx-1]-1, ratio[idx]-1
        r_star = r_s[idx-1] - y1*(r_s[idx]-r_s[idx-1])/(y2-y1)
        culpable = ['G20','G11','G02'][int(np.argmax([aG20[idx],aG11[idx],aG02[idx]]))]
    else:
        r_star, culpable = float('nan'), None
    imin = int(np.argmin(r_s))

    return dict(sol=sol, lam=lam, r=r_s, pr=pr_s, G20=G20_s, G11=G11_s, G02=G02_s,
                clas=clas, aG20=aG20, aG11=aG11, aG02=aG02, ratio=ratio,
                r_star=r_star, culpable=culpable, imin=imin,
                r_min=r_s[imin], lam_min=lam[imin])


# =========================================================================
# ETAPA 5 — CASO PRINCIPAL (G0 = 1e-2) Y NÚMEROS CLAVE
# =========================================================================
G0_PRINCIPAL = 1e-2
D = integrar(G0_PRINCIPAL)
r_s, pr_s, G20_s, G11_s, G02_s = D['r'], D['pr'], D['G20'], D['G11'], D['G02']
lam, imin = D['lam'], D['imin']
r_star, r_min, lam_min = D['r_star'], D['r_min'], D['lam_min']

H_series = H_func(r_s, pr_s, G20_s, G11_s, G02_s)
drift = np.max(np.abs(H_series - H_series[0]))

print("================ NÚMEROS PARA EL INFORME (G0=1e-2) ================")
print(f"Condiciones iniciales: r0=10, p_r0={pr_onshell(10.0):.5f}, G20=G02={G0_PRINCIPAL}, G11=0, E={E_val}")
print(f"Conservación H_eff:  H0={H_series[0]:.6e},  deriva máx={drift:.2e}")
print(f"Ruptura:  r* = {r_star:.4f}  (la dispara {D['culpable']})")
print(f"   momentos cerca de r*: G20={G20_s[D['sol'].t.searchsorted(0)] if False else G20_s[np.argmin(np.abs(r_s-r_star))]:.3e}")
print(f"Boomerang: r_min = {r_min:.4f} en lambda={lam_min:.3f}")
rdc = rclas_f(*[a[imin] for a in D['sol'].y]); rdk = rcorr_f(*[a[imin] for a in D['sol'].y])
print(f"   dr/dλ en el retorno: clásico={rdc:+.4f}, corrección={rdk:+.4f}, suma={rdc+rdk:+.4f}")
print(f"   p_r en el retorno = {pr_s[imin]:+.3f}  (NO se anula: no es retorno físico)")


# =========================================================================
# ETAPA 6 — COMPARACIÓN CON LA GEODÉSICA CLÁSICA PURA (momentos = 0)
# =========================================================================
def clas_sys(l, y):
    rr, pp = y
    ff = 1 - rs_val/rr
    fp = rs_val/rr**2
    Wp = -E_val**2*fp/ff**2
    return [pp*ff, 0.5*Wp - 0.5*fp*pp**2]
def hz_clas(l, y):
    return y[0] - 1.000001*rs_val
hz_clas.terminal = True
hz_clas.direction = -1
sc = solve_ivp(clas_sys, (0, 60), [10.0, pr_onshell(10.0)], events=[hz_clas],
               method='DOP853', rtol=1e-11, atol=1e-13, max_step=0.01)
r_apo = rs_val/(1 - E_val**2)
print(f"Clásico puro: llega a r={sc.y[0,-1]:.5f} en lambda={sc.t[-1]:.3f} (dr/dλ->{sc.y[1,-1]*(1-rs_val/sc.y[0,-1]):.3f})")
print(f"   apoapsis clásica f=E²: r={r_apo:.2f} (no hay retorno interior clásico)")


# =========================================================================
# ETAPA 7 — BARRIDO: acercamiento a 2m según el paquete inicial G0
# =========================================================================
print("\n== Acercamiento a 2m según G0 ==")
print(f"{'G0':>8} {'r_min':>10} {'r*':>10}")
for g0 in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
    d = integrar(g0)
    print(f"{g0:>8.0e} {d['r_min']:>10.4f} {d['r_star']:>10.4f}")


# =========================================================================
# FIGURA 1 — momento total |p_r| y momentos |G| vs r (rama de caída)
# =========================================================================
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                     'axes.grid': True, 'grid.alpha': 0.25})
COL = {'G20': '#1f77b4', 'G11': '#ff7f0e', 'G02': '#2ca02c', 'pr': '#111111'}
sl = slice(0, imin+1)     # rama de caída (r monótona)

fig1, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
a1.semilogy(r_s[sl], np.abs(pr_s[sl]), color=COL['pr'], lw=2.3)
a1.axvline(r_star, color='crimson', lw=1.6)
a1.text(r_star-0.1, a1.get_ylim()[1]*0.3, f'$r_*={r_star:.2f}$', color='crimson', ha='right', fontsize=10)
a1.axvline(rs_val, ls=':', color='k', lw=1.0)
a1.set_xlim(10.0, r_min-0.05); a1.set_xlabel(r'$r$'); a1.set_ylabel(r'$|p_r|$  (log)')
a1.set_title('Momento total durante la caída')
for nm, ser in [('G20',G20_s),('G11',G11_s),('G02',G02_s)]:
    a2.semilogy(r_s[sl], np.abs(ser[sl]), color=COL[nm], lw=2, label=fr'$|G^{{{nm[1]},{nm[2]}}}|$')
a2.axvline(r_star, color='crimson', lw=1.6)
a2.text(r_star-0.1, a2.get_ylim()[1]*0.3, f'$r_*={r_star:.2f}$', color='crimson', ha='right', fontsize=10)
a2.axvline(rs_val, ls=':', color='k', lw=1.0)
a2.set_xlim(10.0, r_min-0.05); a2.set_xlabel(r'$r$'); a2.set_ylabel('momentos  (log)')
a2.set_title('Momentos de 2º orden durante la caída'); a2.legend(loc='upper right')
fig1.suptitle('Sistema autoconsistente: momento total y momentos $G$ vs $r$', y=1.00, fontsize=13)
fig1.tight_layout(); fig1.savefig('informe_fig1_momentos.png', dpi=140, bbox_inches='tight')


# =========================================================================
# FIGURA 2 — ruptura: clásico vs aportación de cada G en dp_r/dλ
# =========================================================================
clas, aG20, aG11, aG02, ratio = D['clas'], D['aG20'], D['aG11'], D['aG02'], D['ratio']
fig2, (b1, b2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
b1.semilogy(r_s[sl], clas[sl], color='#777', lw=2.3, ls='--', label='clásico')
b1.semilogy(r_s[sl], aG20[sl], color=COL['G20'], lw=1.8, label=r'aport. $G^{2,0}$')
b1.semilogy(r_s[sl], aG11[sl], color=COL['G11'], lw=1.8, label=r'aport. $G^{1,1}$')
b1.semilogy(r_s[sl], aG02[sl], color=COL['G02'], lw=1.8, label=r'aport. $G^{0,2}$')
b1.axvline(r_star, color='crimson', lw=1.8)
b1.text(r_star-0.1, b1.get_ylim()[1]*0.3, f'$r_*={r_star:.2f}$', color='crimson', ha='right', fontsize=10)
b1.set_xlim(10.0, r_min-0.05); b1.set_xlabel(r'$r$'); b1.set_ylabel(r'aportación a $dp_r/d\lambda$  (log)')
b1.set_title('Cada corrección alcanza al término clásico'); b1.legend(loc='lower left', fontsize=9)
b2.semilogy(r_s[sl], ratio[sl], color='#6a3d9a', lw=2.3)
b2.axhline(1.0, color='crimson', ls='--', lw=1.4)
b2.plot([r_star], [1.0], 'o', color='crimson', ms=8, zorder=5)
b2.annotate(f'$r_*={r_star:.2f}$', xy=(r_star, 1.0), xytext=(r_star+1.0, 8),
            color='crimson', fontsize=11, arrowprops=dict(arrowstyle='->', color='crimson'))
b2.set_xlim(10.0, r_min-0.05); b2.set_xlabel(r'$r$')
b2.set_ylabel(r'$\max_i|\mathrm{aport.}\,G_i|\,/\,|\mathrm{clásico}|$')
b2.set_title('Razón corrección/clásico: cruza 1 en la ruptura')
fig2.suptitle('¿Dónde muere la aproximación de 2º orden?', y=1.00, fontsize=13)
fig2.tight_layout(); fig2.savefig('informe_fig2_ruptura.png', dpi=140, bbox_inches='tight')


# =========================================================================
# FIGURA 3 — boomerang: r(λ) rebota; descomposición de dr/dλ
# =========================================================================
rdot_clas = rclas_f(r_s, pr_s, G20_s, G11_s, G02_s)
rdot_corr = rcorr_f(r_s, pr_s, G20_s, G11_s, G02_s)
rdot_full = rdot_clas + rdot_corr
fig3, (e1, e2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
e1.plot(lam, r_s, 'k', lw=2.4)
e1.plot(lam_min, r_min, 'o', color='crimson', ms=9, zorder=5)
e1.axhline(r_star, color='crimson', ls='--', lw=1.3)
e1.text(lam[0], r_star+0.15, f'$r_*={r_star:.2f}$ (muere la aprox.)', color='crimson', fontsize=9)
e1.axhline(rs_val, color='gray', ls=':', lw=1.3)
e1.text(lam[0], rs_val+0.15, 'horizonte $2m$', color='gray', fontsize=9)
e1.annotate(f'retorno\n$r_{{min}}={r_min:.2f}$', xy=(lam_min, r_min),
            xytext=(lam_min-6, 5.3), color='crimson', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='crimson'))
e1.set_xlabel(r'$\lambda$'); e1.set_ylabel(r'$r$'); e1.set_title(r'$r(\lambda)$ baja, se frena y rebota')
e2.plot(lam, rdot_clas, lw=2, label=r'clásico  $p_r f$')
e2.plot(lam, rdot_corr, lw=2, label=r"corrección  $f'G^{1,1}+\frac{1}{2}p_r G^{2,0}f''$")
e2.plot(lam, rdot_full, 'k', lw=2.4, label=r'suma $=dr/d\lambda$')
e2.axhline(0, color='crimson', ls='--', lw=1.2)
e2.plot(lam_min, 0, 'o', color='crimson', ms=8, zorder=5)
e2.set_ylim(-15, 15); e2.set_xlabel(r'$\lambda$'); e2.set_ylabel(r'$dr/d\lambda$')
e2.set_title('La corrección cancela al clásico: $dr/d\\lambda=0$'); e2.legend(fontsize=9, loc='upper left')
fig3.suptitle('El boomerang: artefacto de la retroalimentación descontrolada', y=1.00, fontsize=13)
fig3.tight_layout(); fig3.savefig('informe_fig3_boomerang.png', dpi=140, bbox_inches='tight')


# =========================================================================
# FIGURA 4 — desacople momento-velocidad + comparación clásica
# =========================================================================
fig4, (A, Bx) = plt.subplots(1, 2, figsize=(13.5, 5.3))
A.plot(lam, r_s, 'b', lw=2.4, label='autoconsistente (rebota)')
A.plot(sc.t, sc.y[0], 'g--', lw=2.2, label='clásico puro (llega a $2m$)')
A.plot(lam_min, r_min, 'o', color='crimson', ms=9, zorder=5)
A.axhline(rs_val, color='k', ls=':', lw=1.3); A.text(0.3, 2.15, 'horizonte $2m$', fontsize=9)
A.axhline(r_star, color='crimson', ls=':', lw=1.1)
A.text(0.3, r_star+0.15, f'$r_*={r_star:.2f}$', color='crimson', fontsize=9)
A.set_xlabel(r'$\lambda$'); A.set_ylabel(r'$r$'); A.set_ylim(1.5, 12)
A.set_title('El clásico alcanza $2m$; el truncado rebota antes'); A.legend(loc='upper right', fontsize=9)
sl2 = slice(max(0, imin-1500), imin+1)
Bx.plot(lam[sl2], pr_s[sl2], color='#2ca02c', lw=2.4)
Bxt = Bx.twinx()
Bxt.plot(lam[sl2], rdot_full[sl2], color='#111', lw=2.4)
Bxt.axhline(0, color='crimson', ls='--', lw=1.3)
Bx.plot(lam_min, pr_s[imin], 'o', color='#2ca02c', ms=9, zorder=5)
Bxt.plot(lam_min, rdot_full[imin], 'o', color='#111', ms=9, zorder=5)
Bx.annotate(f'$p_r={pr_s[imin]:.1f}$\n(sigue hacia adentro)', xy=(lam_min, pr_s[imin]),
            xytext=(lam_min-5, pr_s[imin]+3), color='#2ca02c', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#2ca02c'))
Bxt.annotate(r'$dr/d\lambda\to 0$', xy=(lam_min, 0), xytext=(lam_min-6, 4),
             color='#111', fontsize=10, arrowprops=dict(arrowstyle='->', color='#111'))
Bx.set_xlabel(r'$\lambda$'); Bx.set_ylabel(r'$p_r$', color='#2ca02c')
Bxt.set_ylabel(r'$dr/d\lambda$', color='#111'); Bxt.set_ylim(-8, 8)
Bx.set_title(r'En el rebote: momento $\neq 0$ pero velocidad $=0$')
fig4.suptitle('El rebote no es un retorno físico: momento y velocidad se desacoplan', y=1.00, fontsize=12.5)
fig4.tight_layout(); fig4.savefig('informe_fig4_desacople.png', dpi=140, bbox_inches='tight')


# =========================================================================
# FIGURA 5 — el rebote como PLIEGUE (trayectoria completa vs r)
# -------------------------------------------------------------------------
# Al no recortar en r_min, la rama de regreso (punteada) repasa los mismos
# radios que la de caída (sólida) y se superpone: ese pliegue ES el rebote.
# Se usa G0=1e-2 porque deja una rama de regreso larga y bien visible
# (con G0 mucho menor el rebote colapsa en un pico y no se aprecia).
# =========================================================================
caida = slice(0, imin+1); regreso = slice(imin, None)
fig5, (g1, g2) = plt.subplots(1, 2, figsize=(13.5, 5.3))
g1.semilogy(r_s[caida], np.abs(pr_s[caida]), color=COL['pr'], lw=2.3, label='caída')
g1.semilogy(r_s[regreso], np.abs(pr_s[regreso]), color=COL['pr'], lw=2.0, ls=':', label='regreso (rebote)')
g1.plot(r_min, np.abs(pr_s[imin]), 'o', color='crimson', ms=9, zorder=5)
g1.annotate(f'rebote\n$r_{{min}}={r_min:.2f}$', xy=(r_min, np.abs(pr_s[imin])),
            xytext=(r_min+1.3, np.abs(pr_s[imin])*0.3), color='crimson', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='crimson'))
g1.axvline(rs_val, ls='--', color='r', lw=1.3); g1.text(rs_val+0.03, g1.get_ylim()[0]*3, '$2m$', color='r', fontsize=10)
g1.set_xlim(10.0, rs_val-0.02); g1.set_xlabel(r'$r$'); g1.set_ylabel(r'$|p_r|$  (log)')
g1.set_title('Momento total: caída y regreso se pliegan'); g1.legend(loc='upper left', fontsize=9)
for nm, ser in [('G20',G20_s),('G11',G11_s),('G02',G02_s)]:
    g2.semilogy(r_s[caida], np.abs(ser[caida]), color=COL[nm], lw=2, label=fr'$|G^{{{nm[1]},{nm[2]}}}|$')
    g2.semilogy(r_s[regreso], np.abs(ser[regreso]), color=COL[nm], lw=1.8, ls=':')
g2.axvline(r_min, color='crimson', lw=1.2, alpha=.5)
g2.annotate(f'rebote\n$r_{{min}}={r_min:.2f}$', xy=(r_min, 1e-3), xytext=(r_min+1.5, 1e-4),
            color='crimson', fontsize=10, arrowprops=dict(arrowstyle='->', color='crimson'))
g2.axvline(rs_val, ls='--', color='r', lw=1.3); g2.text(rs_val+0.03, g2.get_ylim()[0]*3, '$2m$', color='r', fontsize=10)
g2.set_xlim(10.0, rs_val-0.02); g2.set_xlabel(r'$r$'); g2.set_ylabel('momentos  (log)')
g2.set_title('Momentos $G$: mismo pliegue en $r_{min}$'); g2.legend(loc='upper left', fontsize=9)
fig5.suptitle('El rebote visto contra $r$: un pliegue de la trayectoria en $r_{min}$', y=1.00, fontsize=12.5)
fig5.tight_layout(); fig5.savefig('informe_fig5_rebote.png', dpi=140, bbox_inches='tight')

print("\nFiguras guardadas: informe_fig1..5")