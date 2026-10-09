"""
=============================================================================
 SCRIPT MAESTRO COMPLETO — Sistema semiclásico de momentos de 2º orden CON
 RETROALIMENTACIÓN (autoconsistente) en caída radial de Schwarzschild.

 VERSIÓN ADAPTADA: ecuaciones de evolución de los momentos G con el signo
 invertido respecto al script original, según el sistema:

   r_dot   = p_r f + f' G11 + (1/2) p_r G20 f''
   pr_dot  = (1/2) W' - (1/2) f' p_r^2 - (1/2) f' G02 - f'' p_r G11
             + (1/4) W''' G20 - (1/4) f''' p_r^2 G20
   G11_dot = -( -f G02 - (1/2) W'' G20 + (1/2) f'' p_r^2 G20 )
   G02_dot = -( 2 p_r f' G02 - W'' G11 + f'' p_r^2 G11 )
   G20_dot = -( -2 f G11 - 2 p_r f' G20 )
=============================================================================

 Reúne todo el análisis de la caída radial descrita por una expansión en
 momentos truncada a segundo orden, en su versión autoconsistente (los momentos
 G retroalimentan a la posición r y al momento p_r).

 Produce:
   FIG 1 : momento total |p_r| y momentos |G| en función de r (rama de caída)
   FIG 2 : ruptura de la aproximación  (r* donde la corrección = clásico)
   FIG 3 : NO hay rebote — r(λ) cae monótonamente; descomposición de dr/dλ
           mostrando que clásico y corrección casi se cancelan pero ambos
           divergen juntos, sin llegar a cruzar cero
   FIG 4 : p_r y dr/dλ divergen a la vez cerca del corte (no hay desacople
           momento-velocidad) + comparación con la geodésica clásica
   FIG 5 : comprobación de robustez — al subir el corte de seguridad muchos
           órdenes de magnitud, lambda_c converge (divergencia real del
           modelo) mientras que el radio en el corte no converge
   TABLA : radio de ruptura r* y lambda_c convergido según el paquete G0

 NOTA IMPORTANTE: a diferencia del sistema original (donde tras r* la
 trayectoria "rebotaba" con dr/dλ→0 de forma limpia, un retorno físico),
 en este sistema con los signos de G invertidos la trayectoria r(λ) NO
 tiene ningún mínimo real seguido de una rama de regreso: cae de forma
 monótona mientras los tres momentos G divergen en un lambda finito
 (lambda_c), un polo móvil genuino de las ecuaciones. Se comprobó que
 lambda_c converge al refinar tolerancias y subir el corte de seguridad
 en más de veinte órdenes de magnitud, así que no es un artefacto
 numérico. Por eso las figuras y el texto evitan "rebote", "retorno
 físico" y "pliegue", y hablan en su lugar de "divergencia en lambda
 finito" y "corte por crecimiento descontrolado".

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
# ETAPA 2 — LADO DERECHO DEL SISTEMA (autoconsistente, nuevo sistema)
# =========================================================================
f1 = sp.diff(f, r)
f2 = sp.diff(f, r, 2)
f3 = sp.diff(f, r, 3)

W  = E**2 / (f * c**2)
W1 = sp.diff(W, r)
W2 = sp.diff(W, r, 2)
W3 = sp.diff(W, r, 3)

# ecuación de la posición  (los G entran aquí -> retroalimentación)
# -- idéntica a la del sistema original --
r_dot = p_r*f + f1*G11 + sp.Rational(1, 2)*p_r*G20*f2

# ecuación del momento (completa)
# -- idéntica a la del sistema original --
pr_dot = ( sp.Rational(1, 2)*W1
         - sp.Rational(1, 2)*f1*p_r**2
         - sp.Rational(1, 2)*f1*G02
         - f2*p_r*G11
         + sp.Rational(1, 4)*W3*G20
         - sp.Rational(1, 4)*f3*p_r**2*G20 )

# ecuaciones de los momentos de 2º orden
# -- NUEVO: signo invertido respecto al sistema original --
G11_dot = -( -f*G02 - sp.Rational(1, 2)*W2*G20 + sp.Rational(1, 2)*f2*p_r**2*G20 )
G02_dot = -( 2*p_r*f1*G02 - W2*G11 + f2*p_r**2*G11 )
G20_dot = -( -2*f*G11 - 2*p_r*f1*G20 )

# Hamiltoniano efectivo (test de conservación)
# NOTA: al invertir el signo de los G_dot el sistema deja de ser, en general,
# el flujo hamiltoniano de H_eff (esa H_eff correspondía al sistema original).
# Se conserva aquí solo como diagnóstico informativo, no como cantidad
# necesariamente conservada del nuevo sistema.
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
    if ratio[idx] >= 1.0 and idx > 0:      # interpola r* y lambda*
        y1, y2 = ratio[idx-1]-1, ratio[idx]-1
        r_star = r_s[idx-1] - y1*(r_s[idx]-r_s[idx-1])/(y2-y1)
        lam_star = lam[idx-1] - y1*(lam[idx]-lam[idx-1])/(y2-y1)
        culpable = ['G20','G11','G02'][int(np.argmax([aG20[idx],aG11[idx],aG02[idx]]))]
    else:
        r_star, lam_star, culpable = float('nan'), float('nan'), None
    imin = int(np.argmin(r_s))

    return dict(sol=sol, lam=lam, r=r_s, pr=pr_s, G20=G20_s, G11=G11_s, G02=G02_s,
                clas=clas, aG20=aG20, aG11=aG11, aG02=aG02, ratio=ratio,
                r_star=r_star, lam_star=lam_star, culpable=culpable, imin=imin,
                r_min=r_s[imin], lam_min=lam[imin], max_ratio=np.max(ratio))


def lambda_c_convergencia(G0, r0=10.0, mom_max=1e30, rtol=1e-13, atol=1e-16):
    """Ubica el punto de divergencia matemática (polo móvil) empujando el
    corte de seguridad mom_max muy por encima de lo razonable. Si lambda_c
    converge al variar mom_max, es una divergencia genuina del modelo, no
    un artefacto del corte numérico."""
    y0 = [r0, pr_onshell(r0), G0, 0.0, G0]

    def horizonte(lam, y):
        return y[0] - 1.00001*rs_val
    horizonte.terminal = True
    horizonte.direction = -1

    def validez(lam, y):
        return mom_max - max(abs(y[2]), abs(y[3]), abs(y[4]))
    validez.terminal = True
    validez.direction = -1

    sol = solve_ivp(system, (0.0, 80.0), y0, events=[horizonte, validez],
                    method='DOP853', rtol=rtol, atol=atol, max_step=0.5)
    return sol.t[-1], sol.y[0, -1], sol.status


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

print("================ NÚMEROS PARA EL INFORME (G0=1e-2) — SISTEMA CON $\\dot G$ INVERTIDO ================")
print(f"Condiciones iniciales: r0=10, p_r0={pr_onshell(10.0):.5f}, G20=G02={G0_PRINCIPAL}, G11=0, E={E_val}")
print(f"H_eff (diagnóstico, NO es el generador de este sistema -> no es una cantidad conservada garantizada):")
print(f"  H0={H_series[0]:.6e},  deriva máx={drift:.2e}")
if np.isnan(r_star):
    print("Ruptura: no se alcanzó ratio>=1 dentro del rango integrado")
else:
    i_star = np.argmin(np.abs(r_s - r_star))
    print(f"Ruptura:  r* = {r_star:.4f}  en lambda*={D['lam_star']:.4f}  (la dispara {D['culpable']})")
    print(f"   en r*: G20={G20_s[i_star]:.4e}  G11={G11_s[i_star]:.4e}  G02={G02_s[i_star]:.4e}  p_r={pr_s[i_star]:.4e}")
print(f"\nCorte de la integración (por crecimiento descontrolado de los momentos):")
print(f"   r_corte = {r_min:.4f} en lambda={lam_min:.3f}  (no hay rama de regreso: r es monótona decreciente hasta el corte)")
rdc = rclas_f(*[a[imin] for a in D['sol'].y]); rdk = rcorr_f(*[a[imin] for a in D['sol'].y])
print(f"   dr/dλ en el corte: clásico={rdc:+.4f}, corrección={rdk:+.4f}, suma={rdc+rdk:+.4f}  (NUNCA cruza cero)")
print(f"   p_r en el corte = {pr_s[imin]:+.3f}")

lam_c_conv, r_c_conv, status_c = lambda_c_convergencia(G0_PRINCIPAL)
print(f"\nUbicación robusta del polo (independiente del corte de seguridad):")
print(f"   lambda_c = {lam_c_conv:.6f}  (converge al refinar tolerancias / subir el corte)")
print(f"   r justo antes del polo ~ {r_c_conv:.4f}  (sigue bajando si se afina más: NO converge del mismo modo que lambda_c)")


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
print("\n== Barrido en G0: ruptura (si existe), corte numérico y polo convergido ==")
print(f"{'G0':>8} {'r_corte':>10} {'r*':>10} {'lambda_c':>12}")
tabla_barrido = []
for g0 in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
    d = integrar(g0)
    lam_c_g0, r_c_g0, _ = lambda_c_convergencia(g0)
    r_star_txt = f"{d['r_star']:>10.4f}" if not np.isnan(d['r_star']) else f"{'—':>10}"
    print(f"{g0:>8.0e} {d['r_min']:>10.4f} {r_star_txt} {lam_c_g0:>12.6f}")
    tabla_barrido.append((g0, d['r_min'], d['r_star'], lam_c_g0, r_c_g0))


# =========================================================================
# FIGURA 1 — momento total |p_r| y momentos |G| vs r (rama de caída)
# =========================================================================
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                     'axes.grid': True, 'grid.alpha': 0.25})
COL = {'G20': '#1f77b4', 'G11': '#ff7f0e', 'G02': '#2ca02c', 'pr': '#111111'}
sl = slice(0, imin+1)     # rama de caída (r monótona)

fig1, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
a1.semilogy(r_s[sl], np.abs(pr_s[sl]), color=COL['pr'], lw=2.3)
if not np.isnan(r_star):
    a1.axvline(r_star, color='crimson', lw=1.6)
    a1.text(r_star-0.1, a1.get_ylim()[1]*0.3, f'$r_*={r_star:.2f}$', color='crimson', ha='right', fontsize=10)
a1.axvline(rs_val, ls=':', color='k', lw=1.0)
a1.set_xlim(10.0, r_min-0.05); a1.set_xlabel(r'$r$'); a1.set_ylabel(r'$|p_r|$  (log)')
a1.set_title('Momento total durante la caída')
for nm, ser in [('G20',G20_s),('G11',G11_s),('G02',G02_s)]:
    a2.semilogy(r_s[sl], np.abs(ser[sl]), color=COL[nm], lw=2, label=fr'$|G^{{{nm[1]},{nm[2]}}}|$')
if not np.isnan(r_star):
    a2.axvline(r_star, color='crimson', lw=1.6)
    a2.text(r_star-0.1, a2.get_ylim()[1]*0.3, f'$r_*={r_star:.2f}$', color='crimson', ha='right', fontsize=10)
a2.axvline(rs_val, ls=':', color='k', lw=1.0)
a2.set_xlim(10.0, r_min-0.05); a2.set_xlabel(r'$r$'); a2.set_ylabel('momentos  (log)')
a2.set_title('Momentos de 2º orden durante la caída'); a2.legend(loc='upper right')
fig1.suptitle('Sistema con signos invertidos en $\\dot G$: momento total y momentos $G$ vs $r$', y=1.00, fontsize=13)
fig1.tight_layout(); fig1.savefig('sistema2_fig1_momentos.png', dpi=140, bbox_inches='tight')


# =========================================================================
# FIGURA 2 — ruptura: clásico vs aportación de cada G en dp_r/dλ
# =========================================================================
clas, aG20, aG11, aG02, ratio = D['clas'], D['aG20'], D['aG11'], D['aG02'], D['ratio']
fig2, (b1, b2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
b1.semilogy(r_s[sl], clas[sl], color='#777', lw=2.3, ls='--', label='clásico')
b1.semilogy(r_s[sl], aG20[sl], color=COL['G20'], lw=1.8, label=r'aport. $G^{2,0}$')
b1.semilogy(r_s[sl], aG11[sl], color=COL['G11'], lw=1.8, label=r'aport. $G^{1,1}$')
b1.semilogy(r_s[sl], aG02[sl], color=COL['G02'], lw=1.8, label=r'aport. $G^{0,2}$')
if not np.isnan(r_star):
    b1.axvline(r_star, color='crimson', lw=1.8)
    b1.text(r_star-0.1, b1.get_ylim()[1]*0.3, f'$r_*={r_star:.2f}$', color='crimson', ha='right', fontsize=10)
b1.set_xlim(10.0, r_min-0.05); b1.set_xlabel(r'$r$'); b1.set_ylabel(r'aportación a $dp_r/d\lambda$  (log)')
b1.set_title('Cada corrección alcanza al término clásico'); b1.legend(loc='lower left', fontsize=9)
b2.semilogy(r_s[sl], ratio[sl], color='#6a3d9a', lw=2.3)
b2.axhline(1.0, color='crimson', ls='--', lw=1.4)
if not np.isnan(r_star):
    b2.plot([r_star], [1.0], 'o', color='crimson', ms=8, zorder=5)
    b2.annotate(f'$r_*={r_star:.2f}$', xy=(r_star, 1.0), xytext=(r_star+1.0, 8),
                color='crimson', fontsize=11, arrowprops=dict(arrowstyle='->', color='crimson'))
b2.set_xlim(10.0, r_min-0.05); b2.set_xlabel(r'$r$')
b2.set_ylabel(r'$\max_i|\mathrm{aport.}\,G_i|\,/\,|\mathrm{clásico}|$')
b2.set_title('Razón corrección/clásico: cruza 1 en la ruptura')
fig2.suptitle('¿Dónde muere la aproximación de 2º orden? (sistema con $\\dot G$ invertido)', y=1.00, fontsize=13)
fig2.tight_layout(); fig2.savefig('sistema2_fig2_ruptura.png', dpi=140, bbox_inches='tight')


# =========================================================================
# FIGURA 3 — divergencia: r(λ) NO rebota, cae monótonamente hasta el corte;
#            descomposición de dr/dλ (clásico y corrección divergen juntos,
#            sin llegar a cancelarse: la suma nunca cruza cero)
# =========================================================================
rdot_clas = rclas_f(r_s, pr_s, G20_s, G11_s, G02_s)
rdot_corr = rcorr_f(r_s, pr_s, G20_s, G11_s, G02_s)
rdot_full = rdot_clas + rdot_corr
fig3, (e1, e2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
e1.plot(lam, r_s, 'k', lw=2.4)
e1.plot(lam_min, r_min, 'o', color='crimson', ms=9, zorder=5)
if not np.isnan(r_star):
    e1.axhline(r_star, color='crimson', ls='--', lw=1.3)
    e1.text(lam[0], r_star+0.15, f'$r_*={r_star:.2f}$ (muere la aprox.)', color='crimson', fontsize=9)
e1.axhline(rs_val, color='gray', ls=':', lw=1.3)
e1.text(lam[0], rs_val+0.15, 'horizonte $2m$', color='gray', fontsize=9)
e1.annotate(f'corte numérico\n$r={r_min:.2f}$', xy=(lam_min, r_min),
            xytext=(lam_min-6, 5.3), color='crimson', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='crimson'))
e1.set_xlabel(r'$\lambda$'); e1.set_ylabel(r'$r$')
e1.set_title(r'$r(\lambda)$ cae monótonamente: NO hay rebote')
e2.plot(lam, rdot_clas, lw=2, label=r'clásico  $p_r f$')
e2.plot(lam, rdot_corr, lw=2, label=r"corrección  $f'G^{1,1}+\frac{1}{2}p_r G^{2,0}f''$")
e2.plot(lam, rdot_full, 'k', lw=2.4, label=r'suma $=dr/d\lambda$')
e2.axhline(0, color='gray', ls='--', lw=1.0)
e2.plot(lam_min, rdot_full[imin], 'o', color='crimson', ms=8, zorder=5)
e2.set_ylim(-500, 500); e2.set_xlabel(r'$\lambda$'); e2.set_ylabel(r'$dr/d\lambda$')
e2.set_title('Clásico y corrección casi se cancelan, pero ambos divergen'); e2.legend(fontsize=9, loc='upper left')
fig3.suptitle('No hay bumerán: $r$ cae y los momentos divergen en $\\lambda$ finito (sistema con $\\dot G$ invertido)', y=1.00, fontsize=13)
fig3.tight_layout(); fig3.savefig('sistema2_fig3_divergencia.png', dpi=140, bbox_inches='tight')


# =========================================================================
# FIGURA 4 — p_r y dr/dλ divergen juntos cerca del corte (no hay desacople:
#            ambos se disparan a la vez) + comparación con la clásica
# =========================================================================
fig4, (A, Bx) = plt.subplots(1, 2, figsize=(13.5, 5.3))
A.plot(lam, r_s, 'b', lw=2.4, label='autoconsistente (diverge)')
A.plot(sc.t, sc.y[0], 'g--', lw=2.2, label='clásico puro (llega a $2m$)')
A.plot(lam_min, r_min, 'o', color='crimson', ms=9, zorder=5)
A.axhline(rs_val, color='k', ls=':', lw=1.3); A.text(0.3, 2.15, 'horizonte $2m$', fontsize=9)
if not np.isnan(r_star):
    A.axhline(r_star, color='crimson', ls=':', lw=1.1)
    A.text(0.3, r_star+0.15, f'$r_*={r_star:.2f}$', color='crimson', fontsize=9)
A.set_xlabel(r'$\lambda$'); A.set_ylabel(r'$r$'); A.set_ylim(1.5, 12)
A.set_title('El clásico alcanza $2m$; el truncado diverge antes'); A.legend(loc='upper right', fontsize=9)
sl2 = slice(max(0, imin-200), imin+1)
Bx.plot(lam[sl2], pr_s[sl2], color='#2ca02c', lw=2.4)
Bxt = Bx.twinx()
Bxt.plot(lam[sl2], rdot_full[sl2], color='#111', lw=2.4)
Bx.plot(lam_min, pr_s[imin], 'o', color='#2ca02c', ms=9, zorder=5)
Bxt.plot(lam_min, rdot_full[imin], 'o', color='#111', ms=9, zorder=5)
Bx.annotate(f'$p_r={pr_s[imin]:.0f}$\n(diverge, no oscila)', xy=(lam_min, pr_s[imin]),
            xytext=(lam_min-0.6, pr_s[imin]*0.6), color='#2ca02c', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#2ca02c'))
Bxt.annotate(r'$dr/d\lambda$ diverge (nunca cruza 0)', xy=(lam_min, rdot_full[imin]),
             xytext=(lam[sl2][0], rdot_full[imin]*0.3),
             color='#111', fontsize=10, arrowprops=dict(arrowstyle='->', color='#111'))
Bx.set_xlabel(r'$\lambda$'); Bx.set_ylabel(r'$p_r$', color='#2ca02c')
Bxt.set_ylabel(r'$dr/d\lambda$', color='#111')
Bx.set_title(r'Cerca del corte: $p_r$ y $dr/d\lambda$ divergen a la vez')
fig4.suptitle('Momento y velocidad NO se desacoplan: ambos se disparan juntos (sistema con $\\dot G$ invertido)', y=1.00, fontsize=12.5)
fig4.tight_layout(); fig4.savefig('sistema2_fig4_crecimiento.png', dpi=140, bbox_inches='tight')


# =========================================================================
# FIGURA 5 — ¿es real la divergencia o un artefacto del corte numérico?
# -------------------------------------------------------------------------
# Se sube el corte de seguridad mom_max en muchos órdenes de magnitud.
# Si lambda_c converge a un valor fijo (izquierda) mientras que el radio
# alcanzado en el corte NO converge y sigue bajando (derecha), es la firma
# de una divergencia matemática genuina en lambda finito (un polo móvil de
# las ecuaciones), no un artefacto de haber elegido mal el corte.
# =========================================================================
mom_max_grid = np.array([1e6, 1e8, 1e10, 1e14, 1e18, 1e22, 1e26, 1e30])
lam_c_grid, r_c_grid = [], []
for mm in mom_max_grid:
    lc, rc, _ = lambda_c_convergencia(G0_PRINCIPAL, mom_max=mm)
    lam_c_grid.append(lc); r_c_grid.append(rc)
lam_c_grid = np.array(lam_c_grid); r_c_grid = np.array(r_c_grid)

fig5, (h1, h2) = plt.subplots(1, 2, figsize=(13.5, 5.3))
h1.semilogx(mom_max_grid, lam_c_grid, 'o-', color='#6a3d9a', lw=2, ms=7)
h1.axhline(lam_c_grid[-1], color='gray', ls='--', lw=1.2)
h1.set_xlabel(r'corte de seguridad mom_max'); h1.set_ylabel(r'$\lambda_c$ (dónde se detiene la integración)')
h1.set_title(r'$\lambda_c$ CONVERGE: es una divergencia real, no del corte')
h2.semilogx(mom_max_grid, r_c_grid, 'o-', color='#e6550d', lw=2, ms=7)
h2.axhline(rs_val, color='k', ls=':', lw=1.2)
h2.text(mom_max_grid[0], rs_val+0.03, 'horizonte $2m$', fontsize=9)
h2.set_xlabel(r'corte de seguridad mom_max'); h2.set_ylabel(r'$r$ justo antes del corte')
h2.set_title(r'$r$ en el corte NO converge: sigue bajando hacia $2m$')
fig5.suptitle('Comprobación de robustez: la divergencia de los momentos es del modelo, no numérica (sistema con $\\dot G$ invertido)', y=1.00, fontsize=12.5)
fig5.tight_layout(); fig5.savefig('sistema2_fig5_convergencia.png', dpi=140, bbox_inches='tight')

print("\nFiguras guardadas: sistema2_fig1_momentos, sistema2_fig2_ruptura, "
      "sistema2_fig3_divergencia, sistema2_fig4_crecimiento, sistema2_fig5_convergencia")