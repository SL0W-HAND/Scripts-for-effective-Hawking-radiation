"""
=============================================================================
 FOTONES EN EDDINGTON--FINKELSTEIN CON MOMENTOS CUANTICOS (2º orden)
 Figuras D_fig1 ... D_fig7 en un solo script.
=============================================================================
 Fotones (m=0). Los 4 casos (2 cartas x 2 ramas) se integran SIEMPRE HACIA
 ADELANTE en lambda, y cada uno recibe su condicion inicial en el PUNTO DE
 PARTIDA REAL de su trayectoria:

     ramas ENTRANTES  ->  se siembran FUERA,  en r0 = 6   (caen hacia 2m)
     ramas SALIENTES  ->  se siembran DENTRO, en r0 = 1   (emergen hacia afuera)

 Asi la condicion de minima incertidumbre es una condicion INICIAL genuina en
 los cuatro casos.

 Capa de masa del foton H=0 -> dos raices por carta:
     p_r = 0              REGULAR en el horizonte
     p_r = -sgn*2E/(cf)   DIVERGE en el horizonte
 con sgn = -1 (carta AVANZADA, v=t+r_*/c) y sgn = +1 (RETARDADA, u=t-r_*/c).

 Estado inicial saturado (hbar=1):  G20=sigma^2, G02=hbar^2/(4 sigma^2), G11=0.

 DIAGNOSTICO DE VALIDEZ: para fotones el termino clasico de dp_r/dl es
 -1/2 f' p_r^2, que se ANULA en la rama regular (p_r=0); ese cociente es por
 tanto inservible. Usamos la ecuacion de la posicion, cuyo termino clasico
 f p_r -/+ E/c NO se anula:
        ratio = |1/2 f'' p_r G20 + f' G11| / |f p_r -/+ E/c|

 COORDENADA NULA. De los super-Hamiltonianos

   AVANZADA :  H = 1/2 [ (2/c) p_v p_r + f p_r^2 ]   ->  dv/dl = dH/dp_v = p_r/c
   RETARDADA:  H = 1/2 [ -(2/c) p_u p_r + f p_r^2 ]  ->  du/dl = dH/dp_u = -p_r/c

 El termino con la coordenada nula es LINEAL en p_r, asi que no recibe
 correcciones de los momentos G. Con c=1:  dv/dl = +p_r,  du/dl = -p_r  (EXACTO).
 En la rama regular clasica p_r = 0  =>  v = const (avanzada) o u = const
 (retardada); cualquier corrimiento es un efecto PURAMENTE CUANTICO.

 Rayos nulos de referencia (con r_* = r + 2m ln|r/2m - 1|):
   avanzada : entrantes  v = const ;  salientes  v - 2 r_* = const
   retardada: salientes  u = const ;  entrantes  u + 2 r_* = const

 Unidades geometrizadas c=G=1 => r_s = 2.
=============================================================================
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# CONSTANTES
# =========================================================================
rs_val = 2.0
hbar = 1.0
E_val = 1.0
SIGMA2_0 = 0.5                              # G20 inicial (varianza de posición)
G02_0 = hbar**2 / (4 * SIGMA2_0)            # G02 inicial (paquete de mínima incertidumbre)
R_FUERA = 6.0                               # radio de siembra/llegada fuera del horizonte
R_DENTRO = 1.0                              # radio de siembra/llegada dentro del horizonte


# =========================================================================
# SIMBOLICO: ecuaciones de movimiento de (p_r, G20, G11, G02) a 2º orden
# =========================================================================
r, p_r = sp.symbols('r p_r', real=True)
G20, G11, G02 = sp.symbols('G20 G11 G02', real=True)
c, G, M, E = sp.symbols('c G M E', positive=True)

f = 1 - 2*G*M / (c**2 * r)
f1 = sp.diff(f, r)
f2 = sp.diff(f, r, 2)
f3 = sp.diff(f, r, 3)

dp_r_expr = (-sp.Rational(1, 2)*f1*p_r**2
             - sp.Rational(1, 4)*f3*p_r**2*G20
             - f2*p_r*G11
             - sp.Rational(1, 2)*f1*G02)
dG20_expr = -2*f1*p_r*G20 - 2*f*G11
dG11_expr = sp.Rational(1, 2)*f2*p_r**2*G20 - f*G02
dG02_expr = f2*p_r**2*G11 + 2*f1*p_r*G02
correccion_dr = sp.Rational(1, 2)*f2*p_r*G20 + f1*G11

const_subs = {c: 1.0, G: 1.0, M: 1.0, E: E_val}
state = (r, p_r, G20, G11, G02)
a_funcion_numerica = lambda expr: sp.lambdify(state, expr.subs(const_subs), 'numpy')

# signo=-1 avanzada, signo=+1 retardada
DR_CLASICO = {
    -1: a_funcion_numerica(f*p_r - E/c),
    +1: a_funcion_numerica(f*p_r + E/c),
}
DR_CORRECCION = a_funcion_numerica(correccion_dr)
DR_DLAMBDA = {
    -1: a_funcion_numerica(f*p_r - E/c + correccion_dr),
    +1: a_funcion_numerica(f*p_r + E/c + correccion_dr),
}
DPR_DLAMBDA = a_funcion_numerica(dp_r_expr)
DG20_DLAMBDA = a_funcion_numerica(dG20_expr)
DG11_DLAMBDA = a_funcion_numerica(dG11_expr)
DG02_DLAMBDA = a_funcion_numerica(dG02_expr)
INVARIANTE_U = a_funcion_numerica(G20*G02 - G11**2)


def rstar(x):
    """Coordenada tortuga, valida a ambos lados del horizonte."""
    x = np.asarray(x, dtype=float)
    return x + rs_val * np.log(np.abs(x/rs_val - 1.0) + 1e-300)


def momento_inicial(signo, rama, r0):
    f0 = 1 - rs_val/r0
    p_r0 = 0.0 if rama == 'reg' else -signo * 2*E_val/f0
    es_saliente = (f0*p_r0 + signo*E_val) > 0
    return p_r0, es_saliente


def integrar(signo, rama, r0, momento_max=1e5):
    """y = [chi, r, p_r, G20, G11, G02];  chi = v (signo=-1) o u (signo=+1)
       dchi/dl = -signo * p_r  ->  +p_r avanzada, -p_r retardada.
       Integracion SIEMPRE hacia adelante en lambda."""
    p_r0, es_saliente = momento_inicial(signo, rama, r0)
    derivadas_G = [DR_DLAMBDA[signo], DPR_DLAMBDA, DG20_DLAMBDA, DG11_DLAMBDA, DG02_DLAMBDA]

    def sistema_extendido(lam, y):
        estado = y[1:]
        return [-signo*y[2]] + [fn(*estado) for fn in derivadas_G]

    r_objetivo = R_FUERA if es_saliente else R_DENTRO

    def llega_a_r_objetivo(lam, y):
        return (r_objetivo - y[1]) if es_saliente else (y[1] - r_objetivo)
    llega_a_r_objetivo.terminal = True
    llega_a_r_objetivo.direction = -1

    def diverge_momentos(lam, y):
        return momento_max - max(abs(y[2]), abs(y[3]), abs(y[4]), abs(y[5]))
    diverge_momentos.terminal = True
    diverge_momentos.direction = -1

    sol = solve_ivp(sistema_extendido, (0, 200), [0.0, r0, p_r0, SIGMA2_0, 0.0, G02_0],
                    method='DOP853', rtol=1e-11, atol=1e-14,
                    events=[llega_a_r_objetivo, diverge_momentos], max_step=0.001)

    estado = sol.y[1:]
    rr, pp, g20, g11, g02 = estado
    ratio = np.abs(DR_CORRECCION(*estado)) / np.maximum(np.abs(DR_CLASICO[signo](*estado)), 1e-300)

    # primer radio donde la correccion supera al termino clasico
    por_encima = ratio >= 1.0
    if por_encima.size and por_encima[0]:
        r_ruptura, tipo_ruptura = rr[0], 'desde el inicio'
    else:
        i = int(np.argmax(por_encima))
        r_ruptura, tipo_ruptura = ((rr[i], 'cruza') if (por_encima.size and por_encima[i])
                                   else (float('nan'), 'nunca'))

    dr = np.diff(rr)
    return dict(lam=sol.t, chi=sol.y[0], r=rr, pr=pp, G20=g20, G11=g11, G02=g02,
                sig=np.sqrt(np.abs(g20)), sigp=np.sqrt(np.abs(g02)), U=INVARIANTE_U(*estado),
                ratio=ratio, rstar=r_ruptura, tipo=tipo_ruptura,
                mono=bool(np.all(dr >= -1e-12) or np.all(dr <= 1e-12)),
                ih=int(np.argmin(np.abs(rr - rs_val))),
                cruzo=(rr.max() > rs_val) and (rr.min() < rs_val),
                p0=p_r0, r0=r0, sale=es_saliente, exploto=len(sol.t_events[1]) > 0)


def clasico(signo, rama, r0):
    """Trayectoria clasica (sin momentos G), frenada junto al horizonte."""
    p_r0, es_saliente = momento_inicial(signo, rama, r0)
    r_objetivo = R_FUERA if es_saliente else R_DENTRO

    def sistema_clasico(lam, y):
        R, P = y
        return [(1 - rs_val/R)*P + signo*E_val, -0.5*(rs_val/R**2)*P**2]

    def llega_a_r_objetivo(lam, y):
        return (r_objetivo - y[0]) if es_saliente else (y[0] - r_objetivo)
    llega_a_r_objetivo.terminal = True
    llega_a_r_objetivo.direction = -1

    def junto_al_horizonte(lam, y):
        return abs(y[0] - rs_val) - 0.0008
    junto_al_horizonte.terminal = True
    junto_al_horizonte.direction = -1

    return solve_ivp(sistema_clasico, (0, 200), [r0, p_r0], method='DOP853',
                     rtol=1e-12, atol=1e-14,
                     events=[llega_a_r_objetivo, junto_al_horizonte], max_step=0.0005)


# =========================================================================
# INTEGRACION DE LAS CUATRO RAMAS (cada una sembrada donde empieza su viaje)
# =========================================================================
PLAN_INTEGRACION = [
    ('AV', 'reg',  -1, 'reg',  R_FUERA,  'AVANZADA · entrante regular ($p_r{=}0$)'),
    ('AV', 'sing', -1, 'sing', R_DENTRO, 'AVANZADA · saliente NO regular ($p_r{=}2E/cf$)'),
    ('RE', 'reg',  +1, 'reg',  R_DENTRO, 'RETARDADA · saliente regular ($p_r{=}0$)'),
    ('RE', 'sing', +1, 'sing', R_FUERA,  'RETARDADA · entrante NO regular ($p_r{=}{-}2E/cf$)'),
]

resultados = {}
resultados_clasicos = {}
for carta, rama, signo, rama_ci, r0, etiqueta in PLAN_INTEGRACION:
    resultados[(carta, rama)] = integrar(signo, rama_ci, r0)
    resultados_clasicos[(carta, rama)] = clasico(signo, rama_ci, r0)
etiquetas = {(carta, rama): etiqueta for carta, rama, _, _, _, etiqueta in PLAN_INTEGRACION}
etiqueta_rama = lambda k: etiquetas[k].split('· ')[1]
etiqueta_corta = lambda k: etiquetas[k].replace('· ', '').split(' ($')[0]


# =========================================================================
# NUMEROS PARA EL INFORME
# =========================================================================
print("======= 4 CASOS · salientes sembradas DENTRO (r0=1), entrantes fuera (r0=6) =======")
print(f"sigma^2={SIGMA2_0}, hbar={hbar}, E={E_val}, U0=1/4, r_s=2\n")
for carta, rama, _, _, r0, etiqueta in PLAN_INTEGRACION:
    d = resultados[(carta, rama)]
    C = resultados_clasicos[(carta, rama)]
    nombre_chi = 'v' if carta == 'AV' else 'u'
    print(etiqueta.replace('$', '').replace('{=}', '=').replace('{-}', '-').replace('\\', ''))
    print(f"   siembra en r0={r0} ({'DENTRO' if r0 < rs_val else 'fuera'}),  p_r(0)={d['p0']:+7.3f},  "
          f"sentido: {'saliente' if d['sale'] else 'entrante'}")
    print(f"   recorrido r: {d['r'][0]:.2f} -> {d['r'][-1]:.3f}   cruza 2m: {d['cruzo']}   "
          f"monotona: {d['mono']}   explota: {d['exploto']}")
    if not d['mono']:
        r_rebote = d['r'].max() if d['sale'] else d['r'].min()
        print(f"   *** REBOTE (artefacto) en r={r_rebote:.3f} ***")
    if d['cruzo']:
        ih = d['ih']
        print(f"   en 2m: p_r={d['pr'][ih]:+.4f}  sigma_r={d['sig'][ih]:.4f}  "
              f"sigma_r/r_s={d['sig'][ih]/rs_val:.2f}  {nombre_chi}={d['chi'][ih]:+.4f}")
    else:
        print(f"   NUNCA llega a 2m (se queda en r={d['r'].min():.3f}); no hay valores en el horizonte")
    nota = "   (clasico regular: 0 exacto)" if rama == 'reg' else ""
    print(f"   {nombre_chi}: 0 -> {d['chi'][-1]:+.4g}{nota}")
    print(f"   validez: ratio(dr) max={d['ratio'].max():.2f}  -> r*={d['rstar']:.4f} ({d['tipo']})")
    print(f"   U drift={np.max(np.abs(d['U']-0.25)):.1e}   clasico llega a r={C.y[0][-1]:.4f}\n")


# =========================================================================
# FIGURAS
# =========================================================================
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11.5,
                     'axes.grid': True, 'grid.alpha': 0.25})
COLOR_REGULAR = '#1f77b4'
COLOR_SINGULAR = '#d62728'
COLOR_RAYO_NULO = '#9e9e9e'
ESTILOS = {('AV', 'reg'): (COLOR_REGULAR, '-'), ('RE', 'reg'): ('#2ca02c', '--'),
           ('AV', 'sing'): (COLOR_SINGULAR, '-'), ('RE', 'sing'): ('#ff7f0e', '--')}
CARTAS = [('AV', 'Carta AVANZADA'), ('RE', 'Carta RETARDADA')]


def horizonte_h(ax):
    ax.axhline(rs_val, ls='--', color='k', lw=1.5, alpha=.75)


def horizonte_v(ax):
    ax.axvline(rs_val, ls='--', color='k', lw=1.5, alpha=.75)
    ax.axvspan(0, rs_val, color='k', alpha=0.05)


# ---------- FIG 1: r(lambda) por carta, marcando el punto de siembra ----------
fig1, paneles = plt.subplots(1, 2, figsize=(13.5, 5.4))
for ax, carta, titulo in [(paneles[0], 'AV', 'Carta AVANZADA  $v=t+r_*/c$'),
                          (paneles[1], 'RE', 'Carta RETARDADA  $u=t-r_*/c$')]:
    for rama, color in [('reg', COLOR_REGULAR), ('sing', COLOR_SINGULAR)]:
        d = resultados[(carta, rama)]
        C = resultados_clasicos[(carta, rama)]
        ax.plot(d['lam'], d['r'], color=color, lw=2.5, label=etiqueta_rama((carta, rama)))
        ax.plot(0, d['r'][0], 'o', color=color, ms=9, zorder=6)
        ax.plot(C.t, C.y[0], color=color, lw=1.1, ls=':')
        if not d['mono']:
            k = int(np.argmax(d['r'])) if d['sale'] else int(np.argmin(d['r']))
            ax.plot(d['lam'][k], d['r'][k], 'X', color=color, ms=11, zorder=6)
            ax.annotate(f"rebote\n$r\\approx{d['r'][k]:.2f}$", xy=(d['lam'][k], d['r'][k]),
                        xytext=(d['lam'][k]+0.7, d['r'][k]-1.6), color=color, fontsize=8.5,
                        arrowprops=dict(arrowstyle='->', color=color))
    horizonte_h(ax)
    ax.text(0.05, rs_val+0.15, 'horizonte $2m$', fontsize=9)
    ax.axhspan(0, rs_val, color='k', alpha=0.05)
    ax.text(0.05, 0.55, 'interior', fontsize=8.5, color='gray')
    ax.set_ylim(0, 9.8)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$r$')
    ax.set_title(titulo)
    ax.legend(fontsize=8.3, loc='upper left')
fig1.suptitle('Salientes sembradas DENTRO ($r_0{=}1$), entrantes fuera ($r_0{=}6$); '
              'círculo $=$ punto de siembra, punteado $=$ clásica', y=1.00, fontsize=11.8)
fig1.tight_layout()
fig1.savefig('D_fig1_cartas.png', dpi=140, bbox_inches='tight')

# ---------- FIG 2: p_r vs r ----------
fig2, paneles = plt.subplots(1, 2, figsize=(13.5, 5.2))
for ax, (carta, titulo) in zip(paneles, CARTAS):
    for rama, color in [('reg', COLOR_REGULAR), ('sing', COLOR_SINGULAR)]:
        d = resultados[(carta, rama)]
        ax.plot(d['r'], d['pr'], color=color, lw=2.5, label=etiqueta_rama((carta, rama)))
        ax.plot(d['r'][0], d['pr'][0], 'o', color=color, ms=8, zorder=6)
    ax.axhline(0, color='gray', ls=':', lw=1.3)
    horizonte_v(ax)
    ax.set_xlim(0.6, 6.3)
    ax.set_ylim(-30, 10)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$p_r$')
    ax.set_title(titulo)
    ax.legend(fontsize=8.3, loc='lower right')
    ax.text(rs_val+0.07, 7, '$2m$', fontsize=10)
fig2.suptitle('Momento radial: pequeño en las ramas regulares, enorme en las no regulares',
              y=1.00, fontsize=12.3)
fig2.tight_layout()
fig2.savefig('D_fig2_momento.png', dpi=140, bbox_inches='tight')

# ---------- FIG 3: sigma_r vs r y vs lambda ----------
fig3, (panel_r, panel_lam) = plt.subplots(1, 2, figsize=(13.5, 5.2))
for k, (color, estilo) in ESTILOS.items():
    d = resultados[k]
    panel_r.semilogy(d['r'], d['sig'], color=color, ls=estilo, lw=2.3, label=etiqueta_corta(k))
    panel_r.plot(d['r'][0], d['sig'][0], 'o', color=color, ms=7, zorder=6)
    panel_lam.semilogy(d['lam'], d['sig'], color=color, ls=estilo, lw=2.3)
horizonte_v(panel_r)
panel_r.text(rs_val+0.07, panel_r.get_ylim()[1]*0.5, '$2m$', fontsize=10)
panel_r.set_xlim(0.6, 6.3)
panel_r.set_xlabel(r'$r$')
panel_r.set_ylabel(r'$\sigma_r=\sqrt{G^{2,0}}$  (log)')
panel_r.set_title(r'$\sigma_r(r)$  (círculo $=$ siembra)')
panel_r.legend(fontsize=7.8, loc='upper left')
panel_lam.set_xlabel(r'$\lambda$')
panel_lam.set_ylabel(r'$\sigma_r$  (log)')
panel_lam.set_title(r'$\sigma_r(\lambda)$')
fig3.suptitle(r'Ancho del paquete: la saliente regular sembrada dentro es la más localizada',
              y=1.00, fontsize=12.3)
fig3.tight_layout()
fig3.savefig('D_fig3_sigma.png', dpi=140, bbox_inches='tight')

# ---------- FIG 4: validez (ratio en dr/dl) y localizacion sigma_r/r ----------
fig4, (panel_ratio, panel_loc) = plt.subplots(1, 2, figsize=(13.5, 5.2))
for k, (color, estilo) in ESTILOS.items():
    d = resultados[k]
    panel_ratio.semilogy(d['r'], np.maximum(d['ratio'], 1e-6), color=color, ls=estilo, lw=2.3,
                         label=etiqueta_corta(k))
    panel_loc.plot(d['r'], d['sig']/d['r'], color=color, ls=estilo, lw=2.3)
panel_ratio.axhline(1.0, color='k', ls='--', lw=1.5)
horizonte_v(panel_ratio)
panel_ratio.text(rs_val+0.07, 2e-5, '$2m$', fontsize=10)
panel_ratio.set_xlim(0.6, 6.3)
panel_ratio.set_xlabel(r'$r$')
panel_ratio.set_ylabel(r'$|$corr.$|/|$clás.$|$ en $dr/d\lambda$  (log)')
panel_ratio.set_title('Validez: por debajo de 1 la truncación aguanta')
panel_ratio.legend(fontsize=7.6, loc='lower left')
panel_loc.axhline(1.0, color='k', ls='--', lw=1.5)
horizonte_v(panel_loc)
panel_loc.text(rs_val+0.07, 2.1, '$2m$', fontsize=10)
panel_loc.set_xlim(0.6, 6.3)
panel_loc.set_ylim(0, 2.5)
panel_loc.set_xlabel(r'$r$')
panel_loc.set_ylabel(r'$\sigma_r/r$')
panel_loc.set_title(r'Localización: $\sigma_r/r<1$ significa paquete localizado')
fig4.suptitle('Sólo las ramas regulares mantienen la truncación bajo control', y=1.00, fontsize=12.3)
fig4.tight_layout()
fig4.savefig('D_fig4_validez.png', dpi=140, bbox_inches='tight')

# ---------- FIG 5: conservacion de U ----------
fig5, panel_U = plt.subplots(figsize=(6.9, 5.2))
for k, (color, estilo) in ESTILOS.items():
    d = resultados[k]
    panel_U.semilogy(d['r'], np.abs(d['U']-0.25)+1e-20, color=color, ls=estilo, lw=2.2,
                     label=etiqueta_corta(k))
horizonte_v(panel_U)
panel_U.text(rs_val+0.07, 1e-9, '$2m$', fontsize=10)
panel_U.set_xlim(0.6, 6.3)
panel_U.set_xlabel(r'$r$')
panel_U.set_ylabel(r'$|U-\hbar^2/4|$  (log)')
panel_U.set_title(r'Invariante $U$: exacto en las regulares')
panel_U.legend(fontsize=7.6, loc='upper left')
fig5.suptitle('El estado permanece saturado mientras la aproximación es válida', y=1.00, fontsize=12.3)
fig5.tight_layout()
fig5.savefig('D_fig5_incertidumbre.png', dpi=140, bbox_inches='tight')

# ---------- FIG 6: diagramas EF  (r,v) y (r,u) ----------
fig6, (panel_av, panel_re) = plt.subplots(1, 2, figsize=(13.8, 6.0))

# --- panel A: carta AVANZADA, plano (r, v) ---
r_fuera = np.linspace(2.004, 6.6, 700)
r_dentro = np.linspace(0.62, 1.996, 700)
for k, offset in enumerate(np.arange(-14, 10, 2.0)):        # salientes: v = 2 r_* + const
    panel_av.plot(r_fuera, 2*rstar(r_fuera) + offset, color=COLOR_RAYO_NULO, lw=0.8, alpha=.55,
                  label='rayos salientes' if k == 0 else None)
    panel_av.plot(r_dentro, 2*rstar(r_dentro) + offset, color=COLOR_RAYO_NULO, lw=0.8, alpha=.55)
for k, v_const in enumerate(np.arange(-12, 6, 1.5)):         # entrantes: v = const
    panel_av.plot([0.62, 6.6], [v_const, v_const], color=COLOR_RAYO_NULO, lw=0.7, ls=':', alpha=.7,
                  label='rayos entrantes ($v=$cte)' if k == 0 else None)
for rama, color in [('reg', COLOR_REGULAR), ('sing', COLOR_SINGULAR)]:
    d = resultados[('AV', rama)]
    panel_av.plot(d['r'], d['chi'], color=color, lw=3.0, zorder=5, label=etiqueta_rama(('AV', rama)))
    panel_av.plot(d['r'][0], d['chi'][0], 'o', color=color, ms=10, zorder=6)
panel_av.axvline(rs_val, color='k', lw=2.0, ls='--', zorder=4)
panel_av.text(rs_val+0.06, 4.2, 'horizonte $2m$', fontsize=9.5, rotation=90, va='top')
panel_av.axvspan(0.62, rs_val, color='k', alpha=0.06, zorder=0)
panel_av.text(0.72, -11.3, 'interior', fontsize=9, color='gray')
panel_av.set_xlim(0.62, 6.6)
panel_av.set_ylim(-12, 5)
panel_av.set_xlabel(r'$r$')
panel_av.set_ylabel(r'$v=t+r_*/c$')
panel_av.set_title('Carta AVANZADA: plano $(r,v)$')
panel_av.legend(fontsize=8.0, loc='lower right', framealpha=.92)

# --- panel B: carta RETARDADA, plano (r, u) ---
for k, offset in enumerate(np.arange(-8, 18, 2.0)):          # entrantes: u = -2 r_* + const
    panel_re.plot(r_fuera, -2*rstar(r_fuera) + offset, color=COLOR_RAYO_NULO, lw=0.8, alpha=.55,
                  label='rayos entrantes' if k == 0 else None)
    panel_re.plot(r_dentro, -2*rstar(r_dentro) + offset, color=COLOR_RAYO_NULO, lw=0.8, alpha=.55)
for k, u_const in enumerate(np.arange(-4, 14, 1.5)):          # salientes: u = const
    panel_re.plot([0.62, 6.6], [u_const, u_const], color=COLOR_RAYO_NULO, lw=0.7, ls=':', alpha=.7,
                  label='rayos salientes ($u=$cte)' if k == 0 else None)
for rama, color in [('reg', COLOR_REGULAR), ('sing', COLOR_SINGULAR)]:
    d = resultados[('RE', rama)]
    panel_re.plot(d['r'], d['chi'], color=color, lw=3.0, zorder=5, label=etiqueta_rama(('RE', rama)))
    panel_re.plot(d['r'][0], d['chi'][0], 'o', color=color, ms=10, zorder=6)
    if rama == 'sing':
        i_rebote = int(np.argmin(d['r']))
        panel_re.plot(d['r'][i_rebote], d['chi'][i_rebote], 'X', color=color, ms=13, zorder=7)
        panel_re.annotate(f"rebote\n$r={d['r'][i_rebote]:.2f}$",
                          xy=(d['r'][i_rebote], d['chi'][i_rebote]),
                          xytext=(d['r'][i_rebote]-2.0, d['chi'][i_rebote]+1.3),
                          color=color, fontsize=9, arrowprops=dict(arrowstyle='->', color=color))
panel_re.axvline(rs_val, color='k', lw=2.0, ls='--', zorder=4)
panel_re.text(rs_val+0.06, 12.6, 'horizonte $2m$', fontsize=9.5, rotation=90, va='top')
panel_re.axvspan(0.62, rs_val, color='k', alpha=0.06, zorder=0)
panel_re.text(0.72, -3.4, 'interior', fontsize=9, color='gray')
panel_re.set_xlim(0.62, 6.6)
panel_re.set_ylim(-4, 13)
panel_re.set_xlabel(r'$r$')
panel_re.set_ylabel(r'$u=t-r_*/c$')
panel_re.set_title('Carta RETARDADA: plano $(r,u)$')
panel_re.legend(fontsize=8.0, loc='upper right', framealpha=.92)

fig6.suptitle(r'Diagramas de Eddington--Finkelstein: trayectorias sobre la malla de rayos nulos '
              r'(círculo $=$ siembra)', y=0.99, fontsize=12.3)
fig6.tight_layout()
fig6.savefig('D_fig6_diagramas_vu.png', dpi=140, bbox_inches='tight')

# ---------- FIG 7: el corrimiento cuántico de la coordenada nula ----------
fig7, (panel_chi, panel_pr) = plt.subplots(1, 2, figsize=(13.5, 5.0))
for carta, color, nombre_chi, estilo in [('AV', COLOR_REGULAR, 'v', '-'),
                                         ('RE', '#2ca02c', 'u', '--')]:
    d = resultados[(carta, 'reg')]
    panel_chi.plot(d['r'], d['chi'], color=color, ls=estilo, lw=2.6,
                   label=fr'{"AVANZADA" if carta == "AV" else "RETARDADA"}: ${nombre_chi}(r)$')
    panel_chi.plot(d['r'][0], d['chi'][0], 'o', color=color, ms=8, zorder=6)
    panel_pr.plot(d['r'], d['pr'], color=color, ls=estilo, lw=2.6,
                  label=fr'$d{nombre_chi}/d\lambda=\mp p_r$')
panel_chi.axhline(0, color='k', lw=1.6, ls=':', label='clásico ($p_r{=}0$): cte')
panel_chi.axvline(rs_val, color='k', lw=1.5, ls='--')
panel_chi.axvspan(0.9, rs_val, color='k', alpha=0.06)
panel_chi.text(rs_val+0.07, 1.55, '$2m$', fontsize=10)
panel_chi.set_xlim(0.9, 6.1)
panel_chi.set_xlabel(r'$r$')
panel_chi.set_ylabel(r'$v$  o  $u$')
panel_chi.set_title('Ramas regulares: la coordenada nula se corre')
panel_chi.legend(fontsize=8.6, loc='upper left')

panel_pr.axhline(0, color='k', lw=1.6, ls=':')
panel_pr.axvline(rs_val, color='k', lw=1.5, ls='--')
panel_pr.axvspan(0.9, rs_val, color='k', alpha=0.06)
panel_pr.text(rs_val+0.07, 0.02, '$2m$', fontsize=10)
panel_pr.set_xlim(0.9, 6.1)
panel_pr.set_xlabel(r'$r$')
panel_pr.set_ylabel(r'$p_r$')
panel_pr.set_title(r'El motor del corrimiento: $p_r\neq0$ inducido por los $G$')
panel_pr.legend(fontsize=9, loc='lower left')

fig7.suptitle(r'El corrimiento de $v$ (o $u$) es un efecto puramente cuántico: '
              r'clásicamente $p_r=0$ y la coordenada nula sería constante', y=1.00, fontsize=11.8)
fig7.tight_layout()
fig7.savefig('D_fig7_corrimiento.png', dpi=140, bbox_inches='tight')


# =========================================================================
# EXPORTAR NUMEROS
# =========================================================================
with open('numeros.txt', 'w') as archivo:
    for carta, rama, _, _, r0, etiqueta in PLAN_INTEGRACION:
        d = resultados[(carta, rama)]
        nombre_chi = 'v' if carta == 'AV' else 'u'
        ih = d['ih']
        en_horizonte = lambda x: f"{x:.4f}" if d['cruzo'] else "n/a"
        archivo.write(f"{carta}-{rama} r0={r0} p0={d['p0']:.4f} cruzo={d['cruzo']} mono={d['mono']} "
                      f"pr_h={en_horizonte(d['pr'][ih])} sig_h={en_horizonte(d['sig'][ih])} "
                      f"ratio_max={d['ratio'].max():.2f} rstar={d['rstar']:.4f}({d['tipo']}) "
                      f"Udrift={np.max(np.abs(d['U']-0.25)):.1e} "
                      f"{nombre_chi}_fin={d['chi'][-1]:.4g} "
                      f"{nombre_chi}_en_2m={en_horizonte(d['chi'][ih])}\n")

print("Figuras D_fig1 ... D_fig7 guardadas.")
