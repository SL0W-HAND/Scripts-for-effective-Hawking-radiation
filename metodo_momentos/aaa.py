"""
Pipeline para detectar polos numericos (y su naturaleza: simple vs
espurio vs branch point) en trayectorias de solve_ivp, usando el
algoritmo AAA (scipy.interpolate.AAA).

Contexto fisico: en caida radial en Schwarzschild, en coordenadas
canonicas (r, p_r), el momento p_r diverge en el horizonte r=2m porque
esas coordenadas son patologicas ahi (igual que el tiempo t de
Schwarzschild). El objetivo es DETECTAR esa divergencia solo con los
datos que entrega un integrador "normal" antes de frenarse por
stiffness, sin usar ningun atajo analitico -- porque ese es el
escenario real para un sistema modificado (p.ej. con momentos G) donde
no existe una variable regularizadora conocida de antemano.

Requiere scipy >= 1.15 (scipy.interpolate.AAA).
"""

import numpy as np
from scipy.interpolate import AAA


def limpiar_datos(r, p, n_sub=4000):
    """Filtra no-finitos, duplicados, ordena, y submuestrea (AAA no
    necesita millones de puntos; una malla de unos pocos miles alcanza
    y hace el SVD de cada paso mucho mas rapido)."""
    r = np.asarray(r); p = np.asarray(p)
    mask = np.isfinite(r) & np.isfinite(p)
    r, p = r[mask], p[mask]
    r, idx = np.unique(r, return_index=True)
    p = p[idx]
    if len(r) > n_sub:
        sel = np.linspace(0, len(r) - 1, n_sub).astype(int)
        r, p = r[sel], p[sel]
    return r, p


def ajustar_aaa(r, p, max_terms=60, rtol=1e-8, n_sub=4000, cleanup_tol=1e-13):
    """Corre AAA sobre los datos y devuelve el objeto fit + info de
    convergencia. clean_up() (scipy.interpolate.AAA) ya se aplica solo
    en el constructor (clean_up=True por default) para sacar dobletes de
    Froissart -- pares polo/cero casi cancelados que son puro RUIDO
    numerico. Lo volvemos a llamar explicito por si conviene otro
    cleanup_tol, y devolvemos cuantos saco."""
    r_s, p_s = limpiar_datos(r, p, n_sub=n_sub)
    fit = AAA(r_s, p_s, max_terms=max_terms, rtol=rtol)
    convergio = fit.errors[-1] <= rtol * np.linalg.norm(p_s, ord=np.inf)
    n_dobletes = fit.clean_up(cleanup_tol=cleanup_tol)
    return fit, convergio, n_dobletes


def polo_dominante(fit, r_guess=None, ventana=0.05):
    """De todos los polos que devolvio AAA (ya limpios de dobletes de
    Froissart), identifica el 'dominante': el de mayor |residuo|
    (opcionalmente restringido a una ventana alrededor de r_guess).
    Devuelve (r_h, residuo, polos_en_ventana, residuos_en_ventana)."""
    pol, res = fit.poles(), fit.residues()
    if r_guess is not None:
        cerca = np.abs(pol - r_guess) < ventana
        if not cerca.any():
            cerca = np.ones(len(pol), dtype=bool)
    else:
        cerca = np.ones(len(pol), dtype=bool)
    orden = np.argsort(-np.abs(res[cerca]))
    r_h = pol[cerca][orden][0]
    Res = res[cerca][orden][0]
    return r_h, Res, pol[cerca][orden], res[cerca][orden]


def chequeo_directo_residuo(r, p, r_h, lado='derecha', n_puntos=8):
    """Verificacion INDEPENDIENTE de AAA: la definicion misma de
    residuo de un polo simple es

        Res = lim_{r->r_h} (r - r_h) * p(r)

    Si al acercarse a r_h con los datos crudos ese producto converge a
    un numero finito y NO nulo, es la prueba de que hay un polo SIMPLE
    (orden 1). Si tendiera a 0 -> no hay polo ahi (a lo sumo algo mas
    suave, como un log). Si divergiera -> polo de orden >= 2.
    Devuelve los ultimos `n_puntos` valores de (r-r_h)*p(r) mas cercanos
    a r_h, del lado pedido."""
    r_h_real = r_h.real if np.iscomplexobj(r_h) else r_h
    if lado == 'derecha':
        m = r > r_h_real
    else:
        m = r < r_h_real
    r_m, p_m = r[m], p[m]
    orden = np.argsort(np.abs(r_m - r_h_real))
    r_cerca = r_m[orden][:n_puntos]
    p_cerca = p_m[orden][:n_puntos]
    producto = (r_cerca - r_h_real) * p_cerca
    return r_cerca, producto


def diagnostico_estructura(pol_ventana, res_ventana, ratio_tol=0.02):
    """Distingue polo simple de OTRA cosa (branch point tipo log, etc.)
    mirando cuantos polos en la ventana tienen residuo comparable al
    dominante -- NO alcanza con mirar si son 'chicos' en absoluto (eso
    ya lo filtra clean_up para ruido), hay que comparar relativo al
    mayor. Un polo simple aislado tiene un solo residuo relevante; un
    branch point aproximado por AAA se ve como una 'escalera' de varios
    polos con residuos que decrecen SUAVEMENTE (no son ruido) a medida
    que se acercan al punto de acumulacion."""
    if len(res_ventana) <= 1:
        return "polo_simple", "un solo polo en la ventana -> consistente con polo simple aislado"
    mayor = np.abs(res_ventana[0])
    comparables = np.sum(np.abs(res_ventana) > ratio_tol * mayor)
    if comparables == 1:
        return "polo_simple", (f"1 polo dominante, el resto con residuo < {ratio_tol:.0%} del mayor "
                                f"-> consistente con polo simple aislado")
    else:
        return "branch_point", (f"{comparables} polos con residuo comparable (>{ratio_tol:.0%} del mayor), "
                                 f"apilados en la misma zona -> probable branch point (p.ej. log), NO polo simple")


def im_S_desde_residuo(residuo):
    """Si p(r) tiene un polo simple con residuo Res en r_h, deformar el
    contorno de integracion de S=int p dr con un pequeno semicirculo
    alrededor de r_h (evitando el eje real ahi) produce, por el teorema
    de los residuos, una parte imaginaria

        Im(S) = pi * |Res|

    independiente del radio del semicirculo (limite radio->0). Este es
    el numero relevante para el calculo tipo Hawking/tunneling."""
    return np.pi * abs(residuo.real if np.iscomplexobj(residuo) else residuo)


def contour_check(fit, r_h, eps_list=(1e-2, 1e-3, 1e-4, 1e-5), sentido=-1, n=40000):
    """Chequeo cruzado NUMERICO de la formula de arriba: integra
    literalmente la funcion racional que devolvio AAA (evaluable en
    todo el plano complejo, a diferencia de los datos crudos que solo
    viven en el eje real) sobre un semicirculo de radio eps alrededor
    de r_h, y devuelve la parte imaginaria para cada eps -- deberia dar
    siempre el mismo numero, indicando convergencia."""
    r_h_c = complex(r_h)
    resultados = {}
    for eps in eps_list:
        thetas = np.linspace(0, sentido * np.pi, n)
        rr = r_h_c + eps * np.exp(1j * thetas)
        drdtheta = 1j * eps * np.exp(1j * thetas)
        vals = fit(rr) * drdtheta
        I = np.trapezoid(vals, thetas)
        resultados[eps] = I
    return resultados


def analizar_trayectoria(r, p, r_guess=None, nombre="p(r)", max_terms=60, rtol=1e-8):
    """Corre el pipeline completo y imprime un resumen legible."""
    print(f"\n{'='*70}\nAnalizando: {nombre}\n{'='*70}")
    fit, convergio = ajustar_aaa(r, p, max_terms=max_terms, rtol=rtol)
    print(f"AAA: {fit.errors.size} terminos soporte, convergio={convergio}, "
          f"error final={fit.errors[-1]:.3e}")

    r_h, Res, pol_cerca, res_cerca = polo_dominante(fit, r_guess=r_guess)
    print(f"Polo dominante: r_h = {r_h.real:.10f}  (im={r_h.imag:.2e})")
    print(f"Residuo:        Res = {Res.real:.8f}  (im={Res.imag:.2e})")

    espurio, msg = hay_dobletes_espurios(pol_cerca, res_cerca)
    print(f"Estructura de polos cercanos: {msg}")

    r_cerca, producto = chequeo_directo_residuo(r, p, r_h, lado='derecha')
    print("\nChequeo directo (SIN AAA): Res = lim (r-r_h)*p(r) con datos crudos")
    for rr, pp in zip(r_cerca, producto):
        print(f"   r={rr:.15f}    (r-r_h)*p = {pp.real:.6f}")

    Im_S = im_S_desde_residuo(Res)
    print(f"\nIm(S) = pi*|Res|  =  {Im_S:.6f}")

    return dict(fit=fit, r_h=r_h, Res=Res, espurio=espurio,
                producto_check=(r_cerca, producto), Im_S=Im_S)


if __name__ == "__main__":
    # Ejemplo minimo de uso con datos sinteticos (reemplazar por los
    # (r_raw, p_raw) que devuelva tu propio solve_ivp)
    r_demo = np.linspace(3, 10, 400)
    p_demo = -2.0 / (r_demo - 2.0) + 0.3 * r_demo
    analizar_trayectoria(r_demo, p_demo, r_guess=2.0, nombre="demo sintetica")