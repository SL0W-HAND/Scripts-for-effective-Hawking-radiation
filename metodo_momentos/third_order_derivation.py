"""
Derivación simbólica del sistema de momentos a ORDEN 3.
Reutiliza el álgebra de Poisson de G_poisson_brackets.py (ecs. 7 y 8) y le
aplica los coeficientes del Hamiltoniano clásico H para obtener dG^{a,b}/dλ.
"""
import sympy as sp
from sympy import Rational, expand, symbols, Function
from G_poisson_brackets import hbar, G, pb

# ---- símbolos de las derivadas de f y W = E^2/(f c^2), y de p_r ----
p_r = symbols('p_r', real=True)
f, f1, f2, f3, f4 = symbols('f f1 f2 f3 f4', real=True)      # f, f', f'', f''', f''''
W2, W3, W4 = symbols('W2 W3 W4', real=True)                  # W'', W''', W''''

# ---- coeficientes h_{c,d} = (1/(c! d!)) d^{c+d}H_cl/(dr^c dp_r^d) ----
# H_cl = 1/2 (f p_r^2 - W)   =>   d^2H/dr^2 = 1/2(f'' p_r^2 - W''), etc.
H_COEFS = {
    (2, 0): Rational(1, 4) * (f2 * p_r**2 - W2),   # (1/2!)   d^2H/dr^2
    (1, 1): f1 * p_r,                              #          d^2H/dr dp_r
    (0, 2): Rational(1, 2) * f,                    # (1/2!)   d^2H/dp_r^2
    (3, 0): Rational(1, 12) * (f3 * p_r**2 - W3),  # (1/3!)   d^3H/dr^3
    (2, 1): Rational(1, 2) * f2 * p_r,             # (1/2!1!) d^3H/dr^2 dp_r
    (1, 2): Rational(1, 2) * f1,                   # (1/1!2!) d^3H/dr dp_r^2
    (0, 3): sp.Integer(0),                         #          d^3H/dp_r^3 = 0 (H cuadrático en p_r)
}


def orden_total(term):
    """Orden total de un monomio = suma de (a+b) de cada G^{a,b} que contiene,
    más 2 por cada potencia de hbar (en el esquema G^{a,b} ~ hbar^{(a+b)/2})."""
    orden = 0
    for factor in sp.Mul.make_args(term):
        base = factor.base if factor.is_Pow else factor
        exponente = factor.exp if factor.is_Pow else 1
        if isinstance(base, sp.Function) and base.func is G:
            a, b = base.args
            orden += int(a + b) * exponente
        elif base == hbar:
            orden += 2 * exponente
    return orden


def reducir_y_truncar(expr, orden_max=3):
    """Aplica G^{0,0}=1, anula los momentos de primer orden y los de índice
    negativo, y descarta los monomios de orden total > orden_max."""
    expr = expand(expr)

    sustituciones = {}
    for termino in expr.atoms(Function):
        if termino.func is G:
            a, b = termino.args
            if a == 0 and b == 0:
                sustituciones[termino] = 1
            elif a + b == 1:
                sustituciones[termino] = 0
            elif a < 0 or b < 0:
                sustituciones[termino] = 0
    expr = expand(expr.subs(sustituciones))

    truncado = sum(t for t in sp.Add.make_args(expr) if orden_total(t) <= orden_max)
    return expand(truncado)


# ---- ecuaciones de los momentos:  dG^{a,b}/dλ = sum_{c,d} h_{c,d} {G^{a,b}, G^{c,d}} ----
MOMENTOS = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]

if __name__ == '__main__':
    print("=" * 70)
    print("ECUACIONES DE MOVIMIENTO DE LOS MOMENTOS (orden 3)")
    print("=" * 70)

    ecuaciones = {}
    for a, b in MOMENTOS:
        lado_derecho = sum(h_cd * pb(a, b, c, d)
                            for (c, d), h_cd in H_COEFS.items() if h_cd != 0)
        lado_derecho = reducir_y_truncar(lado_derecho, orden_max=3)
        ecuaciones[(a, b)] = lado_derecho

        print(f"\n dG^{{{a},{b}}}/dλ =")
        sp.pprint(lado_derecho)
        if lado_derecho.has(hbar):
            print("   [OJO: contiene hbar, no debería a este orden]")
