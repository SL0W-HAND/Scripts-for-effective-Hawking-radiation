"""
Corchetes de Poisson para los momentos G^{a,b} en la convención POSICIÓN-PRIMERO

    G^{a,b} = < (x - <x>)^a (p - <p>)^b >_Weyl ,     {x, p} = 1 .

La fórmula general de Bojowald et al., PRD 84, 043514 (2011), ec. (5), está
escrita con el MOMENTO en el primer índice:  G_B^{a,b} = <(P-<P>)^a (V-<V>)^b>,
con {V, P} = 1.  Identificando P = p, V = x se tiene G^{a,b} = G_B^{b,a}, así que
el corchete en nuestra convención se obtiene intercambiando índices:

    {G^{a,b}, G^{c,d}} = swap( {G_B^{b,a}, G_B^{d,c}} )

    = b c G^{a,b-1} G^{c-1,d} - a d G^{a-1,b} G^{c,d-1}
      + sum_n (i hbar/2)^{n-1} K^n_{badc} G^{a+c-n, b+d-n}

con n impar, 1 <= n <= min(a+c, b+d, a+b, c+d)  (límite INCLUSIVO), y

    K^n_{abcd} = sum_{m=0}^{n} (-1)^m m! (n-m)! C(a,m) C(b,n-m) C(c,n-m) C(d,m).

Como K^n_{badc} = (-1)^n K^n_{abcd} y n es impar, el resultado es exactamente
-1 por la ec. (5) de Bojowald leída con posición primero (pb_B); esto se usa
como prueba cruzada.  Copiar la ec. (5) sin adaptar la convención (como en
Aragón-Muñoz et al. 2020, ec. 7) invierte el signo de todos los {G, G}.
"""

from sympy import I, binomial, factorial, symbols, Function, expand, Rational

hbar = symbols('hbar', positive=True)   # ħ
G = Function('G')                        # G(a, b)  <->  G^{a,b}


def K(n, a, b, c, d):
    total = 0
    for m in range(n + 1):
        total += ((-1)**m * factorial(m) * factorial(n - m)
                  * binomial(a, m) * binomial(b, n - m)
                  * binomial(c, n - m) * binomial(d, m))
    return total


def _rango_n(a, b, c, d):
    M = int(min(a + c, b + d, a + b, c + d))
    return range(1, M + 1, 2)            # 1, 3, 5, ...  (n <= M)


def pb_B(a, b, c, d):
    """Ec. (5) de Bojowald tal cual (convención momento-primero).
    Leída con posición-primero da el signo INCORRECTO; solo se usa para pruebas."""
    res = a*d*G(a - 1, b)*G(c, d - 1) - b*c*G(a, b - 1)*G(c - 1, d)
    for n in _rango_n(a, b, c, d):
        coef = K(n, a, b, c, d)
        if coef != 0:
            res += (I*hbar/2)**(n - 1) * coef * G(a + c - n, b + d - n)
    return expand(res)


def _swap(expr):
    """Transpone los índices de cada G: G(i, j) -> G(j, i)."""
    return expr.replace(lambda e: isinstance(e, Function) and e.func is G,
                        lambda e: G(e.args[1], e.args[0]))


def pb(a, b, c, d):
    """{G^{a,b}, G^{c,d}} en la convención posición-primero (intercambio de índices)."""
    return expand(_swap(pb_B(b, a, d, c)))


def reducir_momentos(expr):
    """G^{0,0}=1, momentos de primer orden y de índice negativo nulos."""
    repl = {}
    for f in expr.atoms(Function):
        if f.func is G:
            i, j = f.args
            if i == 0 and j == 0:
                repl[f] = 1
            elif i + j == 1 or i < 0 or j < 0:
                repl[f] = 0
    return expand(expr.subs(repl))


def _pruebas():
    g = lambda a, b: G(a, b)
    esperados = {
        (2, 0, 0, 2): 4*g(1, 1),
        (0, 2, 2, 0): -4*g(1, 1),
        (2, 0, 1, 1): 2*g(2, 0),
        (1, 1, 0, 2): 2*g(0, 2),
        (1, 1, 1, 1): 0,
        (3, 0, 0, 3): 9*g(2, 2) - 9*g(2, 0)*g(0, 2) - Rational(3, 2)*hbar**2,
        (0, 3, 3, 0): -9*g(2, 2) + 9*g(2, 0)*g(0, 2) + Rational(3, 2)*hbar**2,
        (2, 1, 1, 2): 3*g(2, 2) + g(2, 0)*g(0, 2) - 4*g(1, 1)**2 + hbar**2/2,
    }
    for (a, b, c, d), val in esperados.items():
        assert expand(reducir_momentos(pb(a, b, c, d)) - val) == 0, (a, b, c, d)
    rango = range(5)
    for a in rango:
        for b in rango:
            for c in rango:
                for d in rango:
                    x = reducir_momentos(pb(a, b, c, d))
                    assert expand(x + reducir_momentos(pb_B(a, b, c, d))) == 0
                    assert expand(x + reducir_momentos(pb(c, d, a, b))) == 0
    # Ehrenfest, partícula libre H = p^2/2m:  d<dx^2>/dt = (1/2m){G20,G02} = (2/m) G11
    m = symbols('m', positive=True)
    assert expand(reducir_momentos(pb(2, 0, 0, 2))/(2*m) - 2*g(1, 1)/m) == 0
    print("Todas las pruebas pasaron (valores esperados, pb = -pb_B, antisimetría, Ehrenfest).")


if __name__ == '__main__':
    from sympy import pprint
    _pruebas()
    casos = [(2, 0, 0, 2), (2, 0, 1, 1), (1, 1, 0, 2), (1, 1, 1, 1)]   # (1,1,1,1) vale 0
    for (a, b, c, d) in casos:
        print(f"{{G^{{{a},{b}}}, G^{{{c},{d}}}}} =")
        pprint(reducir_momentos(pb(a, b, c, d)))
        print("-" * 55)
