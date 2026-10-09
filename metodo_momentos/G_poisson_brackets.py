"""
Corchetes de Poisson para los momentos G^{a,b}.
Implementa las ecuaciones (7) y (8):

    {G^{a,b}, G^{c,d}} = a d G^{a-1,b} G^{c,d-1} - b c G^{a,b-1} G^{c-1,d}
                         + sum_n (i hbar / 2)^{n-1} K^n_{abcd} G^{a+c-n, b+d-n}

con n impar, 1 <= n < min(a+c, b+d, a+b, c+d), y

    K^n_{abcd} = sum_{s=0}^{n} (-1)^s s! (n-s)! C(a,s) C(b,n-s) C(c,n-s) C(d,s)
"""

from sympy import I, binomial, factorial, symbols, Function, expand

hbar = symbols('hbar', positive=True)   # ħ
G = Function('G')                        # G(a, b)  <->  G^{a,b}


def K(n, a, b, c, d):
    total = 0
    for s in range(n + 1):
        term = ((-1)**s * factorial(s) * factorial(n - s)
                 * binomial(a, s) * binomial(b, n - s)
                 * binomial(c, n - s) * binomial(d, s))
        total += term
    return total


def pb(a, b, c, d):
    """Corchete de Poisson {G^{a,b}, G^{c,d}} de la ec. (7).
    a, b, c, d deben ser enteros no negativos."""
    # parte "clásica" (los dos primeros términos)
    res = a*d*G(a - 1, b)*G(c, d - 1) - b*c*G(a, b - 1)*G(c - 1, d)

    # suma cuántica: n impar con 1 <= n <= min(a+c, b+d, a+b, c+d)
    M = int(min(a + c, b + d, a + b, c + d))
    for n in range(1, M + 1, 2):         # 1, 3, 5, ...  (<= M)
        coef = K(n, a, b, c, d)
        if coef != 0:
            res += (I*hbar/2)**(n - 1) * coef * G(a + c - n, b + d - n)
    return expand(res)


def reducir_momentos(expr):
    """Aplica las convenciones de los momentos centrados:
    G^{0,0}=1 y los momentos de primer orden G^{1,0}=G^{0,1}=0."""
    repl = {}
    for f in expr.atoms(Function):
        if f.func is G:
            i, j = f.args
            if i == 0 and j == 0:
                repl[f] = 1
            elif i + j == 1:
                repl[f] = 0
    return expand(expr.subs(repl))


if __name__ == '__main__':
    from sympy import pprint

    casos = [(0, 2, 0, 2),
             (0, 2,2,0),
             (0, 2,1,1),
             (2, 0, 0, 2),
             (2, 0,2,0),
             (2, 0,1,1),
             (1, 1, 0, 2),
             (1,1,2,0),
             (1,1,1,1)]    # este genera un término en (i hbar/2)^2

    for (a, b, c, d) in casos:
        br = pb(a, b, c, d)
        print(f"{{G^{{{a},{b}}}, G^{{{c},{d}}}}} =")
        pprint(reducir_momentos(br))
        print("-" * 55)