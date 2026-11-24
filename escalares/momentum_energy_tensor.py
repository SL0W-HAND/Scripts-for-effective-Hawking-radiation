from ricci import ricci_tensor
from christoffel import g_ij, gij, c, r, m, l, r0, m_regular
from sympy import symbols, pi, Rational, cancel, factor, simplify, latex, limit

# -------------------------------------------------------------------
# Tensor energía-momento efectivo
#
# Se usan las coordenadas (x^0, r, theta, phi) con x^0 = ct, de modo que
# todas las componentes del tensor de Einstein tienen unidades de
# 1/longitud^2. Las ecuaciones de Einstein
#
#   G_mu_nu = R_mu_nu - (1/2) g_mu_nu R = (8 pi G / c^4) T_mu_nu
#
# definen T_mu_nu = c^4/(8 pi G) G_mu_nu para cualquier métrica.
#
# Para una métrica estática y esféricamente simétrica, T^mu_nu es
# diagonal y se identifica con un fluido anisótropo:
#
#   T^mu_nu = diag(-u, p_r, p_t, p_t),
#
# con u la densidad de energía y p_r, p_t las presiones radial y
# tangencial. La NEC, T_mu_nu k^mu k^nu >= 0 para todo k nulo, equivale a
#
#   u + p_r >= 0   y   u + p_t >= 0.
#
# Para Schwarzschild (vacío) se espera T_mu_nu = 0.
# -------------------------------------------------------------------

G_N = symbols('G', positive=True)   # constante de Newton

# --- Escalar de Ricci R = g^mu_nu R_mu_nu ---
R = 0
for mu in range(4):
    for nu in range(4):
        if gij[mu][nu] != 0 and ricci_tensor[mu][nu] != 0:
            R += gij[mu][nu] * ricci_tensor[mu][nu]
R = cancel(R)

# --- Tensor de Einstein G_mu_nu ---
einstein_down = [[cancel(ricci_tensor[mu][nu] - Rational(1, 2) * g_ij[mu][nu] * R)
                  for nu in range(4)]
                 for mu in range(4)]

# --- T^mu_nu = c^4/(8 pi G) g^mu_alpha G_alpha_nu ---
kappa_inv = c**4 / (8 * pi * G_N)
T_mixed = [[factor(cancel(kappa_inv * sum(gij[mu][alpha] * einstein_down[alpha][nu]
                                           for alpha in range(4))))
            for nu in range(4)]
           for mu in range(4)]

# --- Densidad de energía y presiones ---
u   = factor(simplify(-T_mixed[0][0]))
p_r = factor(simplify(T_mixed[1][1]))
p_t = factor(simplify(T_mixed[2][2]))

# --- Combinaciones de la NEC ---
nec_radial     = factor(simplify(u + p_r))
nec_tangencial = factor(simplify(u + p_t))

# -------------------------------------------------------------------
# Modelos concretos a partir de la forma genérica
#
# Si la métrica activa en christoffel.py es la genérica f = 1 - 2 m(r)/r,
# las expresiones anteriores quedan en términos de m(r), m'(r), m''(r).
# Sustituir aquí la función de masa de cada modelo es mucho más rápido
# que calcular el tensor de Einstein con la métrica completa (Bardeen,
# con sus potencias (r^2 + r0^2)^(3/2), puede tardar más de 15 minutos).
# -------------------------------------------------------------------

mass_functions = {
    "Bardeen": m*r**3/(r**2 + r0**2)**Rational(3, 2),
    "Hayward": m*r**3/(r**3 + 2*m*l**2),
}


def substitute_mass(expr, m_expr):
    """Sustituye m(r) por una función de masa concreta y evalúa las derivadas."""
    return factor(simplify(expr.subs(m_regular, m_expr).doit()))


if __name__ == "__main__":
    for name, val in [("u", u), ("p_r", p_r), ("p_t", p_t),
                      ("u + p_r", nec_radial), ("u + p_t", nec_tangencial)]:
        print(f"{name} &= {latex(val)} \\\\")
        print('#'*80)
        print(f"{name} &= {val} \\\\")

    if u.has(m_regular):
        for model, m_expr in mass_functions.items():
            print()
            print('=' * 80)
            print(f"{model}: m(r) = {m_expr}")
            print('=' * 80)
            for name, val in [("u", u), ("p_r", p_r), ("p_t", p_t),
                              ("u + p_r", nec_radial), ("u + p_t", nec_tangencial)]:
                val_model = substitute_mass(val, m_expr)
                print(f"{name} &= {latex(val_model)} \\\\")
                print('#'*80)
                print(f"{name} &= {val_model} \\\\")
            u_model = substitute_mass(u, m_expr)
            print(f"u(0) = {simplify(limit(u_model, r, 0))}")
