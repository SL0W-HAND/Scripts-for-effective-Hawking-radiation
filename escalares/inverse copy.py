from sympy import symbols, Matrix, sin, cos, simplify, pretty_print, factor, cancel, together, trigsimp, latex

# ------------------------------
# Definición de símbolos
# ------------------------------
t, r, theta, phi = symbols('t r theta phi', real=True)
a, m = symbols('a m', real=True)
# Estas son las variables objetivo para el resultado final
delta_s, rho_s, sigma_s = symbols('delta rho sigma', real=True)

def smart_subs(expr, old, new):
    """Sustituye en expr todas las ocurrencias algebraicamente equivalentes a old por new."""
    return expr.replace(
        lambda subexpr: simplify(subexpr - old) == 0,
        lambda subexpr: new
    )

# ------------------------------
# Definición de las expresiones completas
# ------------------------------
delta_expr = r**2 - 2*m*r + a**2
rho_expr   = r**2 + a**2*cos(theta)**2
sigma_expr = (r**2 + a**2)**2 - a**2*sin(theta)**2*(r**2 - 2*m*r + a**2)

# ------------------------------
# Métrica de Kerr (g_ij)
# ------------------------------
g_ij = Matrix([
    [-(1 - 2*m*r/rho_expr),          0, 0, -2*m*a*r*sin(theta)**2/rho_expr],
    [0,                             rho_expr/delta_expr, 0, 0],
    [0,                             0, rho_expr, 0],
    [-2*m*a*r*sin(theta)**2/rho_expr, 0, 0, sigma_expr*sin(theta)**2/rho_expr]
])

# ------------------------------
# Métrica inversa de Kerr (g^ij)
# ------------------------------
gij_inv = g_ij.inv()

# Sustituciones para variables cortas
subs_dict = {
    sigma_expr: sigma_s,
    delta_expr: delta_s,
    rho_expr: rho_s
}

# Simplificación robusta: varias rutinas en cascada
def full_simplify(expr):
    expr = expr.subs(subs_dict)
    expr = together(expr)
    expr = cancel(expr)
    expr = simplify(expr)
    expr = factor(expr)
    expr = trigsimp(expr)
    return expr

gij_inv = gij_inv.applyfunc(full_simplify)
gij_inv = gij_inv.subs(subs_dict)
gij_inv = gij_inv.applyfunc(lambda e: smart_subs(e, sigma_expr, sigma_s))
gij_inv = gij_inv.applyfunc(lambda e: smart_subs(e, delta_expr, delta_s))
gij_inv = gij_inv.applyfunc(lambda e: smart_subs(e, rho_expr, rho_s))


# ------------------------------
# Resultados
# ------------------------------
print("\nMétrica inversa de Kerr (g^ij) en términos de Δ, ρ y Σ:")
pretty_print(gij_inv)
print("\n")
print(latex(gij_inv))
