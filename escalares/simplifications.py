# -*- coding: utf-8 -*-
#
# Este script utiliza la biblioteca SymPy para simplificar expresiones matemáticas.
# SymPy es una biblioteca de Python para matemáticas simbólicas.
# Si no la tienes instalada, puedes hacerlo con el siguiente comando:
# pip install sympy

import sympy as sp

# Definir los símbolos y parámetros
t, r, theta, phi = sp.symbols('t r theta phi')
a, m, c = sp.symbols('a m c', real=True)
Delta_s, Sigma_s, rho_s = sp.symbols('Delta Sigma rho', real=True)

# Definir los componentes clave de la métrica de Kerr
delta_expr = r**2 - 2*m*r + a**2
rho_expr = r**2 + a**2*sp.cos(theta)**2
sigma_expr = (r**2 + a**2)**2 - a**2*sp.sin(theta)**2*delta_expr

# Las coordenadas se usan para la diferenciación
indices = [t, r, theta, phi]

# Métrica de Kerr (g_ij)
g_ij = sp.Matrix([
    [-(1 - 2*m*r/(rho_expr)), 0, 0, -2*m*c*a*r*(sp.sin(theta)**2)/(rho_expr)],
    [0, rho_expr/delta_expr, 0, 0],
    [0, 0, rho_expr, 0],
    [-2*m*c*a*r*(sp.sin(theta)**2)/(rho_expr), 0, 0, sigma_expr*(sp.sin(theta)**2)/(rho_expr)]
])

# Métrica inversa g^ij
gij = g_ij.inv()

def calcular_christoffel(upper_idx, lower_idx1, lower_idx2):
    """
    Calcula un solo símbolo de Christoffel dado sus índices.
    Los índices son 0-basados (0=t, 1=r, 2=theta, 3=phi).
    """
    christoffel_sum = 0
    # Iterar sobre el índice de suma
    for i in range(len(indices)):
        term_inv_metric = gij[upper_idx, i]
        
        diff1 = sp.diff(g_ij[i, lower_idx2], indices[lower_idx1])
        diff2 = sp.diff(g_ij[i, lower_idx1], indices[lower_idx2])
        diff3 = sp.diff(g_ij[lower_idx1, lower_idx2], indices[i])
        
        christoffel_sum += sp.Rational(1,2) * term_inv_metric * (diff1 + diff2 - diff3)
        
    return christoffel_sum

def simplificar_expresiones():
    """
    Calcula y simplifica un símbolo de Christoffel.
    """
    print("--- Calculando y simplificando el símbolo de Christoffel Gamma^t_tr ---")
    
    # Calcular la expresión completa del símbolo Gamma^t_tr
    # Los índices 0, 1, 1 corresponden a (t, r, r) en SymPy.
    christoffel_expr = calcular_christoffel(0, 1, 1)

    print("\n--- Expresión Original (calculada) ---")
    print(christoffel_expr)

    # Reemplazar las expresiones por sus símbolos simplificados
    expr_reemplazada = christoffel_expr.subs({
        delta_expr: Delta_s,
        sigma_expr: Sigma_s,
        rho_expr: rho_s
    })

    # Simplificar la expresión final
    expresion_simplificada = sp.simplify(expr_reemplazada)

    print("\n--- Expresión Simplificada ---")
    print(expresion_simplificada)
    print("\n----------------------------------")

if __name__ == "__main__":
    simplificar_expresiones()