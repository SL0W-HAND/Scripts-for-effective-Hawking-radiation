from sympy import (symbols, trigsimp, factor, cancel, Function, Matrix,
                   Rational, diff, simplify, latex, init_printing,
                   sin, cos, exp)

# Definición de Símbolos

t, r, theta, phi, Gamma_sym = symbols('t r theta phi Gamma')

indices = [t, r, theta, phi]

# Parámetros y expresiones para la métrica de Kerr
a, m, c, delta_s, sigma_s, rho_s = symbols('a m c Delta Sigma rho', real=True)

# Parámetros adicionales para métricas RBH
l  = symbols('l',   positive=True)   # Hayward: longitud de regularización
r0 = symbols('r_0', positive=True)   # Bardeen: parámetro de regularización
rs = symbols('r_s', positive=True)   # Dymnikova: escala característica

# Expresiones para la métrica (usando _expr para distinguirlas de los símbolos si fuera necesario)
delta = r**2 - 2*m*r + a**2
sigma = (r**2 + a**2)**2 - a**2*sin(theta)**2*delta
rho = r**2 + a**2*cos(theta)**2


# Función para convertir índices a su forma de LaTeX
def latex_index(index_symbol):
    index_dict = {'t': 't', 'r': 'r', 'theta': '\\theta', 'phi': '\\phi'}
    return index_dict.get(str(index_symbol), str(index_symbol))


init_printing(use_unicode=True)

A = Function('A', real=True)(r)
B = Function('B', real=True)(r)

# ======================================================================
# MÉTRICAS DISPONIBLES
# Descomenta el bloque de la métrica que quieras calcular y comenta el
# resto.  La variable 'g_ij' debe quedar definida al final del bloque.
# ======================================================================

# --- SCHWARZSCHILD (vacío, R=0, K=48m²/r⁶) ---
# g_ij = [
#     [-1 + 2*m/r, 0, 0, 0],
#     [0, 1/(1 - 2*m/r), 0, 0],
#     [0, 0, r**2, 0],
#     [0, 0, 0, r**2*sin(theta)**2]
# ]

# --- HAYWARD (activa para prueba) ---
# f_H = 1 - 2*m*r**2/(r**3 + 2*m*l**2)

f_H = 1 - 2*m*r**2 / (r**3 + 2*m*l**2)
g_ij = [
    [-f_H*c**2, 0, 0, 0],
    [0, 1/f_H, 0, 0],
    [0, 0, r**2, 0],
    [0, 0, 0, r**2*sin(theta)**2]
]

# --- BARDEEN ---
# f_B = 1 - 2*m*r**2/(r**2 + r0**2)**Rational(3, 2)

# g_ij = [
#     [-f_B, 0, 0, 0],
#     [0, 1/f_B, 0, 0],
#     [0, 0, r**2, 0],
#     [0, 0, 0, r**2*sin(theta)**2]
# ]

# # --- DYMNIKOVA ---
# f_D = 1 - 2*m*(1 - exp(-r**3/rs**3))/r

# g_ij = [
#     [-f_D, 0, 0, 0],
#     [0, 1/f_D, 0, 0],
#     [0, 0, r**2, 0],
#     [0, 0, 0, r**2*sin(theta)**2]
# ]

# ======================================================================

# Métrica inversa g^ij (tu variable 'gij')
gij = [[Matrix(g_ij).inv()[i, j] for j in range(0, len(g_ij))] for i in range(0, len(g_ij))]


# Función para calcular Símbolos de Christoffel
def Christoffel(upperIndex, lowerIndex1, lowerIndex2, latex_output=True):
    upperIndexSymbol  = indices[upperIndex]
    lowerIndexSymbol1 = indices[lowerIndex1]
    lowerIndexSymbol2 = indices[lowerIndex2]

    Christoffel_sum = 0

    for i in range(0, len(indices), 1):
        term_inv_metric = gij[upperIndex][i]
        diff1 = diff(g_ij[i][lowerIndex2], lowerIndexSymbol1)
        diff2 = diff(g_ij[i][lowerIndex1], lowerIndexSymbol2)
        diff3 = diff(g_ij[lowerIndex1][lowerIndex2], indices[i])
        Christoffel_sum += Rational(1, 2)*term_inv_metric * (diff1 + diff2 - diff3)

    if (Christoffel_sum != 0):
        if (latex_output == True):
            latex_str = (f"\\Gamma^{{{latex_index(upperIndexSymbol)}}}{{ }}"
                         f"_{{{latex_index(lowerIndexSymbol1)} {latex_index(lowerIndexSymbol2)}}} "
                         f"&= {latex(Christoffel_sum)} \\\\")
            print(latex_str)
        else:
            latex_str = (f"\\Gamma^{{{upperIndex}}}{{ }}"
                         f"_{{{lowerIndex1} {lowerIndex2}}} "
                         f"&= {latex(Christoffel_sum)} \\\\")
            print(latex_str)

    return Christoffel_sum


def Christoffel_all(latex_output=False):
    """
    Crea un objeto con indices tipo tensor (que no es un tensor) que
    contiene todos los Christoffel.
    """
    Christoffel_list = []
    for i in range(0, 4, 1):
        i_list = []
        for j in range(0, 4, 1):
            j_list = []
            for k_loop in range(0, 4, 1):
                Christoffel_sym = Christoffel(i, j, k_loop, latex_output=latex_output)
                j_list.append(Christoffel_sym)
            i_list.append(j_list)
        Christoffel_list.append(i_list)
    return Christoffel_list


Christoffel_all()
