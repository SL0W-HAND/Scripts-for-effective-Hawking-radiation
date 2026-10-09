

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.interpolate import AAA

c_val, G_val, M_val = 1.0, 1.0, 1.0     # unidades geometrizadas
E_val, m_val = 1, 0                # energía y masa de la partícula
rs_val = 2.0 * G_val * M_val / c_val**2 # horizonte -> rs_val = 2 (el "valor
                                         # verdadero" que acá conocemos pero
                                         # que el método NO usa como input)

def f_of(r):
    """f(r) = 1 - 2m/r   (factor métrico de Schwarzschild)"""
    return 1.0 - rs_val / r

def f1_of(r):
    """f'(r), derivada elemental -- esto nunca es singular, se puede
    calcular a mano o numéricamente sin ningún cuidado especial."""
    return rs_val / r**2

def W1_of(r):
    """W'(r) donde W(r) = E^2/f(r). ACA está el origen del polo: al
    dividir por f(r)^2, esta cantidad diverge en r=2m."""
    return -E_val**2 * f1_of(r) / f_of(r)**2

def pr_onshell(r0):
    """Condición inicial de p_r en r0, imponiendo la condición de capa
    de masa H = -1/2 (partícula no  masiva, rama entrante)."""
    f0 = f_of(r0)
    arg = (E_val**2 - m_val * f0) / f0**2
    return -np.sqrt(arg) if arg > 0 else 0.0



def sistema_clasico(lam, y):
    r_, p_ = y
    r_dot  = p_ * f_of(r_)
    pr_dot = 0.5*W1_of(r_) - 0.5*f1_of(r_)*p_**2
    return [r_dot, pr_dot]

r0 = 10.0
sol = solve_ivp(sistema_clasico, (0.0, 60.0), [r0, pr_onshell(r0)],
                 method='DOP853', rtol=1e-13, atol=1e-15, max_step=0.05)

r_raw, pr_raw = sol.y[0], sol.y[1]

# Código para generar las 3 gráficas (se añade al final de tu script)

lam = sol.t
r_raw = sol.y[0]
pr_raw = sol.y[1]

rat = AAA(r_raw, pr_raw)
print(rat.poles())
print(rat.residues())
