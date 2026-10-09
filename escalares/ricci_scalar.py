from ricci import ricci_tensor
from christoffel import gij
from sympy import factor, simplify, cancel, latex

# -------------------------------------------------------------------
# Escalar de Ricci: R = g^mu_nu R_mu_nu
# Para Schwarzschild (vacío) se espera: R = 0
# -------------------------------------------------------------------

R = 0
for mu in range(4):
    for nu in range(4):
        if gij[mu][nu] != 0 and ricci_tensor[mu][nu] != 0:
            R += gij[mu][nu] * ricci_tensor[mu][nu]

R = cancel(R)
R = factor(R)

print(f"R &= {latex(R)} \\\\")
print('#'*80)
print(f"R &= {R} \\\\")