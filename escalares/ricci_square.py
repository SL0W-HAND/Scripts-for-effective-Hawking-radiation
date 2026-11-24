from ricci import ricci_tensor
from christoffel import gij
from sympy import factor, simplify, cancel, latex

# -------------------------------------------------------------------
# Cuadrado del tensor de Ricci
#
# Definición:
#   R_ab R^ab = R_ab g^ac g^bd R_cd
#
# donde R^ab se obtiene subiendo los dos índices con la métrica inversa.
#
# Para Schwarzschild (vacío, R_ab = 0) se espera: R_ab R^ab = 0
# -------------------------------------------------------------------

Ric2 = 0

for a in range(4):
    for b in range(4):
        R_ab = ricci_tensor[a][b]
        if R_ab == 0:
            continue
        # Subir los dos índices de la segunda copia
        for c in range(4):
            if gij[a][c] == 0:
                continue
            for d in range(4):
                if gij[b][d] == 0:
                    continue
                R_cd = ricci_tensor[c][d]
                if R_cd == 0:
                    continue
                Ric2 += R_ab * gij[a][c] * gij[b][d] * R_cd

Ric2 = factor(cancel(simplify(Ric2)))

if __name__ == "__main__":
    print(f"R_{{ab}}R^{{ab}} &= {latex(Ric2)} \\\\")
    print('#'*80)
    print(f"R_{{ab}}R^{{ab}} &= {Ric2} \\\\")
