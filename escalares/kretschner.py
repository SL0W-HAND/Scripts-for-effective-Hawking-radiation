from riemann import riemann_down, gij
from sympy import simplify, cancel, latex

# -------------------------------------------------------------------
# Escalar de Kretschner (Kretschmann)
#
# Definición:
#   K = R_abcd R^abcd
#
# donde R^abcd se obtiene subiendo los cuatro índices con la métrica
# inversa: R^abcd = g^ae g^bf g^cg g^dh R_efgh
#
# Para Schwarzschild se espera: K = 48 m^2 / r^6
# -------------------------------------------------------------------


K = 0

for a in range(4):
    for b in range(4):
        for c in range(4):
            for d in range(4):
                R_abcd = riemann_down[a][b][c][d]
                if R_abcd == 0:
                    continue
                # Subir los cuatro índices de la segunda copia
                for e in range(4):
                    if gij[a][e] == 0:
                        continue
                    for f in range(4):
                        if gij[b][f] == 0:
                            continue
                        for g in range(4):
                            if gij[c][g] == 0:
                                continue
                            for h in range(4):
                                if gij[d][h] == 0:
                                    continue
                                R_efgh = riemann_down[e][f][g][h]
                                if R_efgh == 0:
                                    continue
                                K += R_abcd * gij[a][e] * gij[b][f] * gij[c][g] * gij[d][h] * R_efgh

K = cancel(simplify(K))

print(f"K &= {latex(K)} \\\\")
print('#'*80)
print(f"K &= {K} \\\\")
