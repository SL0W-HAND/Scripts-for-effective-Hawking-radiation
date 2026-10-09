from riemann import riemann_up
from sympy import cancel, factor, simplify, latex

# Tensor de Ricci como contraccion del tensor de Riemann:
#   R_mu_nu = R^alpha_mu_alpha_nu = sum_alpha riemann_up[alpha][mu][alpha][nu]

ricci_tensor = [[cancel(sum(riemann_up[alpha][mu][alpha][nu] for alpha in range(4)))
                 for nu in range(4)]
                for mu in range(4)]

if __name__ == "__main__":
    for mu in range(4):
        for nu in range(4):
            val = cancel(ricci_tensor[mu][nu])
            val = factor(val)
            print(f"R_{{{mu}{nu}}} &= {latex(simplify(val))} \\\\")
            print('#'*80)
            print(f"R_{{{mu}{nu}}} &= {val} \\\\")
