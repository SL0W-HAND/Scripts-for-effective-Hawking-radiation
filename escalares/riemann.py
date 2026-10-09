from christoffel import Christoffel_all, g_ij, gij, indices
from sympy import diff, factor, simplify, cancel, latex, Rational

# -------------------------------------------------------------------
# Tensor de Riemann con primer índice arriba: R^rho_sigma_mu_nu
#
# Definición:
#   R^rho_sigma_mu_nu = d_mu Gamma^rho_nu_sigma
#                     - d_nu Gamma^rho_mu_sigma
#                     + Gamma^rho_mu_lam Gamma^lam_nu_sigma
#                     - Gamma^rho_nu_lam Gamma^lam_mu_sigma
# -------------------------------------------------------------------

Gamma = Christoffel_all()


def compute_riemann_up():
    R = [[[[0]*4 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    val = (diff(Gamma[rho][nu][sigma], indices[mu])
                           - diff(Gamma[rho][mu][sigma], indices[nu]))
                    for lam in range(4):
                        val += (  Gamma[rho][mu][lam] * Gamma[lam][nu][sigma]
                                - Gamma[rho][nu][lam] * Gamma[lam][mu][sigma])
                    R[rho][sigma][mu][nu] = factor(cancel(val))
    return R


riemann_up = compute_riemann_up()


# -------------------------------------------------------------------
# Bajar el primer índice: R_alpha_beta_gamma_delta = g_alpha_rho R^rho_beta_gamma_delta
# -------------------------------------------------------------------

def compute_riemann_down():
    R = [[[[0]*4 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for alpha in range(4):
        for beta in range(4):
            for gamma in range(4):
                for delta in range(4):
                    val = 0
                    for rho in range(4):
                        if g_ij[alpha][rho] != 0 and riemann_up[rho][beta][gamma][delta] != 0:
                            val += g_ij[alpha][rho] * riemann_up[rho][beta][gamma][delta]
                    R[alpha][beta][gamma][delta] = factor(cancel(val))
    return R


riemann_down = compute_riemann_down()


# -------------------------------------------------------------------
# Imprimir componentes no nulas (índice arriba)
# -------------------------------------------------------------------

def print_riemann():
    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    val = riemann_up[rho][sigma][mu][nu]
                    if val != 0:
                        print(f"R^{{{rho}}}_{{{sigma} {mu} {nu}}} &= {latex(val)} \\\\")


print_riemann()
