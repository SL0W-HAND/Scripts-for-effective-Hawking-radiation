# Scripts de la tesis
[![DOI](https://zenodo.org/badge/1411246351.svg)](https://doi.org/10.5281/zenodo.24518761)

Scripts de cálculo simbólico y numérico de la tesis *Análisis efectivo de la radiación de agujeros negros regulares*. En la tesis, cada script se cita a pie de página por su ruta relativa a este repositorio (por ejemplo, `AAA_method/validity_and_exponents.py`).

## Requisitos

- Python 3.10 o superior
- `sympy` (cálculo simbólico)
- `numpy`, `matplotlib`
- `scipy` 1.15 o superior (se usa `scipy.interpolate.AAA`)

```bash
pip install sympy numpy scipy matplotlib
```

Cada script se ejecuta por separado, por ejemplo `python escalares/ricci_scalar.py`. Los imports entre scripts se resuelven por carpeta. Las figuras (`*.png`) se generan en el directorio desde el que se ejecuta el script y no se incluyen en el repositorio.

## `escalares/`: geometría y curvatura (capítulos 1 y 3)

Cálculo simbólico con SymPy de las cantidades geométricas de métricas esféricamente simétricas (Schwarzschild, Bardeen, Hayward, Dymnikova) y de Kerr. La métrica se elige en `christoffel.py`, descomentando el bloque `g_ij` correspondiente. Los demás scripts la importan de ahí.

| Script | Qué hace |
|---|---|
| `christoffel.py` | Define la métrica $g_{\mu\nu}$ y su inversa, y calcula los símbolos de Christoffel. Es la base de los demás scripts. |
| `riemann.py` | Tensor de Riemann $R^\rho{}_{\sigma\mu\nu}$ y su versión con todos los índices abajo. |
| `ricci.py` | Tensor de Ricci como contracción del Riemann. |
| `ricci_scalar.py` | Escalar de Ricci $R=g^{\mu\nu}R_{\mu\nu}$. |
| `ricci_square.py` | Cuadrado del tensor de Ricci, $R_{\mu\nu}R^{\mu\nu}$. |
| `Kretschmann.py` | Escalar de Kretschmann, $R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}$. |
| `momentum_energy_tensor.py` | Tensor de Einstein de la métrica con función de masa $m(r)$ e identificación con un fluido anisótropo, $T^\mu{}_\nu=\mathrm{diag}(-\rho,P_r,P_t,P_t)$. |
| `inverse.py` | Inversa de la métrica de Kerr, simplificada en términos de $\Delta$, $\rho$ y $\Sigma$. |
| `simplifications.py` | Simplificación de expresiones de la métrica de Kerr en términos de $\Delta$, $\rho$ y $\Sigma$. |
| `simply.py` | Simplificación y formato LaTeX de los escalares de curvatura de Bardeen y Hayward. |

## `metodo_momentos/`: método de momentos (capítulo 4)

| Script | Qué hace |
|---|---|
| `G_poisson_brackets.py` | Corchetes de Poisson $\{G^{a,b},G^{c,d}\}$ en la convención con la posición en el primer índice. Adapta la fórmula de Bojowald et al. (2011), escrita con el momento en el primer índice, mediante el intercambio de índices. Lo usan los demás scripts para construir las ecuaciones de movimiento. |
| `evolution_U.py` | Comprueba la relación de incertidumbre $U=G^{2,0}G^{0,2}-(G^{1,1})^2\ge\hbar^2/4$ y la conservación de $H_Q$ a lo largo de la caída radial en Schwarzschild. |
| `third_order_derivation.py` | Derivación simbólica del sistema de momentos a tercer orden. |
| `third_order_simulation.py` | Integración del sistema a tercer orden (9 variables) con estados gaussianos y no gaussianos, y figura `third_order_cota.png`. |
| `EF_coordinates.py` | Fotones en coordenadas de Eddington–Finkelstein con momentos de segundo orden: las cuatro ramas (regulares y no regulares), conservación de $U$ y figuras `D_fig1`–`D_fig7`. |
| `pole.py` | Polo y residuo de $p_r(r)$ con AAA en Schwarzschild sin correcciones de momentos (validación clásica). |
| `aaa.py` | Versión anterior del procedimiento de detección de polos con AAA. Se conserva como referencia y la sustituye `AAA_method/sch_simple.py`. |
| `ruptura.py` | Estudio exploratorio de la ruptura de la aproximación $G\approx0$ cerca del horizonte y comparación con un ansatz WKB gaussiano. |
| `f_prime.py` | Derivadas simbólicas de $f=1-2m/r$ y de $1/f$, usadas en las ecuaciones de movimiento. |
| `second_order_system.py` | Versión anterior del sistema de segundo orden, con el signo de los corchetes **incorrecto** (sin el intercambio de índices). Se conserva solo como referencia. La versión correcta es `momentos_cambio/second_order_system_sign_flip.py`. |

## `momentos_cambio/`: sistema de segundo orden corregido (capítulo 4)

| Script | Qué hace |
|---|---|
| `second_order_system_sign_flip.py` | Sistema de segundo orden con el signo correcto de los corchetes, en la caída radial en Schwarzschild. Calcula el radio de ruptura, la divergencia de los momentos en $\lambda_c$, la comparación con la trayectoria clásica y la convergencia con el corte numérico. Genera las figuras `sistema2_fig1`–`sistema2_fig5`. |
| `residue_aaa.py` | Datos de las tablas de la sección de residuos con momentos (fotón, $E=1$): barridos en el corte `mom_max` y en `rtol`, que muestran que $\lambda_c$ converge y $r_c$ no, y la estimación del residuo de $p_r(r)$ con AAA. |

## `AAA_method/`: validez del desarrollo y residuos (capítulo 4)

| Script | Qué hace |
|---|---|
| `momentos_eqs.py` | Utilidades comunes: construye $H_Q$ a segundo orden y las ecuaciones de movimiento a partir de un Hamiltoniano clásico (usando `metodo_momentos/G_poisson_brackets.py`), y funciones de selección de polos y suma de residuos con AAA. |
| `sch_simple.py` | Procedimiento para detectar polos numéricos con AAA en trayectorias de `solve_ivp` y clasificarlos (polo simple, espurio o punto de ramificación). |
| `validity_and_exponents.py` | Cociente de validez $q=\sigma_r/(r-r_s)$ a lo largo de la caída para varios paquetes iniciales $G_0$, exponentes de la divergencia en $\lambda_c-\lambda$ con su sensibilidad a la ventana de ajuste, y valores de $r$, $G^{2,0}$, $q$ y $f+\tfrac12f''G^{2,0}$ para distintos cortes. |
| `ef_advanced_residue.py` | Residuo de $p_r(r)$ con AAA en la carta avanzada de Eddington–Finkelstein (rama saliente no regular del fotón, integrada hacia atrás desde $r_0=6$). Sin momentos recupera el residuo exacto, 4. |
| `residue_sweep_G0.py` | Barrido en $G_0$ del residuo estimado con AAA en las dos cartas (Schwarzschild y EF avanzada), con toda la ventana de datos y con la región $q<0.5$. |
