from sympy import *
import matplotlib.pyplot as plt
import numpy as np

r, m = symbols('r m', real=True, positive=True)

f = 1 - 2*m/r

f_prime = diff(f, r)
f_double_prime = diff(f_prime, r)
f_triple_prime = diff(f_double_prime, r)
inverse_f_prime = diff(1/f, r)

print("f(r) = ", latex(f))
print("f'(r) = ", latex(f_prime))
print("f''(r) = ", latex(f_double_prime))
print("f'''(r) = ", latex(f_triple_prime))
print("1/f'(r) = ", latex(inverse_f_prime))


m_val = 1

ratio_1 = simplify(f_prime)
ratio_2 = simplify(f_double_prime)
ratio_3 = simplify(f_triple_prime)
ratio_4 = simplify(f)
inverse_ratio_1 = simplify(inverse_f_prime)

ratio_1_num = lambdify(r, ratio_1.subs(m, m_val), 'numpy')
ratio_2_num = lambdify(r, ratio_2.subs(m, m_val), 'numpy')
ratio_3_num = lambdify(r, ratio_3.subs(m, m_val), 'numpy')
ratio_4_num = lambdify(r, ratio_4.subs(m, m_val), 'numpy')
inverse_ratio_1_num = lambdify(r, inverse_ratio_1.subs(m, m_val), 'numpy')

r_horizon = 2*m_val
eps = 0.01
r_vals_left = np.linspace(0.01, r_horizon - eps, 250)
r_vals_right = np.linspace(r_horizon + eps, 4*m_val, 250)

plt.figure(figsize=(8, 6))
plt.plot(r_vals_left, ratio_1_num(r_vals_left), color='tab:blue', label="f'")
plt.plot(r_vals_right, ratio_1_num(r_vals_right), color='tab:blue')

plt.plot(r_vals_left, ratio_2_num(r_vals_left), color='tab:orange', label="f''")
plt.plot(r_vals_right, ratio_2_num(r_vals_right), color='tab:orange')

plt.plot(r_vals_left, ratio_3_num(r_vals_left), color='tab:green', label="f'''")
plt.plot(r_vals_right, ratio_3_num(r_vals_right), color='tab:green')

plt.plot(r_vals_left, ratio_4_num(r_vals_left), color='tab:red', label="f")
plt.plot(r_vals_right, ratio_4_num(r_vals_right), color='tab:red')

plt.plot(r_vals_left, inverse_ratio_1_num(r_vals_left), color='tab:purple', label="1/f'")
plt.plot(r_vals_right, inverse_ratio_1_num(r_vals_right), color='tab:purple')

plt.axvline(r_horizon, color='gray', linestyle='--', linewidth=0.8, label='r = 2m (horizon)')

plt.xlabel("r")
plt.ylabel("value")
plt.ylim(-1, 1.5)
plt.title("Ratios of f derivatives to f (m = 1)")
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

