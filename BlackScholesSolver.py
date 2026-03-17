import numpy as np
import scipy.linalg as la
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import time

time_start = time.time()

# Test Parameters
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

# Black-Scholes Parameters
S_max = 200
N = 10000
M = 10000

dt = T / M
dS = S_max / N

# Grid Initialization

V = np.zeros((N + 1, M + 1))
S = np.linspace(0, S_max, N + 1)
V[:, -1] = np.maximum(S - K, 0) 

# Coefficients for the Tridiagonal Matrix
alpha = 0.25 * dt * (sigma**2 * (S / dS)**2 - r * S / dS)
beta = -0.5 * dt * (sigma**2 * (S / dS)**2 + r)
phi = 0.25 * dt * (sigma**2 * (S / dS)**2 + r * S / dS)

LHS = np.zeros((3, N - 1))
LHS[1, :] = 1 - beta[1:N]  
LHS[2, :-1] = -alpha[2:N]    
LHS[0, 1:] = -phi[1:N-1]

LHS_sparse = diags(
    [1 - beta[1:N], -phi[1:N-1], -alpha[2:N]],
    [0, 1, -1],
    format='csc'  
)

lu = splu(LHS_sparse)

for j in range(M - 1, -1, -1):

    time_now = j * dt
    time_next = (j + 1) * dt

    bound_now = S_max - K * np.exp(-r * (T - time_now))
    bound_next = S_max - K * np.exp(-r * (T - time_next))

    RHS = (alpha[1:N] * V[0:-2, j+1] + 
       (1 + beta[1:N]) * V[1:-1, j+1] + 
       phi[1:N] * V[2:, j+1])
    
    RHS[-1] += phi[N-1] * (bound_now + bound_next)
    RHS[0] += alpha[1] * (0 + 0)

    V[0, j] = 0
    V[N, j] = bound_now

    V[1:-1, j] = lu.solve(RHS)

    

"""
# Plot the option price at t=0 (the first column of V)
plt.figure(figsize=(10, 6))
plt.plot(S, V[:, 0], label='Crank-Nicolson Price (t=0)', color='blue', linewidth=2)

# Plot the payoff at maturity (the last column of V)
plt.plot(S[:-5], V[:-5, 0], 'r--', label='Payoff at Maturity (T)', alpha=0.7)

plt.xlabel('Stock Price ($S$)')
plt.ylabel('Option Price ($V$)')
plt.title('Black-Scholes Option Pricing via Crank-Nicolson')
plt.xlim(0, 190)
plt.ylim(0, 110)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""
time_end = time.time()
print(f"Execution Time: {time_end - time_start:.2f} seconds")