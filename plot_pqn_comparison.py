import numpy as np
import matplotlib.pyplot as plt

"""
Plotting script to compare average returns for this PQN implementation to the original
"""

def process_metrics(filename):
    data = np.load(filename)
    steps = data['env_step']  # Shape (30, 244)
    returns = data['moving_avg_return']  # Shape (30, 244)

    n_seeds = returns.shape[0]  # n=30

    # Calculate statistics across seeds (axis 0)
    avg_steps = np.mean(steps, axis=0)
    avg_returns = np.mean(returns, axis=0)
    std_returns = np.std(returns, axis=0)

    # 95% Confidence Interval Calculation
    # CI = 1.96 * (Standard Deviation / sqrt(n))
    ci_95 = 1.96 * (std_returns / np.sqrt(n_seeds))

    return avg_steps, avg_returns, ci_95


# Load data for both implementations
steps_my, returns_my, ci_my = process_metrics('data/pqn_original_cartpole_default_params.npz')
steps_cp, returns_cp, ci_cp = process_metrics('data/pqn_cartpole_metrics.npz')

# Plotting
plt.figure(figsize=(10, 6))

# Plot My Implementation
plt.plot(steps_my, returns_my, label='Seth PQN Implementation', color='blue', linewidth=2)
plt.fill_between(steps_my, returns_my - ci_my, returns_my + ci_my, color='blue', alpha=0.2)

# Plot Cartpole Reference
plt.plot(steps_cp, returns_cp, label='Original PQN Implementation', color='red', linewidth=2)
plt.fill_between(steps_cp, returns_cp - ci_cp, returns_cp + ci_cp, color='red', alpha=0.2)

plt.xlabel('Timesteps')
plt.ylabel('Average Return')
plt.title('Average Return (95% CI)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('pqn_original_cartpole_default_params.png')
