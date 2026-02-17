import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np

def generate_data(n_samples, outlier_fraction, outlier_scale=10):
    n_outliers = int(n_samples * outlier_fraction)
    n_regular = n_samples - n_outliers
    
    # Regular data ~ N(0, 1)
    regular_data = np.random.normal(0, 1, n_regular)
    # Outlier data ~ N(0, 10)
    outlier_data = np.random.normal(0, outlier_scale, n_outliers)
    
    return np.concatenate([regular_data, outlier_data])

def nll_loss(params, x):
    # params = [sigma] (assume mu=0 for simplicity)
    sigma = params[0]
    if sigma <= 1e-5: return np.inf
    # Negative Log Likelihood
    return -np.sum(norm.logpdf(x, loc=0, scale=sigma))

def crps_loss(params, x):
    # params = [sigma]
    sigma = params[0]
    if sigma <= 1e-5: return np.inf
    
    # CRPS analytical solution for Gaussian
    # z = (x - mu) / sigma
    z = (x - 0) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Formula: sigma * [ z*(2Phi - 1) + 2phi - 1/sqrt(pi) ]
    term = z * (2 * Phi - 1) + 2 * phi - (1 / np.sqrt(np.pi))
    return np.sum(sigma * term)

# Setup simulation
np.random.seed(42)
outlier_fractions = np.linspace(0, 0.50, 51) # 0% to 50%
results_mle = []
results_crps = []

# Run simulation across fractions
for frac in outlier_fractions:
    data = generate_data(n_samples=2000, outlier_fraction=frac)
    
    # Minimize MLE
    res_mle = minimize(nll_loss, x0=[1.0], args=(data,), bounds=[(0.01, 20)])
    results_mle.append(res_mle.x[0])
    
    # Minimize CRPS
    res_crps = minimize(crps_loss, x0=[1.0], args=(data,), bounds=[(0.01, 20)])
    results_crps.append(res_crps.x[0])

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Sensitivity Analysis
axes[0].plot(outlier_fractions * 100, results_mle, 'o-', label='MLE Sigma', color='#d62728')
axes[0].plot(outlier_fractions * 100, results_crps, 's-', label='CRPS Sigma', color='#1f77b4')
axes[0].axhline(1.0, color='gray', linestyle=':', label='True Main Body Sigma (1.0)')
axes[0].set_xlabel('Percentage of Outliers (%)')
axes[0].set_ylabel('Estimated Sigma')
axes[0].set_title('Sensitivity to Outliers: MLE vs CRPS')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: specific example at 10% outliers to visualize the fit
data_10 = generate_data(n_samples=5000, outlier_fraction=0.10)
res_mle_10 = minimize(nll_loss, x0=[1.0], args=(data_10,), bounds=[(0.01, 20)]).x[0]
res_crps_10 = minimize(crps_loss, x0=[1.0], args=(data_10,), bounds=[(0.01, 20)]).x[0]

x_grid = np.linspace(-15, 15, 1000)
axes[1].hist(data_10, bins=100, density=True, color='gray', alpha=0.3, label='Data (10% outliers)')
axes[1].plot(x_grid, norm.pdf(x_grid, 0, res_mle_10), color='#d62728', lw=2, label=f'MLE Fit ($\sigma$={res_mle_10:.2f})')
axes[1].plot(x_grid, norm.pdf(x_grid, 0, res_crps_10), color='#1f77b4', lw=2, linestyle='--', label=f'CRPS Fit ($\sigma$={res_crps_10:.2f})')
axes[1].set_title('Distribution Fit with 10% Outliers')
axes[1].set_xlim(-15, 15)
axes[1].legend()

plt.tight_layout()
plt.savefig('crps_vs_mle_tails.png')