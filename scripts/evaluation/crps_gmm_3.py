import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np

def generate_data(n_samples, outlier_fraction, outlier_scale=15):
    """
    Generates data from a mixture of 5 Gaussians to make a 3-component fit an approximation.
    This helps observe how CRPS and MLE behave when the model is under-parameterized.
    """
    n_outliers = int(n_samples * outlier_fraction)
    n_regular = n_samples - n_outliers
    
    # 3 "Main" components
    n1 = int(n_regular * 0.4)
    n2 = int(n_regular * 0.3)
    n3 = n_regular - n1 - n2
    
    d1 = np.random.normal(-2, 0.7, n1)
    d2 = np.random.normal(2, 0.7, n2)
    d3 = np.random.normal(0, 1.2, n3)
    
    # 2 "Outlier" components (one far left, one far right)
    n_o1 = n_outliers // 2
    n_o2 = n_outliers - n_o1
    o1 = np.random.normal(-outlier_scale, 2.0, n_o1)
    o2 = np.random.normal(outlier_scale, 2.0, n_o2)
    
    return np.concatenate([d1, d2, d3, o1, o2])

def unpack_params(params):
    # weights: use softmax to ensure sum to 1 and positivity
    logits = params[0:3]
    weights = np.exp(logits - np.max(logits))
    weights /= np.sum(weights)
    
    means = params[3:6]
    # sigmas: use exponential to ensure positivity
    sigmas = np.exp(params[6:9])
    return weights, means, sigmas

def get_gmm_sigma(weights, means, sigmas):
    """Total standard deviation of the mixture."""
    mu_total = np.sum(weights * means)
    var_total = np.sum(weights * (sigmas**2 + means**2)) - mu_total**2
    return np.sqrt(var_total)

def nll_loss(params, x):
    weights, means, sigmas = unpack_params(params)
    pdf = np.zeros_like(x)
    for w, m, s in zip(weights, means, sigmas):
        pdf += w * norm.pdf(x, loc=m, scale=s)
    # Avoid log(0)
    pdf = np.maximum(pdf, 1e-15)
    return -np.sum(np.log(pdf))

def crps_loss(params, x):
    weights, means, sigmas = unpack_params(params)
    n = len(x)
    
    # Term 1: E|X - y| = sum w_i E|X_i - y|
    term1 = 0
    for w, m, s in zip(weights, means, sigmas):
        diff = x - m
        z = diff / s
        # Formula for E|N(m, s^2) - y|
        e_xi_y = diff * (2 * norm.cdf(z) - 1) + 2 * s * norm.pdf(z)
        term1 += w * np.sum(e_xi_y)
    
    # Term 2: E|X - X'| = sum_i sum_j w_i w_j E|X_i - X_j'|
    # X_i - X_j' ~ N(m_i - m_j, s_i^2 + s_j^2)
    term2 = 0
    for i in range(3):
        for j in range(3):
            m_diff = means[i] - means[j]
            s_comb = np.sqrt(sigmas[i]**2 + sigmas[j]**2)
            z_ij = m_diff / s_comb
            e_xi_xj = m_diff * (2 * norm.cdf(z_ij) - 1) + 2 * s_comb * norm.pdf(z_ij)
            term2 += weights[i] * weights[j] * e_xi_xj
            
    return (term1 / n) - 0.5 * term2

# Setup simulation
np.random.seed(42)
outlier_fractions = np.linspace(0, 0.30, 11) # 0% to 30% outliers
results_mle_sigma = []
results_crps_sigma = []
true_std = []

# Initial guess: 3 spread components
# [logits_w, means, log_sigmas]
x0 = np.array([0, 0, 0, -3.0, 0.0, 3.0, 0.0, 0.0, 0.0])

print("Starting simulation across outlier fractions...")
for frac in outlier_fractions:
    data = generate_data(n_samples=1000, outlier_fraction=frac)
    true_std.append(np.std(data))
    
    # Minimize MLE
    res_mle = minimize(nll_loss, x0, args=(data,), method='L-BFGS-B', tol=1e-4)
    w_m, m_m, s_m = unpack_params(res_mle.x)
    results_mle_sigma.append(get_gmm_sigma(w_m, m_m, s_m))
    
    # Minimize CRPS
    res_crps = minimize(crps_loss, x0, args=(data,), method='L-BFGS-B', tol=1e-4)
    w_c, m_c, s_c = unpack_params(res_crps.x)
    results_crps_sigma.append(get_gmm_sigma(w_c, m_c, s_c))
    
    print(f"Frac: {frac*100:4.1f}% | Data Std: {true_std[-1]:5.2f} | MLE GMM-3 Std: {results_mle_sigma[-1]:5.2f} | CRPS GMM-3 Std: {results_crps_sigma[-1]:5.2f}")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Sensitivity Analysis
axes[0].plot(outlier_fractions * 100, results_mle_sigma, 'o-', label='MLE GMM-3 Fit Std', color='#d62728')
axes[0].plot(outlier_fractions * 100, results_crps_sigma, 's-', label='CRPS GMM-3 Fit Std', color='#1f77b4')
axes[0].plot(outlier_fractions * 100, true_std, 'k--', label='True Data Std', alpha=0.6)
axes[0].set_xlabel('Percentage of Outliers (%)')
axes[0].set_ylabel('Estimated Total Distribution Sigma')
axes[0].set_title('Sensitivity of GMM-3 Spread to Outliers')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Detailed fit at 15% outliers
frac_val = 0.15
data_p = generate_data(n_samples=5000, outlier_fraction=frac_val)
res_mle_p = minimize(nll_loss, x0, args=(data_p,), method='L-BFGS-B').x
res_crps_p = minimize(crps_loss, x0, args=(data_p,), method='L-BFGS-B').x

x_grid = np.linspace(-30, 30, 1000)
axes[1].hist(data_p, bins=100, density=True, color='gray', alpha=0.3, label=f'Data ({frac_val*100}% outliers)')

w_m, m_m, s_m = unpack_params(res_mle_p)
pdf_mle = np.zeros_like(x_grid)
for w, m, s in zip(w_m, m_m, s_m): pdf_mle += w * norm.pdf(x_grid, loc=m, scale=s)
axes[1].plot(x_grid, pdf_mle, color='#d62728', lw=2, label=f'MLE Fit (std={get_gmm_sigma(w_m, m_m, s_m):.2f})')

w_c, m_c, s_c = unpack_params(res_crps_p)
pdf_crps = np.zeros_like(x_grid)
for w, m, s in zip(w_c, m_c, s_c): pdf_crps += w * norm.pdf(x_grid, loc=m, scale=s)
axes[1].plot(x_grid, pdf_crps, color='#1f77b4', lw=2, linestyle='--', label=f'CRPS Fit (std={get_gmm_sigma(w_c, m_c, s_c):.2f})')

axes[1].set_title(f'GMM-3 Distribution Fit ({frac_val*100}% Outliers)')
axes[1].set_xlim(-30, 30)
axes[1].legend()

plt.tight_layout()
plt.savefig('crps_vs_mle_gmm3.png')
print("Successfully generated crps_vs_mle_gmm3.png")
