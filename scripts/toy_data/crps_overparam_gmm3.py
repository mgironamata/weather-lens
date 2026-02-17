import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np

def generate_data(n_samples, outlier_fraction, outlier_scale=10):
    """
    Generates data from a mixture of 2 Gaussians both centered at 0.
    - Component 1: Main body, N(0, 1)
    - Component 2: Outliers, N(0, outlier_scale)
    """
    n_outliers = int(n_samples * outlier_fraction)
    n_regular = n_samples - n_outliers
    
    regular_data = np.random.normal(0, 1, n_regular)
    outlier_data = np.random.normal(0, outlier_scale, n_outliers)
    
    return np.concatenate([regular_data, outlier_data])

def unpack_params(params):
    # weights: use softmax for 3 components
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
        e_xi_y = diff * (2 * norm.cdf(z) - 1) + 2 * s * norm.pdf(z)
        term1 += w * np.sum(e_xi_y)
    
    # Term 2: E|X - X'| = sum_i sum_j w_i w_j E|X_i - X_j'|
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
outlier_fractions = np.linspace(0.01, 0.10, 10) # 1% to 10%
N_TRIALS = 30
sample_sizes = [1000, 5000, 10000, 50000, 100000]

# Dictionary to store results: {N: (mean_mle, std_mle, mean_crps, std_crps)}
results = {}

# Representative data for the right plot (using a high N case, e.g., 50k)
rep_data = {}

# Initial guess: 3 components slightly different to break symmetry
x0 = np.array([0, 0, 0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

print(f"Starting multi-N simulation ({N_TRIALS} trials per N)...")

for N in sample_sizes:
    print(f"Processing N = {N}...")
    mle_runs = np.zeros((N_TRIALS, len(outlier_fractions)))
    crps_runs = np.zeros((N_TRIALS, len(outlier_fractions)))
    
    for trial in range(N_TRIALS):
        np.random.seed(42 + trial)
        for i, frac in enumerate(outlier_fractions):
            data = generate_data(n_samples=N, outlier_fraction=frac)
            
            # MLE optimization
            res_mle = minimize(nll_loss, x0, args=(data,), method='L-BFGS-B', tol=1e-5)
            w_m, m_m, s_m = unpack_params(res_mle.x)
            mle_runs[trial, i] = get_gmm_sigma(w_m, m_m, s_m)
            
            # CRPS optimization
            res_crps = minimize(crps_loss, x0, args=(data,), method='L-BFGS-B', tol=1e-5)
            w_c, m_c, s_c = unpack_params(res_crps.x)
            crps_runs[trial, i] = get_gmm_sigma(w_c, m_c, s_c)
            
            # Save representative fit for right plot (at 3% outliers, N=100k, robust trial)
            if N == 100000 and i == 2 and crps_runs[trial, i] < 1.3:
                rep_data['data'] = data.copy()
                rep_data['mle_params'] = res_mle.x
                rep_data['crps_params'] = res_crps.x

    results[N] = (np.mean(mle_runs, axis=0), np.std(mle_runs, axis=0),
                  np.mean(crps_runs, axis=0), np.std(crps_runs, axis=0))

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

colors = plt.cm.viridis(np.linspace(0, 0.9, len(sample_sizes)))

# Plot 1: Sensitivity vs N
for i, N in enumerate(sample_sizes):
    m_mean, m_std, c_mean, c_std = results[N]
    
    # MLE lines (dashed)
    axes[0].plot(outlier_fractions * 100, m_mean, '--', color=colors[i], alpha=0.6)
    # CRPS lines (solid)
    axes[0].plot(outlier_fractions * 100, c_mean, '-', color=colors[i], label=f'N={N}')
    # Shading for CRPS (to keep plot clean, only shade CRPS or use low alpha)
    axes[0].fill_between(outlier_fractions * 100, c_mean - c_std, c_mean + c_std, color=colors[i], alpha=0.1)

axes[0].set_xlabel('Percentage of Outliers (%)')
axes[0].set_ylabel('Total Distribution Sigma')
axes[0].set_title('Robustness of CRPS Fit across Sample Sizes (N)')
axes[0].legend(title='Sample Size')
axes[0].grid(True, alpha=0.3)
axes[0].text(0.05, 0.95, 'Solid: CRPS | Dashed: MLE', transform=axes[0].transAxes, verticalalignment='top')

# Plot 2: Detailed fit (Log-y)
if 'data' in rep_data:
    data_p = rep_data['data']
    res_mle_p = rep_data['mle_params']
    res_crps_p = rep_data['crps_params']
    
    x_grid = np.linspace(-30, 30, 1000)
    axes[1].hist(data_p, bins=150, density=True, color='gray', alpha=0.3, label='Data (3% outliers, N=100k)')
    
    w_m, m_m, s_m = unpack_params(res_mle_p)
    pdf_mle = np.zeros_like(x_grid)
    for w, m, s in zip(w_m, m_m, s_m): pdf_mle += w * norm.pdf(x_grid, loc=m, scale=s)
    axes[1].plot(x_grid, pdf_mle, color='#d62728', lw=2, label=f'MLE (std={get_gmm_sigma(w_m, m_m, s_m):.2f})')
    
    w_c, m_c, s_c = unpack_params(res_crps_p)
    pdf_crps = np.zeros_like(x_grid)
    for w, m, s in zip(w_c, m_c, s_c): pdf_crps += w * norm.pdf(x_grid, loc=m, scale=s)
    axes[1].plot(x_grid, pdf_crps, color='#1f77b4', lw=2, ls='--', label=f'CRPS (std={get_gmm_sigma(w_c, m_c, s_c):.2f})')
    
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-5, 1)
    axes[1].set_xlim(-20, 20)
    axes[1].set_title('GMM-3 Distribution Fit (3% Outliers, Log-Y)')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.2)

plt.tight_layout()
plt.savefig(f'crps_vs_mle_N_comparison_{N_TRIALS}.png')
print(f"Successfully generated crps_vs_mle_N_comparison_{N_TRIALS}.png")