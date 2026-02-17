import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

def generate_data(n_samples, outlier_fraction, sigma1=1.0, sigma2=10.0):
    """
    Generates data from a mixture of 2 Gaussians:
    - Component 1: Main body, N(0, sigma1)
    - Component 2: Outliers, N(0, sigma2)
    """
    n_outliers = int(n_samples * outlier_fraction)
    n_regular = n_samples - n_outliers
    
    regular_data = np.random.normal(0, sigma1, n_regular)
    outlier_data = np.random.normal(0, sigma2, n_outliers)
    
    return np.concatenate([regular_data, outlier_data])

def crps_loss_gmm2_landscape(x, w1, m1, s1, w2, m2, s2):
    """
    Calculate CRPS for a 2-component GMM with fixed parameters.
    """
    n = len(x)
    weights = [w1, w2]
    means = [m1, m2]
    sigmas = [s1, s2]
    
    # Term 1: E|X - y| = sum w_i E|X_i - y|
    term1 = 0
    for w, m, s in zip(weights, means, sigmas):
        diff = x - m
        z = diff / s
        e_xi_y = diff * (2 * norm.cdf(z) - 1) + 2 * s * norm.pdf(z)
        term1 += w * np.sum(e_xi_y)
    
    # Term 2: E|X - X'| = sum_i sum_j w_i w_j E|X_i - X_j'|
    term2 = 0
    for i in range(2):
        for j in range(2):
            m_diff = means[i] - means[j]
            s_comb = np.sqrt(sigmas[i]**2 + sigmas[j]**2)
            z_ij = m_diff / s_comb
            e_xi_xj = m_diff * (2 * norm.cdf(z_ij) - 1) + 2 * s_comb * norm.pdf(z_ij)
            term2 += weights[i] * weights[j] * e_xi_xj
            
    return (term1 / n) - 0.5 * term2

def nll_loss_gmm2_landscape(x, w1, m1, s1, w2, m2, s2):
    """
    Calculate negative log-likelihood for a 2-component GMM with fixed parameters.
    """
    pdf = w1 * norm.pdf(x, loc=m1, scale=s1) + w2 * norm.pdf(x, loc=m2, scale=s2)
    pdf = np.maximum(pdf, 1e-15)
    return -np.sum(np.log(pdf))

def unpack_params_floating(params):
    # weights: use sigmoid for 2 components
    w1 = 1.0 / (1.0 + np.exp(-params[0]))
    w2 = 1.0 - w1
    # Means fixed at 0 for this experiment
    means = [0.0, 0.0]
    # sigmas: use exponential to ensure positivity
    sigmas = np.exp(params[1:3])
    return w1, w2, means, sigmas

def objective_fixed_stds(opt_params, x, loss_func, w1, w2):
    log_s1, log_s2 = opt_params
    return loss_func(x, w1, 0.0, np.exp(log_s1), w2, 0.0, np.exp(log_s2))

def objective_floating(params, x, loss_func):
    w1, w2, means, sigmas = unpack_params_floating(params)
    return loss_func(x, w1, means[0], sigmas[0], w2, means[1], sigmas[1])

# parameters for the experiment
N_SAMPLES = 5000
OUTLIER_FRAC = 0.10
S1_DGP = 1.0
S2_DGP = 4.0

# Generate representative data for landscape
np.random.seed(42)
rep_data = generate_data(N_SAMPLES, OUTLIER_FRAC, S1_DGP, S2_DGP)

W1 = 1.0 - OUTLIER_FRAC
W2 = OUTLIER_FRAC

# Fitting trials
N_TRIALS = 40
results = {'crps': [], 'mle': []}

print(f"Running {N_TRIALS} trials (N={N_SAMPLES})...")
for trial in range(N_TRIALS):
    trial_seed = 200 + trial
    np.random.seed(trial_seed)
    data = generate_data(N_SAMPLES, OUTLIER_FRAC, S1_DGP, S2_DGP)
    
    # Noise settings
    INIT_NOISE_STD = 1.0

    # 1. Fixed weights/means optimization
    # Start near true values but with some noise
    x0_fixed = [np.log(S1_DGP) + np.random.normal(0, INIT_NOISE_STD/5), 
                np.log(S2_DGP) + np.random.normal(0, INIT_NOISE_STD)]
    
    for key, loss_func in [('crps', crps_loss_gmm2_landscape), ('mle', nll_loss_gmm2_landscape)]:
        res = minimize(objective_fixed_stds, x0_fixed, args=(data, loss_func, W1, W2))
        results[key].append([np.exp(res.x[0]), np.exp(res.x[1])])

# Convert to arrays
for k in results: results[k] = np.array(results[k])

# Determine plot limits from data points
all_s1 = np.concatenate([results['crps'][:, 0], results['mle'][:, 0]])
all_s2 = np.concatenate([results['crps'][:, 1], results['mle'][:, 1]])

# Add 1.0 (true value) to the range calculation to ensure it's included
s1_min, s1_max = min(np.min(all_s1), S1_DGP), max(np.max(all_s1), S1_DGP)
s2_min, s2_max = min(np.min(all_s2), S2_DGP), max(np.max(all_s2), S2_DGP)

# Add 10% padding
s1_pad = (s1_max - s1_min) * 0.15
s2_pad = (s2_max - s2_min) * 0.15
s1_lims = (max(0.01, s1_min - s1_pad), s1_max + s1_pad)
s2_lims = (max(0.01, s2_min - s2_pad), s2_max + s2_pad)

# Define the grid for sigmas based on these limits
s1_range = np.linspace(s1_lims[0], s1_lims[1], 100)
s2_range = np.linspace(s2_lims[0], s2_lims[1], 100)

S1_GRID, S2_GRID = np.meshgrid(s1_range, s2_range)
CRPS_VALS = np.zeros_like(S1_GRID)
NLL_VALS = np.zeros_like(S1_GRID)

print(f"Calculating CRPS and NLL landscapes over dynamic range: s1={s1_lims}, s2={s2_lims}...")
for i in range(len(s2_range)):
    for j in range(len(s1_range)):
        CRPS_VALS[i, j] = crps_loss_gmm2_landscape(rep_data, W1, 0.0, S1_GRID[i,j], W2, 0.0, S2_GRID[i,j])
        NLL_VALS[i, j] = nll_loss_gmm2_landscape(rep_data, W1, 0.0, S1_GRID[i,j], W2, 0.0, S2_GRID[i,j])

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for idx, (title, vals, label) in enumerate([
    ('CRPS Loss Landscape: $\sigma_1/\sigma_{1,DGP}$ vs $\sigma_2/\sigma_{2,DGP}$', CRPS_VALS, 'CRPS'),
    ('NLL Loss Landscape: $\sigma_1/\sigma_{1,DGP}$ vs $\sigma_2/\sigma_{2,DGP}$', NLL_VALS, 'MLE')
]):
    ax = axes[idx]
    key = label.lower()
    
    # Plot using relative units
    cp = ax.contour(S1_GRID/S1_DGP, S2_GRID/S2_DGP, vals, levels=50, cmap='viridis')
    
    ax.scatter(1.0, 1.0, color='red', marker='x', s=150, lw=3, label='True DGP', zorder=5)
    ax.scatter(results[key][:, 0]/S1_DGP, results[key][:, 1]/S2_DGP, 
               color='white', edgecolors='black', marker='o', s=80, label='Fixed Weights/Means Fits', zorder=4)
    
    ax.set_title(title)
    ax.set_xlabel('Relative $\sigma_1$ ($\sigma_1 / \sigma_{1,ref}$)')
    ax.set_ylabel('Relative $\sigma_2$ ($\sigma_2 / \sigma_{2,ref}$)')
    ax.set_xlim(s1_lims[0]/S1_DGP, s1_lims[1]/S1_DGP)
    ax.set_ylim(s2_lims[0]/S2_DGP, s2_lims[1]/S2_DGP)
    ax.legend(prop={'size': 8})
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()
plt.savefig('crps_vs_mle_std_landscape.png')
print("Successfully generated crps_vs_mle_std_landscape.png")
