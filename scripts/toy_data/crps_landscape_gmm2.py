import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

def generate_data(n_samples, outlier_fraction, outlier_scale=10):
    """
    Generates data from a mixture of 2 Gaussians:
    - Component 1: Main body, N(0, 1)
    - Component 2: Outliers, N(0, outlier_scale)
    """
    n_outliers = int(n_samples * outlier_fraction)
    n_regular = n_samples - n_outliers
    
    regular_data = np.random.normal(0, 1, n_regular)
    outlier_data = np.random.normal(0, outlier_scale, n_outliers)
    
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
    means = params[1:3]
    # sigmas: use exponential to ensure positivity
    sigmas = np.exp(params[3:5])
    return w1, w2, means, sigmas

def objective_fixed_comp2(opt_params, x, loss_func, w1, m1, s1, w2):
    m2, log_s2 = opt_params
    return loss_func(x, w1, m1, s1, w2, m2, np.exp(log_s2))

def objective_fixed_comp1(opt_params, x, loss_func, w2, m2, s2, w1):
    m1, log_s1 = opt_params
    return loss_func(x, w1, m1, np.exp(log_s1), w2, m2, s2)

def objective_floating(params, x, loss_func):
    w1, w2, means, sigmas = unpack_params_floating(params)
    return loss_func(x, w1, means[0], sigmas[0], w2, means[1], sigmas[1])

# parameters for the experiment
N_SAMPLES = 5000
OUTLIER_FRAC = 0.03
OUTLIER_SCALE_DGP = 10.0

# Generate representative data for landscape
np.random.seed(42)
rep_data = generate_data(N_SAMPLES, OUTLIER_FRAC, OUTLIER_SCALE_DGP)

# Fix component parameters for landscapes
W1 = 1.0 - OUTLIER_FRAC
M1 = 0.0
S1 = 1.0
W2 = OUTLIER_FRAC
M2 = 0.0
S2 = OUTLIER_SCALE_DGP

# Define the grids
# Comp 2 Grid (Outliers)
m2_range = np.linspace(-2.5, 2.5, 50)
s2_range = np.linspace(7.5, 12.5, 50)
M2_GRID, S2_GRID = np.meshgrid(m2_range, s2_range)

# Comp 1 Grid (Main Body)
m1_range = np.linspace(-0.5, 0.5, 50)
s1_range = np.linspace(0.5, 1.5, 50)
M1_GRID, S1_GRID = np.meshgrid(m1_range, s1_range)

CRPS_VALS_2 = np.zeros_like(M2_GRID)
NLL_VALS_2 = np.zeros_like(M2_GRID)
CRPS_VALS_1 = np.zeros_like(M1_GRID)
NLL_VALS_1 = np.zeros_like(M1_GRID)

print("Calculating CRPS and NLL landscapes...")
for i in range(50):
    for j in range(50):
        # Component 2 landscape (Comp 1 fixed)
        CRPS_VALS_2[i, j] = crps_loss_gmm2_landscape(rep_data, W1, M1, S1, W2, M2_GRID[i,j], S2_GRID[i,j])
        NLL_VALS_2[i, j] = nll_loss_gmm2_landscape(rep_data, W1, M1, S1, W2, M2_GRID[i,j], S2_GRID[i,j])
        # Component 1 landscape (Comp 2 fixed)
        CRPS_VALS_1[i, j] = crps_loss_gmm2_landscape(rep_data, W1, M1_GRID[i,j], S1_GRID[i,j], W2, M2, S2)
        NLL_VALS_1[i, j] = nll_loss_gmm2_landscape(rep_data, W1, M1_GRID[i,j], S1_GRID[i,j], W2, M2, S2)

# Fitting trials
N_TRIALS = 10
results_fixed_c1 = {'crps': [], 'mle': []}
results_fixed_c2 = {'crps': [], 'mle': []}
results_floating = {'crps': [], 'mle': [], 'comp1': {'crps': [], 'mle': []}}

print(f"Running {N_TRIALS} trials (N={N_SAMPLES})...")
for trial in range(N_TRIALS):
    trial_seed = 100 + trial
    np.random.seed(trial_seed)
    data = generate_data(N_SAMPLES, OUTLIER_FRAC, OUTLIER_SCALE_DGP)
    
    # Noise settings
    INIT_NOISE_STD = 2.0
    WEIGHT_NOISE_STD = 0.5 # Less noise for weights

    # 1. Fixed parameters optimization
    # Start near true values but with lots of noise
    x0_f2 = [np.random.normal(M2, INIT_NOISE_STD), np.log(S2) + np.random.normal(0, INIT_NOISE_STD/2)]
    x0_f1 = [np.random.normal(M1, INIT_NOISE_STD/2), np.log(S1) + np.random.normal(0, INIT_NOISE_STD/2)] 
    
    # Comp 2 optimization (Comp 1 fixed)
    res_crps_f2 = minimize(objective_fixed_comp2, x0_f2, args=(data, crps_loss_gmm2_landscape, W1, M1, S1, W2))
    results_fixed_c2['crps'].append([res_crps_f2.x[0], np.exp(res_crps_f2.x[1])])
    res_mle_f2 = minimize(objective_fixed_comp2, x0_f2, args=(data, nll_loss_gmm2_landscape, W1, M1, S1, W2))
    results_fixed_c2['mle'].append([res_mle_f2.x[0], np.exp(res_mle_f2.x[1])])

    # Comp 1 optimization (Comp 2 fixed)
    res_crps_f1 = minimize(objective_fixed_comp1, x0_f1, args=(data, crps_loss_gmm2_landscape, W2, M2, S2, W1))
    results_fixed_c1['crps'].append([res_crps_f1.x[0], np.exp(res_crps_f1.x[1])])
    res_mle_f1 = minimize(objective_fixed_comp1, x0_f1, args=(data, nll_loss_gmm2_landscape, W2, M2, S2, W1))
    results_fixed_c1['mle'].append([res_mle_f1.x[0], np.exp(res_mle_f1.x[1])])
    
    # 2. Floating parameters optimization
    logit_w1_0 = np.log(W1 / (1.0 - W1))
    x0_floating = [logit_w1_0 + np.random.normal(0, WEIGHT_NOISE_STD), 
                   M1 + np.random.normal(0, INIT_NOISE_STD), 
                   M2 + np.random.normal(0, INIT_NOISE_STD),
                   np.log(S1) + np.random.normal(0, INIT_NOISE_STD), 
                   np.log(S2) + np.random.normal(0, INIT_NOISE_STD)]
    
    for loss_key, loss_func in [('crps', crps_loss_gmm2_landscape), ('mle', nll_loss_gmm2_landscape)]:
        res = minimize(objective_floating, x0_floating, args=(data, loss_func))
        w1_fit, w2_fit, m_fit, s_fit = unpack_params_floating(res.x)
        sort_idx = np.argsort(s_fit) # [small_sigma_idx, large_sigma_idx]
        
        results_floating[loss_key].append([m_fit[sort_idx[1]], s_fit[sort_idx[1]]]) # Outlier component
        results_floating['comp1'][loss_key].append([m_fit[sort_idx[0]], s_fit[sort_idx[0]]]) # Main body component

# Convert to arrays
for d in [results_fixed_c1, results_fixed_c2]:
    for k in d: d[k] = np.array(d[k])
for k in ['crps', 'mle']:
    results_floating[k] = np.array(results_floating[k])
    results_floating['comp1'][k] = np.array(results_floating['comp1'][k])

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Plot Config: (row, col, landscape_vals, grid_m, grid_s, true_m, true_s, fixed_vals, floating_vals, title)
plot_configs = [
    (0, 0, CRPS_VALS_1, M1_GRID, S1_GRID, M1, S1, results_fixed_c1['crps'], results_floating['comp1']['crps'], 'CRPS Landscape: Component 1'),
    (0, 1, NLL_VALS_1, M1_GRID, S1_GRID, M1, S1, results_fixed_c1['mle'], results_floating['comp1']['mle'], 'NLL Landscape: Component 1'),
    (1, 0, CRPS_VALS_2, M2_GRID, S2_GRID, M2, S2, results_fixed_c2['crps'], results_floating['crps'], 'CRPS Landscape: Component 2'),
    (1, 1, NLL_VALS_2, M2_GRID, S2_GRID, M2, S2, results_fixed_c2['mle'], results_floating['mle'], 'NLL Landscape: Component 2')
]

for row, col, vals, gm, gs, tm, ts, fixed, floating, title in plot_configs:
    ax = axes[row, col]
    cp = ax.contourf(gm, gs, vals, levels=50, cmap='viridis')
    fig.colorbar(cp, ax=ax)
    
    ax.scatter(tm, ts, color='red', marker='x', s=150, lw=3, label='True DGP Params', zorder=5)
    ax.scatter(fixed[:, 0], fixed[:, 1], color='white', edgecolors='black', marker='o', s=80, label='Fixed Fits', zorder=4)
    ax.scatter(floating[:, 0], floating[:, 1], color='cyan', edgecolors='black', marker='s', s=80, label='Floating Fits', zorder=4)
    
    ax.set_title(title)
    ax.set_xlabel('Mean ($\mu$)')
    ax.set_ylabel('Std ($\sigma$)')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('crps_vs_mle_landscape_2x2.png')
print("Successfully generated crps_vs_mle_landscape_2x2.png")
