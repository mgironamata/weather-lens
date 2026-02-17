import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation, PillowWriter

def generate_data(n_samples, outlier_fraction, sigma1=1.0, sigma2=5.0):
    n_outliers = int(n_samples * outlier_fraction)
    n_regular = n_samples - n_outliers
    regular_data = np.random.normal(0, sigma1, n_regular)
    outlier_data = np.random.normal(0, sigma2, n_outliers)
    return np.concatenate([regular_data, outlier_data])

def crps_loss_gmm2_landscape(x, w1, w2, s1, s2):
    n = len(x)
    weights = [w1, w2]
    means = [0.0, 0.0]
    sigmas = [s1, s2]
    
    term1 = 0
    for w, s in zip(weights, sigmas):
        z = x / s
        e_xi_y = x * (2 * norm.cdf(z) - 1) + 2 * s * norm.pdf(z)
        term1 += w * np.sum(e_xi_y)
    
    term2 = 0
    for i in range(2):
        for j in range(2):
            s_comb = np.sqrt(sigmas[i]**2 + sigmas[j]**2)
            # Since means are 0, E|Xi - Xj'| is simply the expected absolute value of a N(0, s_comb^2)
            # E|N(0, s^2)| = s * sqrt(2/pi)
            e_xi_xj = s_comb * np.sqrt(2 / np.pi)
            term2 += weights[i] * weights[j] * e_xi_xj
            
    return (term1 / n) - 0.5 * term2

def nll_loss_gmm2_landscape(x, w1, w2, s1, s2):
    pdf = w1 * norm.pdf(x, loc=0.0, scale=s1) + w2 * norm.pdf(x, loc=0.0, scale=s2)
    pdf = np.maximum(pdf, 1e-15)
    return -np.sum(np.log(pdf))

def objective_fixed_stds(params, x, loss_func, w1, w2):
    s1, s2 = np.exp(params)
    return loss_func(x, w1, w2, s1, s2)

# Config
N_SAMPLES = 5000
S1_DGP = 1.0
S2_DGP = 5.0
# Start at 0%, increase by 1%, up to 100%
OUTLIER_FRACTIONS = np.arange(0, 1.01, 0.01)
N_TRIALS = 10

# 1. Pre-calculate all trials for all frames to find global limits
print(f"Pre-calculating fits for {len(OUTLIER_FRACTIONS)} frames...")
all_trial_results = [] # List of dicts per frame

for frac in OUTLIER_FRACTIONS:
    w1, w2 = 1.0 - frac, frac
    frame_trials = {'crps': [], 'mle': []}
    
    for t in range(N_TRIALS):
        np.random.seed(100 + t + int(frac*10000))
        data = generate_data(N_SAMPLES, frac, S1_DGP, S2_DGP)
        
        # Consistent initialization near truth
        x0 = [np.log(S1_DGP), np.log(S2_DGP)]
              
        res_m = minimize(objective_fixed_stds, x0, args=(data, nll_loss_gmm2_landscape, w1, w2))
        frame_trials['mle'].append(np.exp(res_m.x))
        
        res_c = minimize(objective_fixed_stds, x0, args=(data, crps_loss_gmm2_landscape, w1, w2))
        frame_trials['crps'].append(np.exp(res_c.x))
    
    all_trial_results.append(frame_trials)

# Determine global limits from all trials
all_s1_fits = []
all_s2_fits = []
for frame in all_trial_results:
    for key in ['crps', 'mle']:
        pts = np.array(frame[key])
        all_s1_fits.extend(pts[:, 0] / S1_DGP)
        all_s2_fits.extend(pts[:, 1] / S2_DGP)

s1_min, s1_max = min(all_s1_fits), max(all_s1_fits)
s2_min, s2_max = min(all_s2_fits), max(all_s2_fits)

# Set specific limits as requested: x from 0 to 2, y from -0.5 to 2.5
s1_lims = [0.0, 2.0]
s2_lims = [-0.5, 2.5]

# Grid for landscapes - cover the full visible extent (using small epsilon for positivity)
s1_grid_range = np.linspace(0.01, 2.0, 50)
s2_grid_range = np.linspace(-0.5, 2.5, 60)
S1_REL, S2_REL = np.meshgrid(s1_grid_range, s2_grid_range)

# 2. Pre-calculate all landscapes to find global scales for fixed colormaps
print(f"Pre-calculating landscapes for {len(OUTLIER_FRACTIONS)} frames...")
all_crps_landscapes = []
all_nll_landscapes = []

for frac in OUTLIER_FRACTIONS:
    w1, w2 = 1.0 - frac, frac
    np.random.seed(42)
    rep_data = generate_data(N_SAMPLES, frac, S1_DGP, S2_DGP)
    
    # Vectorized computation of landscapes (as much as possible)
    crps_vals = np.zeros_like(S1_REL)
    nll_vals = np.zeros_like(S1_REL)
    
    for i in range(len(s2_grid_range)):
        for j in range(len(s1_grid_range)):
            s1 = S1_REL[i,j] * S1_DGP
            s2 = max(0.01, S2_REL[i,j] * S2_DGP) # Clamp for calculation
            crps_vals[i, j] = crps_loss_gmm2_landscape(rep_data, w1, w2, s1, s2)
            nll_vals[i, j] = nll_loss_gmm2_landscape(rep_data, w1, w2, s1, s2)
            
    all_crps_landscapes.append(crps_vals)
    all_nll_landscapes.append(nll_vals)

all_crps_landscapes = np.array(all_crps_landscapes)
all_nll_landscapes = np.array(all_nll_landscapes)

# 3. Transform to relative values (ratio to local minimum of each frame)
print("Transforming landscapes to relative values (ratios to local minimum)...")
rel_crps_landscapes = all_crps_landscapes / np.min(all_crps_landscapes, axis=(1, 2), keepdims=True)
rel_nll_landscapes = all_nll_landscapes / np.min(all_nll_landscapes, axis=(1, 2), keepdims=True)

# Define global levels for consistent color scale
# We use percentiles to determine the range to avoid being washed out by spikes
# Since we want a single shared scale, we pick a representative upper bound for both
crps_rel_max = np.percentile(rel_crps_landscapes, 95)
nll_rel_max = np.percentile(rel_nll_landscapes, 85) 
common_rel_max = max(crps_rel_max, nll_rel_max)

common_levels = np.linspace(1.0, common_rel_max, 100)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
plt.subplots_adjust(right=0.85) # Make room for colorbar

# Pre-create the colorbar axis to keep it static
cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])

def update(frame_idx):
    frac = OUTLIER_FRACTIONS[frame_idx]
    
    for ax in axes: ax.clear()
    
    titles = [f'CRPS Rel. Landscape ($Loss/Loss_{{min}}$)\nOutliers: {frac*100:.2f}%', 
              f'NLL Rel. Landscape ($Loss/Loss_{{min}}$)\nOutliers: {frac*100:.2f}%']
    data_keys = ['crps', 'mle']
    landscapes = [rel_crps_landscapes[frame_idx], rel_nll_landscapes[frame_idx]]
    
    last_cf = None
    for i in range(2):
        ax = axes[i]
        # Dense Color Map using common relative ratios and shared scale
        cf = ax.contourf(S1_REL, S2_REL, landscapes[i], levels=common_levels, cmap='viridis', extend='both')
        last_cf = cf
        
        # True value
        ax.scatter(1.0, 1.0, color='red', marker='x', s=100, label='True DGP', zorder=10)
        # Trials
        trial_results = all_trial_results[frame_idx]
        pts = np.array(trial_results[data_keys[i]])
        ax.scatter(pts[:, 0]/S1_DGP, pts[:, 1]/S2_DGP, 
                   facecolor='white', edgecolor='black', s=50, label='Fits', zorder=11)
                   
        ax.set_title(titles[i])
        ax.set_xlabel('$\sigma_1 / \sigma_{1,true}$')
        ax.set_ylabel('$\sigma_2 / \sigma_{2,true}$')
        ax.set_xlim(s1_lims[0], s1_lims[1])
        ax.set_ylim(s2_lims[0], s2_lims[1])
        ax.grid(alpha=0.3, color='white', linestyle=':', lw=0.5)
        ax.legend(loc='upper right', fontsize=8)

    # Update colorbar only if it doesn't exist (static scale)
    if not hasattr(update, "cbar"):
        update.cbar = fig.colorbar(last_cf, cax=cax)
        update.cbar.set_label('Ratio to Minimum Loss ($Loss / Loss_{min}$)')

    return axes

ani = FuncAnimation(fig, update, frames=len(OUTLIER_FRACTIONS), blit=False)
ani.save('crps_vs_mle_landscape_evolution.gif', writer=PillowWriter(fps=5))
print("Successfully generated crps_vs_mle_landscape_evolution.gif")
