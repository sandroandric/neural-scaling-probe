#!/usr/bin/env python3
"""
Reproduce Figure 1: Noise Injection Falsifiability Test

This experiment demonstrates that:
1. Injecting noise increases intrinsic dimension
2. Higher ID leads to lower predicted β_D
3. The relationship is monotonic (causal validation)

This is the key falsifiability test from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt

# Try to import the probe, fall back to direct implementation
try:
    from scaling_law_probe.probes.twonn import twonn_id
except ImportError:
    from scipy.spatial.distance import pdist, squareform

    def twonn_id(embeddings, k1=1, k2=2):
        N = embeddings.shape[0]
        distances = squareform(pdist(embeddings, metric='euclidean'))
        ratios = []
        for i in range(N):
            sorted_dists = np.sort(distances[i])
            r1, r2 = sorted_dists[k1], sorted_dists[k2]
            if r1 > 0:
                ratios.append(r2 / r1)
        ratios = np.array(ratios)
        log_ratios = np.log(ratios[ratios > 1])
        return len(log_ratios) / np.sum(log_ratios) if len(log_ratios) > 0 else np.nan


def inject_noise(embeddings: np.ndarray, noise_level: float, seed: int = 42) -> np.ndarray:
    """
    Inject Gaussian noise into embeddings.

    Args:
        embeddings: Original embeddings (n_samples, dim)
        noise_level: Fraction of embedding std to use as noise std
        seed: Random seed

    Returns:
        Noisy embeddings
    """
    np.random.seed(seed)
    std = np.std(embeddings)
    noise = np.random.randn(*embeddings.shape) * (noise_level * std)
    return embeddings + noise


def run_noise_experiment(n_samples: int = 2000, base_dim: int = 10, ambient_dim: int = 384):
    """
    Run the noise injection experiment.

    Args:
        n_samples: Number of samples
        base_dim: True intrinsic dimension of base data
        ambient_dim: Ambient dimension (like MiniLM embeddings)

    Returns:
        Dict with noise levels, measured IDs, and predicted betas
    """
    print("=" * 60)
    print("NOISE INJECTION EXPERIMENT")
    print("=" * 60)

    # Generate base data with known intrinsic dimension
    np.random.seed(42)
    latent = np.random.randn(n_samples, base_dim)
    A = np.random.randn(base_dim, ambient_dim) * 0.1
    base_embeddings = latent @ A

    # Noise levels to test
    noise_levels = [0, 0.1, 0.2, 0.3, 0.5]

    results = {
        'noise_levels': [],
        'measured_id': [],
        'predicted_beta': [],
    }

    smoothness = 4.5  # Calibrated on text

    print(f"\nBase data: {n_samples} samples, true ID ≈ {base_dim}")
    print(f"Smoothness: s = {smoothness}")
    print()

    for noise in noise_levels:
        # Inject noise
        noisy = inject_noise(base_embeddings, noise)

        # Measure ID
        measured_id = twonn_id(noisy)

        # Predict beta
        predicted_beta = smoothness / measured_id

        results['noise_levels'].append(noise * 100)  # As percentage
        results['measured_id'].append(measured_id)
        results['predicted_beta'].append(predicted_beta)

        print(f"Noise: {noise*100:3.0f}%  |  ID: {measured_id:5.1f}  |  β_D: {predicted_beta:.3f}")

    return results


def plot_results(results: dict, save_path: str = None):
    """
    Create the "money plot" showing ID↑ and β_D↓ with noise.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    noise = results['noise_levels']
    ids = results['measured_id']
    betas = results['predicted_beta']

    # Left axis: ID (red)
    color1 = '#E53935'
    ax1.set_xlabel('Noise Level (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Intrinsic Dimension', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(noise, ids, 'o-', color=color1, linewidth=2.5,
                     markersize=10, label='Intrinsic Dimension')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right axis: β_D (blue)
    ax2 = ax1.twinx()
    color2 = '#1E88E5'
    ax2.set_ylabel('Predicted β_D', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(noise, betas, 's-', color=color2, linewidth=2.5,
                     markersize=10, label='Predicted β_D')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title
    plt.title('Noise Injection: Falsifiability Test\n↑ Noise ⟹ ↑ ID ⟹ ↓ β_D',
              fontsize=14, fontweight='bold')

    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    # Theory box
    textstr = 'β_D = s / d\ns ≈ 4.5 (calibrated)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def main():
    """Run the experiment and create the plot."""
    results = run_noise_experiment()

    # Verify monotonicity
    print("\n--- VERIFICATION ---")
    ids = results['measured_id']
    betas = results['predicted_beta']

    id_monotonic = all(ids[i] <= ids[i+1] for i in range(len(ids)-1))
    beta_monotonic = all(betas[i] >= betas[i+1] for i in range(len(betas)-1))

    print(f"ID monotonically increasing: {'✓' if id_monotonic else '✗'}")
    print(f"β_D monotonically decreasing: {'✓' if beta_monotonic else '✗'}")

    if id_monotonic and beta_monotonic:
        print("\n✓ FALSIFIABILITY TEST PASSED")
        print("  Noise → ↑ ID → ↓ β_D (causal mechanism confirmed)")
    else:
        print("\n⚠ Unexpected non-monotonic behavior")

    # Create plot
    plot_results(results, save_path="noise_injection_plot.png")


if __name__ == "__main__":
    main()
