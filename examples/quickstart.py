#!/usr/bin/env python3
"""
Quickstart: Predict scaling laws in 10 minutes.

This example shows how to:
1. Load text data
2. Measure intrinsic dimension
3. Predict the scaling exponent β_D
4. Interpret the results
"""

import numpy as np


def generate_sample_texts(n_samples: int = 1000) -> list:
    """Generate sample texts for demo (replace with your data)."""
    # Simulate different text types
    np.random.seed(42)

    templates = [
        "The {adj} {noun} {verb} the {noun2}.",
        "In {year}, researchers discovered that {noun} could {verb}.",
        "A {adj} approach to {noun} involves {verb}ing the {noun2}.",
        "The relationship between {noun} and {noun2} is {adj}.",
    ]

    adjectives = ["novel", "efficient", "robust", "scalable", "modern", "classical"]
    nouns = ["model", "algorithm", "system", "method", "approach", "framework"]
    verbs = ["improve", "enhance", "optimize", "transform", "analyze", "predict"]
    years = ["2020", "2021", "2022", "2023", "2024"]

    texts = []
    for _ in range(n_samples):
        template = np.random.choice(templates)
        text = template.format(
            adj=np.random.choice(adjectives),
            noun=np.random.choice(nouns),
            noun2=np.random.choice(nouns),
            verb=np.random.choice(verbs),
            year=np.random.choice(years),
        )
        texts.append(text)

    return texts


def main():
    print("=" * 60)
    print("Neural Scaling Probe - Quickstart")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1] Loading sample texts...")
    texts = generate_sample_texts(n_samples=1000)
    print(f"    Loaded {len(texts)} texts")
    print(f"    Example: '{texts[0]}'")

    # Step 2: Import and create probe
    print("\n[2] Creating scaling probe...")
    try:
        from scaling_law_probe import predict_beta, ScalingProbe

        # Quick one-liner
        print("\n[3] Predicting scaling exponent...")
        beta, id_est = predict_beta(texts, encoder="minilm", n_samples=1000)

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"  Intrinsic Dimension (ID): {id_est:.2f}")
        print(f"  Predicted β_D:            {beta:.3f}")

        # Interpretation
        print(f"\n{'='*60}")
        print("INTERPRETATION")
        print(f"{'='*60}")
        if beta > 0.45:
            print("  → Highly structured data")
            print("  → Expected: FAST scaling with more data")
            print("  → Similar to: Code")
        elif beta > 0.35:
            print("  → Moderately structured data")
            print("  → Expected: MODERATE scaling with more data")
            print("  → Similar to: General text")
        elif beta > 0.25:
            print("  → Less structured data")
            print("  → Expected: SLOWER scaling with more data")
            print("  → Similar to: Scientific/specialized text")
        else:
            print("  → High-dimensional data")
            print("  → Scaling may be slow")
            print("  → Consider: Data curation or filtering")

        print(f"\n{'='*60}")
        print("REFERENCE VALUES")
        print(f"{'='*60}")
        print("  Code:       β_D ≈ 0.45")
        print("  Text:       β_D ≈ 0.34 (Chinchilla)")
        print("  Scientific: β_D ≈ 0.32")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo install the package:")
        print("  pip install scaling-law-probe")
        print("\nOr from source:")
        print("  pip install -e .")


if __name__ == "__main__":
    main()
