from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SAETrainConfig:
    """Training configuration for a single SAE run.

    All architecture-specific fields are present on every config; fields not
    relevant to the chosen arch are simply ignored during training.
    """

    arch: str       # "relu_l1" | "topk" | "jumprelu"
    d_in: int       # ambient dimension of the input data
    d_sae: int      # number of atoms / SAE width  (m)

    # ── ReLU+L1 ────────────────────────────────────────────────────────────
    # 2e-2 calibrated for d=8, sigma=0.05, m=64 circle; gives L0≈9 and score≈1.0
    l1_coeff: float = 2e-2

    # ── TopK ───────────────────────────────────────────────────────────────
    k: int = 9      # features kept per sample; calibrate to match ReLU+L1 L0
    # Optional sweep over k values for the H1 phase-transition test
    # (spec section 6, line 217). If set, the runner trains a separate TopK
    # SAE for each k in the list, holding d_sae fixed.
    K_sweep: Optional[List[int]] = None

    # ── JumpReLU ───────────────────────────────────────────────────────────
    l0_coeff: float = 1e-3
    jumprelu_bandwidth: float = 0.05       # κ: width of the smoothed step
    jumprelu_init_threshold: float = 0.01  # initial θ value

    # ── Optimiser ──────────────────────────────────────────────────────────
    lr: float = 4e-4
    n_steps: int = 30_000
    batch_size: int = 256
    seed: int = 0

    # ── Dead-atom resampling ────────────────────────────────────────────────
    resample_every: int = 2_500   # steps between resampling passes
    dead_threshold: float = 1e-3  # EMA activation rate below this → dead

    # ── Evaluation ─────────────────────────────────────────────────────────
    pca_dim: int = 4             # T2 PCA dimension (clamped to d_sae - 1)
    n_eval_samples: int = 2_000  # fresh samples used for T1/T2 scoring
