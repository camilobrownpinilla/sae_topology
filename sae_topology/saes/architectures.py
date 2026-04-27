"""SAE architecture implementations: ReLU+L1, TopK, JumpReLU.

All three modules share the same interface:
  encode(x)          → sparse codes (n, m)
  decode(z)          → reconstruction (n, d)
  forward(x)         → dict with {recon, codes, mse, ...loss keys}
  normalize_decoder()→ project W_dec rows to unit sphere (in-place, no_grad)
  decoder_atoms      → W_dec as (m, d) numpy array

Architecture design mirrors SAELens exactly (same hyperparameter names and
conventions) so cross-validation against SAELens pretrained SAEs is easy.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── helpers ────────────────────────────────────────────────────────────────


def _init_weights(W_enc: nn.Parameter, W_dec: nn.Parameter) -> None:
    nn.init.kaiming_uniform_(W_enc, a=math.sqrt(5))
    nn.init.kaiming_uniform_(W_dec, a=math.sqrt(5))


# ─── ReLU + L1 ──────────────────────────────────────────────────────────────


class ReluL1SAE(nn.Module):
    """Standard sparse autoencoder with ReLU activations and L1 sparsity penalty.

    Expected to prefer T1: L1 penalises coactivation, driving topology into
    the geometric arrangement of decoder atoms.
    """

    def __init__(self, d_in: int, d_sae: int) -> None:
        super().__init__()
        self.d_in  = d_in
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        _init_weights(self.W_enc, self.W_dec)
        self.normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        pre   = x @ self.W_enc + self.b_enc
        z     = F.relu(pre)
        x_hat = self.decode(z)
        mse   = (x - x_hat).pow(2).mean()
        l1    = z.sum(dim=-1).mean()
        return dict(pre=pre, recon=x_hat, codes=z, mse=mse, l1=l1)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @property
    def decoder_atoms(self) -> np.ndarray:
        return self.W_dec.detach().cpu().numpy()


# ─── TopK ───────────────────────────────────────────────────────────────────


class TopKSAE(nn.Module):
    """Sparse autoencoder with TopK activation — exactly k features per sample.

    Expected to allow T2: TopK permits coactivation patterns that can encode
    topology in the code distribution rather than in the decoder atom geometry.
    """

    def __init__(self, d_in: int, d_sae: int, k: int) -> None:
        super().__init__()
        self.d_in  = d_in
        self.d_sae = d_sae
        self.k     = k

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        _init_weights(self.W_enc, self.W_dec)
        self.normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc           # (n, m)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        pre = x @ self.W_enc + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        x_hat = self.decode(z)
        mse   = (x - x_hat).pow(2).mean()
        l1    = z.sum(dim=-1).mean()                 # reported for monitoring
        return dict(pre=pre, recon=x_hat, codes=z, mse=mse, l1=l1)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @property
    def decoder_atoms(self) -> np.ndarray:
        return self.W_dec.detach().cpu().numpy()


# ─── JumpReLU ───────────────────────────────────────────────────────────────


class JumpReLUSAE(nn.Module):
    """JumpReLU sparse autoencoder with learnable per-feature thresholds.

    Uses a smooth L0 approximation (SAELens V6 / Anthropic tanh variant) so
    that the log-space threshold parameters receive gradients while the hard
    step function is used in the forward pass for reconstruction.

    Gradient split:
      - MSE → updates W_enc, b_enc, W_dec, b_dec  (H treated as constant)
      - smooth L0 → updates log_threshold         (pre_acts detached)
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        bandwidth: float = 0.05,
        init_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_in      = d_in
        self.d_sae     = d_sae
        self.bandwidth = bandwidth

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # log-space keeps θ strictly positive and avoids sign flips
        self.log_threshold = nn.Parameter(
            torch.full((d_sae,), math.log(init_threshold))
        )

        _init_weights(self.W_enc, self.W_dec)
        self.normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        theta = self.log_threshold.exp().detach()    # hard gate, no grad to θ
        return pre * (pre > theta).float()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def _smooth_l0(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Differentiable L0 approximation; gradient flows only to log_threshold."""
        theta = self.log_threshold.exp()
        return torch.sigmoid(
            (pre_acts.detach() - theta) / self.bandwidth
        ).sum(dim=-1).mean()

    def forward(self, x: torch.Tensor) -> dict:
        pre   = x @ self.W_enc + self.b_enc
        theta = self.log_threshold.exp().detach()
        z     = pre * (pre > theta).float()
        x_hat = self.decode(z)
        mse   = (x - x_hat).pow(2).mean()
        l0    = self._smooth_l0(pre)
        l1    = z.sum(dim=-1).mean()                 # reported for monitoring
        return dict(pre=pre, recon=x_hat, codes=z, mse=mse, l0=l0, l1=l1)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @property
    def decoder_atoms(self) -> np.ndarray:
        return self.W_dec.detach().cpu().numpy()


# ─── factory ────────────────────────────────────────────────────────────────


def build_sae(config) -> nn.Module:
    """Construct an SAE module from a SAETrainConfig."""
    arch = config.arch
    if arch == "relu_l1":
        return ReluL1SAE(config.d_in, config.d_sae)
    if arch == "topk":
        return TopKSAE(config.d_in, config.d_sae, config.k)
    if arch == "jumprelu":
        return JumpReLUSAE(
            config.d_in,
            config.d_sae,
            bandwidth=config.jumprelu_bandwidth,
            init_threshold=config.jumprelu_init_threshold,
        )
    raise ValueError(f"Unknown arch '{arch}'. Choose relu_l1, topk, or jumprelu.")
