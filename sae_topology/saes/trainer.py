"""SAETrainer: training loop with unit-norm projection and dead-atom resampling."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import SAETrainConfig


class SAETrainer:
    """Train a sparse autoencoder on online samples from a DGP.

    Online sampling: each step draws a fresh batch from dgp.sample(), giving
    infinite non-repeating data and avoiding fixed-dataset overfitting.

    Unit-norm constraint: W_dec is projected to the unit sphere after every
    gradient step.  This is critical — without it, T1 PH conflates geometry
    and scale.

    Dead-atom resampling: atoms whose exponential-moving-average (EMA)
    activation rate falls below dead_threshold are reset to fresh DGP
    directions every resample_every steps.
    """

    LOG_EVERY = 500  # log metrics every this many steps

    def __init__(self, sae: torch.nn.Module, cfg: SAETrainConfig, dgp) -> None:
        self.sae = sae
        self.cfg = cfg
        self.dgp = dgp

        self.opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

        m = cfg.d_sae
        # EMA activation buffer: per-atom activation rate estimate
        self._ema = torch.zeros(m)

        # Metric logs
        self._log_step:    list[int]   = []
        self._log_mse:     list[float] = []
        self._log_loss:    list[float] = []
        self._log_mean_l0: list[float] = []
        self._log_n_dead:  list[int]   = []

    # ── public ──────────────────────────────────────────────────────────────

    def train(self) -> dict:
        """Run training and return atoms + metric logs.

        Returns
        -------
        dict with keys:
          'atoms'   : np.ndarray (m, d) — final decoder weight matrix
          'metrics' : dict of lists — step, mse, loss, mean_l0, n_dead
        """
        sae = self.sae
        cfg = self.cfg
        sae.train()

        for step in range(cfg.n_steps):
            batch = self._get_batch()
            out   = sae(batch)

            loss = self._compute_loss(out)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            sae.normalize_decoder()
            self._update_ema(out['codes'])

            if step > 0 and step % cfg.resample_every == 0:
                self._resample_dead_atoms()

            if step % self.LOG_EVERY == 0:
                self._record(step, out, loss)

        # Final log entry at the last step
        if (cfg.n_steps - 1) % self.LOG_EVERY != 0:
            out_final = sae(self._get_batch())
            loss_final = self._compute_loss(out_final)
            self._record(cfg.n_steps - 1, out_final, loss_final)

        sae.eval()
        return dict(
            atoms=sae.decoder_atoms,
            metrics=dict(
                step=self._log_step,
                mse=self._log_mse,
                loss=self._log_loss,
                mean_l0=self._log_mean_l0,
                n_dead=self._log_n_dead,
            ),
        )

    # ── private ─────────────────────────────────────────────────────────────

    def _get_batch(self) -> torch.Tensor:
        return torch.from_numpy(
            self.dgp.sample(self.cfg.batch_size).astype(np.float32)
        )

    def _compute_loss(self, out: dict) -> torch.Tensor:
        arch = self.cfg.arch
        if arch == "relu_l1":
            return out['mse'] + self.cfg.l1_coeff * out['l1']
        if arch == "topk":
            return out['mse']
        if arch == "jumprelu":
            return out['mse'] + self.cfg.l0_coeff * out['l0']
        raise ValueError(f"Unknown arch '{arch}'")

    @torch.no_grad()
    def _update_ema(self, codes: torch.Tensor) -> None:
        act_rate = (codes > 0).float().mean(dim=0).cpu()
        self._ema = 0.99 * self._ema + 0.01 * act_rate

    @torch.no_grad()
    def _resample_dead_atoms(self) -> None:
        dead_idx = (self._ema < self.cfg.dead_threshold).nonzero(as_tuple=True)[0]
        n_dead = len(dead_idx)
        if n_dead == 0:
            return

        # Draw fresh DGP samples and project to unit sphere
        fresh = self.dgp.sample(max(4 * n_dead, 32)).astype(np.float32)
        fresh_t = F.normalize(torch.from_numpy(fresh), dim=-1)  # (?, d)

        sae = self.sae
        for i, atom_idx in enumerate(dead_idx):
            direction = fresh_t[i % len(fresh_t)]             # unit vector in R^d
            sae.W_dec.data[atom_idx]    = direction
            sae.W_enc.data[:, atom_idx] = direction
            sae.b_enc.data[atom_idx]    = 0.0

        self._ema[dead_idx] = 0.0

    def _record(self, step: int, out: dict, loss: torch.Tensor) -> None:
        codes  = out['codes'].detach()
        n_dead = int((self._ema < self.cfg.dead_threshold).sum().item())
        mean_l0 = float((codes > 0).float().sum(dim=-1).mean().item())

        self._log_step.append(step)
        self._log_mse.append(float(out['mse'].item()))
        self._log_loss.append(float(loss.item()))
        self._log_mean_l0.append(mean_l0)
        self._log_n_dead.append(n_dead)
