"""Tensor-level tests for the SAELens-port JumpReLU/Step autograd Functions.

We verify three properties without training any SAE:

1. Forward outputs match the hard-gate definitions.
2. Step's backward gives a threshold-gradient zero outside |x-theta|<bw/2 and
   non-zero inside, with the correct sign.
3. JumpReLU's backward gives x_grad following the STE mask (x > theta) and a
   threshold_grad that is rectangle-windowed and scaled by -theta/bw.
"""
from __future__ import annotations

import torch

from sae_topology.saes.architectures import JumpReLU, Step, rectangle


def test_rectangle_indicator():
    x = torch.tensor([-1.0, -0.6, -0.5, 0.0, 0.4, 0.5, 0.6, 1.0])
    out = rectangle(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert torch.equal(out, expected)


def test_step_forward_matches_hard_gate():
    x = torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.3, 0.4]], requires_grad=False)
    threshold = torch.tensor([0.25, 0.25, 0.25])
    out = Step.apply(x, threshold, 0.1)
    expected = (x > threshold).float()
    assert torch.equal(out, expected)


def test_step_threshold_grad_outside_window_is_zero():
    bw = 0.1
    # x is far above threshold (|x - theta| = 0.5 >> bw/2 = 0.05)
    x = torch.full((4, 3), 0.6, requires_grad=False)
    threshold = torch.full((3,), 0.1, requires_grad=True)
    out = Step.apply(x, threshold, bw)
    out.sum().backward()
    assert torch.allclose(threshold.grad, torch.zeros_like(threshold))


def test_step_threshold_grad_inside_window_is_correct():
    bw = 0.1
    # |x - theta| = 0.0 < bw/2; rectangle((x-theta)/bw) = 1.0
    # threshold_grad per-batch = -(1/bw) * 1 * grad_output (= 1 from .sum())
    # summed across batch (n=5) = -5 / bw = -50
    n_batch = 5
    x = torch.full((n_batch, 3), 0.1, requires_grad=False)
    threshold = torch.full((3,), 0.1, requires_grad=True)
    out = Step.apply(x, threshold, bw)
    out.sum().backward()
    expected = torch.full((3,), -n_batch / bw)
    assert torch.allclose(threshold.grad, expected)


def test_jumprelu_forward_matches_hard_gate():
    x = torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.3, 0.4]])
    threshold = torch.tensor([0.25, 0.25, 0.25])
    out = JumpReLU.apply(x, threshold, 0.1)
    expected = x * (x > threshold).float()
    assert torch.equal(out, expected)


def test_jumprelu_x_grad_is_ste_masked():
    x = torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.3, 0.4]], requires_grad=True)
    threshold = torch.tensor([0.25, 0.25, 0.25])
    out = JumpReLU.apply(x, threshold, 0.1)
    out.sum().backward()
    # x_grad = (x > threshold) * grad_output (= 1 from .sum())
    expected = (x > threshold).float()
    assert torch.allclose(x.grad, expected)


def test_jumprelu_threshold_grad_outside_window_is_zero():
    bw = 0.1
    x = torch.full((4, 3), 0.6, requires_grad=False)
    threshold = torch.full((3,), 0.1, requires_grad=True)
    out = JumpReLU.apply(x, threshold, bw)
    out.sum().backward()
    assert torch.allclose(threshold.grad, torch.zeros_like(threshold))


def test_jumprelu_threshold_grad_inside_window_is_correct():
    bw = 0.1
    theta_val = 0.1
    n_batch = 5
    # x ≈ theta; rectangle((x-theta)/bw) = 1
    # per-batch contribution: -(theta/bw) * 1 * grad_output (=1) = -theta/bw
    # summed across batch: -n_batch * theta / bw
    x = torch.full((n_batch, 3), theta_val, requires_grad=False)
    threshold = torch.full((3,), theta_val, requires_grad=True)
    out = JumpReLU.apply(x, threshold, bw)
    out.sum().backward()
    expected = torch.full((3,), -n_batch * theta_val / bw)
    assert torch.allclose(threshold.grad, expected)
