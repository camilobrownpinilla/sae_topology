"""Stage 0 figures per stage0_tuning.md §8.

Functions
---------
- mapper_grid_heatmap(sweep, expected_betti, out_path) — 6×4 cell grid colored
  by correct/incorrect Betti, with (b_0, b_1) text annotation per cell.
- spectrum_overlay_plot(eigs, theory_ratios, multiplicity_check, out_path) —
  empirical vs. theory eigenvalue ratios on log y, with multiplicity-cluster
  annotations.
- mapper_graph_renderings(graph_by_config, gt_coords, eval_inputs, topology,
                          stable_configs, out_dir) — for every config in the
  stable region, save a PNG rendering of the Mapper graph with the per-
  topology GT coordinate coloring (cyclic θ for S^1 / T^2; RGB for S^2;
  fallback first-eigvec color otherwise). Torus emits two PNGs per config
  (one colored by θ, one by φ) so neither cycle is hidden.

The rendering helper imports `_node_color_for_topology` (private) from
`analysis.aggregate` for consistency with the existing per-run diagnostics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import matplotlib

matplotlib.use('Agg')  # always headless

import matplotlib.pyplot as plt

from sae_topology.analysis.plot import (
    plot_mapper_graph,
    plot_spectrum_vs_theory,
)
from sae_topology.mapper import node_means


# ---------------------------------------------------------------------------
# Mapper grid heatmap
# ---------------------------------------------------------------------------

def mapper_grid_heatmap(
    sweep: dict,
    expected_betti: tuple[int, int],
    out_path: Path | str,
    title: str = '',
) -> Path:
    """Render the (n_intervals × overlap) heatmap. Cell color: green if
    `(b_0, b_1)` matches `expected_betti`, red otherwise. Cell text:
    `(b0, b1)` per stage0_tuning.md §8 item 6.
    """
    out_path = Path(out_path)
    ni_values = sorted({k[0] for k in sweep})
    ov_values = sorted({k[1] for k in sweep})

    cell = np.zeros((len(ov_values), len(ni_values)), dtype=int)
    text = np.empty((len(ov_values), len(ni_values)), dtype=object)
    for (ni, ov), entry in sweep.items():
        i = ov_values.index(ov)
        j = ni_values.index(ni)
        b0, b1 = int(entry['b0']), int(entry['b1'])
        cell[i, j] = 1 if (b0, b1) == tuple(expected_betti) else 0
        text[i, j] = f"({b0},{b1})"

    fig, ax = plt.subplots(
        figsize=(1.0 + 0.9 * len(ni_values), 1.0 + 0.9 * len(ov_values))
    )
    cmap = matplotlib.colors.ListedColormap(['#f4cccc', '#b6d7a8'])
    ax.imshow(cell, cmap=cmap, vmin=0, vmax=1, aspect='auto', origin='lower')
    for i in range(len(ov_values)):
        for j in range(len(ni_values)):
            ax.text(j, i, text[i, j], ha='center', va='center', fontsize=10)
    ax.set_xticks(range(len(ni_values)))
    ax.set_xticklabels([str(n) for n in ni_values])
    ax.set_yticks(range(len(ov_values)))
    ax.set_yticklabels([f"{ov:.2f}" for ov in ov_values])
    ax.set_xlabel('n_intervals')
    ax.set_ylabel('overlap')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Spectrum overlay
# ---------------------------------------------------------------------------

def spectrum_overlay_plot(
    eigs: np.ndarray,
    theory_ratios: Iterable[float],
    multiplicity_check: Optional[dict],
    out_path: Path | str,
    title: str = '',
    K: int = 20,
) -> Path:
    """Empirical vs. theory eigenvalue ratios on log y, with multiplicity-
    cluster annotations (vertical bars at each enforced theory level
    spanning the empirical range, labeled with theory_mult / emp_count).
    """
    out_path = Path(out_path)
    fig = plot_spectrum_vs_theory(eigs, theory_ratios, title=title, K=K)
    ax = fig.axes[0]

    if multiplicity_check is not None:
        for c in multiplicity_check.get('per_cluster', []):
            level = c.get('level')
            tm = c.get('theory_mult')
            ec = c.get('emp_count')
            ok = c.get('match')
            if level is None:
                continue
            ax.axhline(level, color=('green' if ok else 'red'),
                       linestyle=':', alpha=0.5)
            ax.text(K - 1, level,
                    f"  {ec}/{tm}",
                    color=('green' if ok else 'red'),
                    fontsize=8, va='center', ha='left')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Mapper graph renderings (per stable config)
# ---------------------------------------------------------------------------

def _node_color_circle(graph: dict, gt: np.ndarray) -> tuple[np.ndarray, str, str]:
    theta = np.arctan2(gt[:, 1], gt[:, 0])
    return node_means(graph, theta), 'twilight', r'mean $\theta$'


def _node_color_sphere(graph: dict, gt: np.ndarray) -> tuple[np.ndarray, str, str]:
    rgb = (gt + 1.0) / 2.0
    return node_means(graph, rgb), '', r'RGB = (x, y, z)'


def _node_color_torus_theta(graph: dict, gt: np.ndarray) -> tuple[np.ndarray, str, str]:
    theta = np.arctan2(gt[:, 1], gt[:, 0])
    return node_means(graph, theta), 'twilight', r'mean $\theta$'


def _node_color_torus_phi(graph: dict, gt: np.ndarray) -> tuple[np.ndarray, str, str]:
    phi = np.arctan2(gt[:, 3], gt[:, 2])
    return node_means(graph, phi), 'twilight', r'mean $\varphi$'


def _save_one(graph: dict, color, cmap: str, label: str, out_path: Path,
              title: str) -> None:
    if cmap == '':
        # RGB array (sphere): ignore cmap.
        plot_mapper_graph(
            graph, node_color=color, ax=None, title=title,
        )
    else:
        plot_mapper_graph(
            graph, node_color=color, cmap=cmap, colorbar=True,
            title=title,
        )
    fig = plt.gcf()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def mapper_graph_renderings(
    graph_by_config: dict,
    gt: Optional[np.ndarray],
    topology: str,
    stable_configs: Iterable[tuple[int, float]],
    out_dir: Path | str,
) -> list[Path]:
    """Render one PNG per (ni, ov) in `stable_configs` using the per-topology
    GT-coordinate coloring. Torus emits two PNGs per config (theta + phi).

    Returns the list of written PNG paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for (ni, ov) in stable_configs:
        graph = graph_by_config.get((int(ni), float(ov)))
        if graph is None:
            continue
        title_base = f"{topology}: ni={ni}, ov={ov:.2f}"

        if topology == 'circle' and gt is not None:
            color, cmap, label = _node_color_circle(graph, gt)
            out_path = out_dir / f"mapper_ni{ni:03d}_ov{int(ov*100):02d}.png"
            _save_one(graph, color, cmap, label, out_path,
                      title=f"{title_base}  ({label})")
            written.append(out_path)
        elif topology == 'sphere' and gt is not None:
            color, cmap, label = _node_color_sphere(graph, gt)
            out_path = out_dir / f"mapper_ni{ni:03d}_ov{int(ov*100):02d}.png"
            _save_one(graph, color, cmap, label, out_path,
                      title=f"{title_base}  ({label})")
            written.append(out_path)
        elif topology == 'torus' and gt is not None:
            for tag, fn in (('theta', _node_color_torus_theta),
                            ('phi', _node_color_torus_phi)):
                color, cmap, label = fn(graph, gt)
                out_path = out_dir / f"mapper_ni{ni:03d}_ov{int(ov*100):02d}_{tag}.png"
                _save_one(graph, color, cmap, label, out_path,
                          title=f"{title_base}  ({label})")
                written.append(out_path)
        else:
            # No GT: uniform color.
            out_path = out_dir / f"mapper_ni{ni:03d}_ov{int(ov*100):02d}.png"
            plot_mapper_graph(graph, node_color=None, title=title_base)
            fig = plt.gcf()
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            written.append(out_path)

    return written


def stable_configs_from_correct_region(
    sweep: dict, expected_betti: tuple[int, int],
) -> list[tuple[int, float]]:
    """Return the (ni, ov) cells in the largest 4-connected correct block.

    Mirrors `mapper.correct_region` but returns the cells themselves (the
    correct_region helper returns only summary stats). Useful so the
    plotting orchestrator knows which configs to render.
    """
    if not sweep:
        return []
    ni_values = sorted({k[0] for k in sweep})
    ov_values = sorted({k[1] for k in sweep})
    ni_idx = {v: i for i, v in enumerate(ni_values)}
    ov_idx = {v: i for i, v in enumerate(ov_values)}
    correct: set[tuple[int, int]] = set()
    for (ni, ov), v in sweep.items():
        if (v['b0'], v['b1']) == tuple(expected_betti):
            correct.add((ni_idx[ni], ov_idx[ov]))

    visited: set[tuple[int, int]] = set()
    best_size = 0
    best_cells: set[tuple[int, int]] = set()
    for start in correct:
        if start in visited:
            continue
        cells: set[tuple[int, int]] = set()
        stack = [start]
        size = 0
        while stack:
            cell = stack.pop()
            if cell in visited or cell not in correct:
                continue
            visited.add(cell)
            cells.add(cell)
            size += 1
            i, j = cell
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                stack.append((i + di, j + dj))
        if size > best_size:
            best_size = size
            best_cells = cells
    return [(ni_values[i], ov_values[j]) for (i, j) in sorted(best_cells)]
