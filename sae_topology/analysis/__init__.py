"""Analysis utilities: plotting helpers for the Mapper + spectral pipeline."""
from .plot import (
    plot_dgp_samples,
    plot_training_curves,
    plot_mapper_graph,
    plot_spectrum_vs_theory,
    plot_topk_phase_transition,
)
from .aggregate import (
    load_results,
    summary_table_figure,
    topk_phase_transition_figure,
    mapper_diagnostic_grid,
    spectrum_overlay_figure,
)

__all__ = [
    "plot_dgp_samples",
    "plot_training_curves",
    "plot_mapper_graph",
    "plot_spectrum_vs_theory",
    "plot_topk_phase_transition",
    "load_results",
    "summary_table_figure",
    "topk_phase_transition_figure",
    "mapper_diagnostic_grid",
    "spectrum_overlay_figure",
]
