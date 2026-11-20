"""
Unified Pipeline Stages

Modular, config-driven pipeline components for the glioma synthetic-data ML
experiments. Each stage is import-safe and also runnable as a CLI
(``python -m pipeline.<stage> --config exp_XXX``):

- TRTR          : train 4 baselines on real data, test on real
- CVAE          : train generative model, generate synthetic spectra
- TSTR          : train 4 baselines on synthetic data, test on real
- TSRTR         : train 4 baselines on real+synthetic (at a target ratio), test on real
- evaluation    : slide-level bootstrap comparison between two regimens
"""

from .trtr_stage import trtr_stage
from .cvae_stage import cvae_stage
from .tstr_stage import tstr_stage
from .tsrtr_stage import tsrtr_stage
from .evaluation_stage import evaluation_stage

__all__ = [
    'trtr_stage',
    'cvae_stage',
    'tstr_stage',
    'tsrtr_stage',
    'evaluation_stage',
]
