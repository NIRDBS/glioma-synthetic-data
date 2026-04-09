"""
exp_102 Unified Pipeline Stages

Modular pipeline components for:
- Preprocessing
- TRTR (Train on Real, Test on Real) baseline training
- CVAE training and synthetic data generation
- TSTR (Train on Synthetic, Test on Real) evaluation
- Statistical evaluation with bootstrap and patient-level stratification
"""

from .preprocessing_stage import preprocessing_stage
from .trtr_stage import trtr_stage
from .cvae_stage import cvae_stage
from .tstr_stage import tstr_stage
from .evaluation_stage import evaluation_stage

__all__ = [
    'preprocessing_stage',
    'trtr_stage',
    'cvae_stage',
    'tstr_stage',
    'evaluation_stage',
]
