"""Regression tests for pipeline/modal_cvae.py's output-path suffixing
(CVAE MIL's shuffled-bag-label control
needs a full CVAE retrain that must never overwrite the real weights attention MIL's
frozen trunk and CVAE MIL's CbR both depend on). Pure function, no Modal/GPU
needed."""
from pipeline.modal_cvae import _effective_output_dirs


def _config(models_dir='results/exp_501/models/cvae', synthetic_dir='results/exp_501/data/synthetic'):
    return {'paths': {'models_cvae_dir': models_dir, 'synthetic_dir': synthetic_dir}}


def test_no_suffix_returns_the_configs_own_paths_unchanged():
    """The real training run's behaviour must be untouched by this feature."""
    models_dir, synthetic_dir = _effective_output_dirs(_config(), store_suffix=None)
    assert models_dir == 'results/exp_501/models/cvae'
    assert synthetic_dir == 'results/exp_501/data/synthetic'


def test_suffix_appends_and_never_collides_with_the_real_path():
    models_dir, synthetic_dir = _effective_output_dirs(_config(), store_suffix='shuffled')
    assert models_dir == 'results/exp_501/models/cvae_shuffled'
    assert synthetic_dir == 'results/exp_501/data/synthetic_shuffled'
    assert models_dir != _config()['paths']['models_cvae_dir']
    assert synthetic_dir != _config()['paths']['synthetic_dir']


def test_empty_string_suffix_behaves_like_none():
    """An accidental empty-string suffix (e.g. from an unset CLI arg) must
    not silently produce a trailing-underscore path that then subtly
    resolves to something unintended -- falsy-suffix means 'real run'."""
    models_dir, synthetic_dir = _effective_output_dirs(_config(), store_suffix='')
    assert models_dir == 'results/exp_501/models/cvae'
    assert synthetic_dir == 'results/exp_501/data/synthetic'
