"""
Per-dimension KL of the CVAE latent posterior, for every fold (App. B, beta
sensitivity: active latent dimensions).

For each config and fold, the fold's own frozen encoder maps that fold's
validation spectra (the held-out test partition, as built by the evaluation
stage) to mu and log sigma^2, and

    KL_d = 1/2 (E[mu_d^2] + E[sigma_d^2] - E[log sigma_d^2] - 1),

with E the mean over the validation spectra. This is the d-th summand of the
training loss's D_KL (src/models/losses.py::compute_kl_loss) averaged over
spectra, so sum_d KL_d is the mean D_KL on the validation set. A dimension is
active if KL_d > 0.01.

The App. B tables were computed on fold 0 only; ``--check`` verifies that this
script reproduces them (to within one unit of the printed third decimal).

Writes results/latent_kl_per_dim/kl_per_dim.json.

Usage:
    python scripts/latent_kl_per_dim.py --check
    python scripts/latent_kl_per_dim.py --exps exp_608
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline._cvae_gen import build_cvae_config, load_encoder_decoder  # noqa: E402
from pipeline.evaluation_stage import _build_fold_val_sets  # noqa: E402
from src.utils.config_manager import ConfigManager  # noqa: E402

OUT = PROJECT_ROOT / 'results' / 'latent_kl_per_dim' / 'kl_per_dim.json'
ACTIVE_THRESHOLD = 0.01

# Table B.1 (beta sweep) then Table B.2 (latent dimension), with their labels.
EXPS = {'exp_607': '0.25', 'exp_608': '0.50', 'exp_609': '0.75', 'exp_601': '1.00',
        'exp_602': '1.25', 'exp_603': '1.50', 'exp_604': '1.75',
        'exp_605': 'Windowed A', 'exp_606': 'Windowed B',
        'exp_701': '0.50', 'exp_702': '0.50'}

# App. B, fold 0: (active dims, total KL, top-3 KL_d), as printed.
APP_B_FOLD0 = {
    'exp_607': (5, 2.347, (0.964, 0.927, 0.402)), 'exp_608': (3, 1.165, (0.779, 0.298, 0.072)),
    'exp_609': (3, 0.759, (0.599, 0.117, 0.030)), 'exp_601': (3, 0.508, (0.427, 0.046, 0.026)),
    'exp_602': (2, 0.253, (0.223, 0.024, 0.001)), 'exp_603': (1, 0.152, (0.149, 0.0, 0.0)),
    'exp_604': (1, 0.091, (0.089, 0.0, 0.0)), 'exp_605': (3, 0.311, (0.273, 0.023, 0.011)),
    'exp_606': (3, 0.811, (0.633, 0.136, 0.029)), 'exp_701': (3, 1.364, (0.947, 0.310, 0.085)),
    'exp_702': (4, 1.354, (0.926, 0.322, 0.073)),
}


def kl_per_dim(encoder, X, n_features):
    """KL_d over the spectra in X, from the encoder's (z_mean, z_log_var) heads."""
    z_mean, z_log_var = encoder.predict(X.reshape(-1, n_features, 1), batch_size=1024, verbose=0)[:2]
    z_mean, z_log_var = z_mean.astype(np.float64), z_log_var.astype(np.float64)
    return 0.5 * ((z_mean ** 2).mean(0) + np.exp(z_log_var).mean(0) - z_log_var.mean(0) - 1.0)


def summarize(kl):
    top = np.sort(kl)[::-1]
    return {'kl_d': kl.tolist(), 'total_kl': float(kl.sum()),
            'active': int((kl > ACTIVE_THRESHOLD).sum()), 'top3': top[:3].tolist()}


def check_fold0(exp, fold0):
    """App. B to within one unit of its last printed digit (exp_702's second KL_d
    is 0.32147 here, printed 0.322 there)."""
    active, total, top3 = APP_B_FOLD0[exp]
    ok = (fold0['active'] == active and abs(fold0['total_kl'] - total) <= 1e-3
          and all(abs(a - b) <= 1e-3 for a, b in zip(fold0['top3'], top3)))
    if not ok:
        return (f'{exp} fold_0 does not reproduce App. B: '
                f"active={fold0['active']} total={fold0['total_kl']:.3f} "
                f"top3={[round(v, 3) for v in fold0['top3']]} vs {APP_B_FOLD0[exp]}")
    return None


B1_EXPS = ['exp_607', 'exp_608', 'exp_609', 'exp_601', 'exp_602', 'exp_603', 'exp_604',
           'exp_605', 'exp_606']
B2_EXPS = ['exp_701', 'exp_608', 'exp_702']


def latex_table(out, exps, first_col, caption, label):
    """Active-dimension count per fold and total KL as mean (SD) over folds."""
    L = [r'\begin{table}[H]', r'\centering', r'\small',
         r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcc}', r'\toprule',
         first_col + r' & Config & Active, folds 0--4 & Total KL, mean (SD) \\',
         r'\midrule']
    for exp in exps:
        e = out[exp]
        kl = np.array([e['folds'][f]['kl_d'] for f in sorted(e['folds'])])
        total = kl.sum(axis=1)
        active = ', '.join(str(int((r > ACTIVE_THRESHOLD).sum())) for r in kl)
        name = e['label'] if first_col == r'$\beta$' else str(e['latent_dim'])
        L.append(rf"{name} & \texttt{{{exp.replace('_', r'\_')}}} & {active} / {e['latent_dim']} & "
                 rf"{total.mean():.3f} ({total.std(ddof=1):.3f}) \\")
    L += [r'\bottomrule', r'\end{tabular*}', rf'\caption{{\small {caption}}}', rf'\label{{{label}}}',
          r'\end{table}']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exps', nargs='*', default=list(EXPS))
    ap.add_argument('--check', action='store_true', help='assert fold 0 reproduces App. B')
    args = ap.parse_args()

    out = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    folds_by_data, failures = {}, []
    for exp in args.exps:
        config = ConfigManager.load_config(exp).to_dict()
        data_key = (config['paths']['processed_file'], tuple(config['preprocessing']['crop_region']))
        if data_key not in folds_by_data:
            print(f'building validation folds for {data_key[0]}', flush=True)
            folds_by_data[data_key] = _build_fold_val_sets(config)
        folds = folds_by_data[data_key]
        n_features = folds[0]['X'].shape[1]
        cvae_config = build_cvae_config(config, n_features)
        weights_dir = PROJECT_ROOT / config['paths']['models_cvae_dir']

        t0 = time.time()
        per_fold = {}
        for fd in folds:
            encoder, _ = load_encoder_decoder(cvae_config, weights_dir, fd['fold_name'])
            per_fold[fd['fold_name']] = {'n_spectra': int(len(fd['X'])),
                                        **summarize(kl_per_dim(encoder, fd['X'], n_features))}
        if args.check:
            failures += [msg for msg in [check_fold0(exp, per_fold['fold_0'])] if msg]
        out[exp] = {'label': EXPS.get(exp, exp), 'latent_dim': cvae_config['latent_dim'],
                    'weights_dir': config['paths']['models_cvae_dir'],
                    'active_threshold': ACTIVE_THRESHOLD, 'folds': per_fold}
        active = [f['active'] for f in per_fold.values()]
        total = [f['total_kl'] for f in per_fold.values()]
        print(f'{exp}: active per fold {active}, total KL per fold '
              f'{[round(t, 3) for t in total]}  ({time.time() - t0:.0f}s)', flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'wrote {OUT.relative_to(PROJECT_ROOT)}')
    if all(e in out for e in B1_EXPS + B2_EXPS):
        tex = OUT.with_name('kl_per_dim_tables.tex')
        tex.write_text(latex_table(
            out, B1_EXPS, r'$\beta$',
            r'Active latent dimensions ($\mathrm{KL}_d > 0.01$, out of 32) per fold, and total KL '
            r'as mean (SD) over the five folds, by $\beta$; each fold\textquotesingle s encoder is '
            r'evaluated on its own test spectra. Windowed A/B (\texttt{exp\_605}/\texttt{exp\_606}) '
            r'are confounded for comparison, their KL reflects a different training history.',
            'tab:beta_kl_dims') + '\n\n' + latex_table(
            out, B2_EXPS, r'Latent dimension $d$',
            r'Active latent dimensions per fold and total KL as mean (SD) over the five folds vs.\ '
            r'nominal $d$, at fixed $\beta=0.50$.', 'tab:beta_dim_crosscheck') + '\n',
            encoding='utf-8')
        print(f'wrote {tex.relative_to(PROJECT_ROOT)}')
    if failures:
        raise SystemExit('\n'.join(failures))
    if args.check:
        print('fold_0 reproduces App. B for every config checked')


if __name__ == '__main__':
    main()
