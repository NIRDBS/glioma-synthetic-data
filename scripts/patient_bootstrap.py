"""
Patient-level (cluster) stratified bootstrap: the alternative statistical
evaluation for every CI-bearing table in the paper.

Resamples PATIENTS with replacement within each class and takes every slide of
each drawn patient (src/utils/statistical_evaluation.py, SECTION 1a). Estimands
and point estimates are unchanged; only the resampling unit changes.

Everything is written under results/patient_bootstrap/ -- no slide-level result
file is touched, and nothing the paper generators or run_suite glob can see it:

  cache/<key>.npz                       bag decisions (bag_ids, bag_pred, bag_true)
  <exp>/regimen_TRTR_vs_<CAND>.json     regimen Delta CIs + CbR CI (stats_eval schema)
  pairwise_latent_dim.json              latent-dim pairwise tests (B7 schema)
  pairwise_pool_size.json               pool-size pairwise tests (B8 schema)
  <exp>/spatial_paired.json             spatial real-vs-shuffled tests (B11 schema)
  <exp>/key_instance_paired.json        key-instance tests (B10 schema)
  manifest_before.json / manifest_after.json   SHA-256 of the protected slide results

Anchor checks (abort on mismatch): before any patient-level number is produced,
the ORIGINAL slide-level bootstrap is re-run on the same inputs and must
reproduce the stored slide-level CIs exactly -- proving these inputs are the
data behind the published numbers.

Usage:
    python scripts/patient_bootstrap.py --stage all
    python scripts/patient_bootstrap.py --stage cache --exps exp_608
    python scripts/patient_bootstrap.py --stage boot
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)   # model dirs in the stats JSONs are repo-relative

from src.utils.statistical_evaluation import (  # noqa: E402
    bag_decisions, bag_level_bootstrap, bag_level_bootstrap_cbr,
    bag_level_patient_bootstrap, bag_level_patient_bootstrap_cbr,
    paired_cluster_test, patient_ids_from_slides, summarize_bootstrap_diffs,
)

RESULTS = PROJECT_ROOT / 'results'
OUT_ROOT = RESULTS / 'patient_bootstrap'
CACHE = OUT_ROOT / 'cache'

# Experiments whose stats-eval tables carry a CI or star in the paper: the
# per-experiment appendix tables (AXIS_GROUPS) incl. the beta table (607-609).
REGIMEN_EXPS = ['exp_501', 'exp_601', 'exp_607', 'exp_608', 'exp_609',
                'exp_701', 'exp_702', 'exp_703', 'exp_704', 'exp_705', 'exp_706',
                'exp_707', 'exp_708',
                'exp_4.608', 'exp_2.607', 'exp_2.608', 'exp_2.609']
CANDIDATES = ('TSTR', 'TSRTR')
MIL_EXPS = ['exp_607', 'exp_608', 'exp_609', 'exp_2.607', 'exp_2.608', 'exp_4.608']
LATENT_BOOT_JSON = RESULTS / 'latent_dim_bootstrap' / 'pairwise_latent_dim_bootstrap.json'
POOL_BOOT_JSON = RESULTS / 'synth_pool_bootstrap' / 'pairwise_pool_size_bootstrap.json'

N_BOOT_PAIRED = 10000
SEED = 42
TOL = 1e-12

# Slide-level results that must be byte-identical before and after a run.
PROTECTED_GLOBS = ['results/*/statistical_eval/**/*', 'results/latent_dim_bootstrap/**/*',
                   'results/synth_pool_bootstrap/**/*', 'results/*/spatial_validation/*.json',
                   'results/*/mil_eval/**/*.json']

LABEL_PREFIX = {'Apollo': 'apollo', 'RF_Standard': 'rf', 'XGBoost': 'xgb', '1D_CNN': 'cnn'}
REGIMEN_CODE = {'TS/TR': 'TSTR', 'TSR/TR': 'TSRTR'}


# --------------------------------------------------------------------------- io

def _guarded(path):
    """Every write goes through here: refuse anything outside OUT_ROOT."""
    p = Path(path).resolve()
    if OUT_ROOT.resolve() not in p.parents:
        raise RuntimeError(f'refusing to write outside {OUT_ROOT}: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path, obj):
    p = _guarded(path)
    p.write_text(json.dumps(obj, indent=1, default=_json_default), encoding='utf-8')
    print(f'  wrote {p.relative_to(PROJECT_ROOT)}', flush=True)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def manifest():
    """SHA-256 of every protected slide-level result file."""
    out = {}
    for pat in PROTECTED_GLOBS:
        for f in glob.glob(pat, recursive=True):
            p = Path(f)
            if p.is_file() and 'patient_bootstrap' not in p.parts:
                out[p.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


def latest_stats_json(exp, cand):
    """Same selection rule as paper/gen_appendix_controlled_experiments.py::latest()."""
    pat = str(RESULTS / exp / 'statistical_eval' / f'statistical_evaluation_TRTR_vs_{cand}_*.json')
    files = sorted(glob.glob(pat), key=lambda f: re.search(r'(\d{8}_\d{6})', f).group(1))
    if not files:
        return None, None
    return Path(files[-1]), json.loads(Path(files[-1]).read_text(encoding='utf-8'))


def _load_module(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close(a, b):
    return abs(float(a) - float(b)) <= TOL


def _check(cond, msg):
    if not cond:
        raise AssertionError(f'ANCHOR MISMATCH: {msg}')


# ------------------------------------------------------------------------ cache

def _data_signature(folds):
    h = hashlib.sha1()
    for fd in folds:
        h.update(fd['fold_name'].encode())
        h.update(np.ascontiguousarray(fd['X']).tobytes())
        h.update(np.ascontiguousarray(fd['y']).tobytes())
        h.update('\0'.join(map(str, fd['slide_ids'])).encode())
    return h.hexdigest()


def _cache_file(key):
    return CACHE / f'{hashlib.sha1(key.encode()).hexdigest()[:20]}.npz'


def _cache_index_update(key, path, meta):
    idx_path = CACHE / 'index.json'
    idx = json.loads(idx_path.read_text(encoding='utf-8')) if idx_path.exists() else {}
    idx[path.name] = {'key': key, **meta}
    _guarded(idx_path).write_text(json.dumps(idx, indent=1, sort_keys=True), encoding='utf-8')


def cached_bags(folds, data_sig, model_dir, prefix, config=None):
    """Bag decisions (bag_ids, bag_pred, bag_true) for one model set on these
    folds; ``prefix='cbr'`` = Classification by Reconstruction with the CVAE in
    ``model_dir``. Computed once, then loaded from the cache."""
    from pipeline.evaluation_stage import MODEL_SPECS, _cbr_predictions, _gather_predictions

    model_dir = Path(model_dir).as_posix()
    key = f'{data_sig}|{model_dir}|{prefix}'
    if prefix == 'cbr':
        key += '|' + json.dumps(config['models']['cvae'], sort_keys=True, default=str)
    path = _cache_file(key)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return z['bag_ids'], z['bag_pred'], z['bag_true']

    print(f'    predicting {prefix:6s} {model_dir}', flush=True)
    if prefix == 'cbr':
        out = _cbr_predictions(folds, config, folds[0]['X'].shape[1], model_dir)
        if out is None:
            raise FileNotFoundError(f'CbR weights missing in {model_dir}')
        preds, y_true, slide_ids = out[:3]
    else:
        ext, model_type = {p: (e, t) for p, e, t in MODEL_SPECS}[prefix]
        preds, y_true, slide_ids = _gather_predictions(folds, model_dir, prefix, ext, model_type)
    bag_ids, bag_pred, bag_true = bag_decisions(preds, slide_ids, y_true)
    bag_ids = np.asarray(bag_ids).astype(str)
    np.savez(_guarded(path), bag_ids=bag_ids, bag_pred=np.asarray(bag_pred), bag_true=np.asarray(bag_true))
    _cache_index_update(key, path, {'model_dir': model_dir, 'prefix': prefix})
    return bag_ids, np.asarray(bag_pred), np.asarray(bag_true)


def _patient_meta(bag_ids, bag_true):
    pats = patient_ids_from_slides(bag_ids)
    per_class = {}
    for p, y in dict(zip(pats, bag_true)).items():
        per_class[str(int(y))] = per_class.get(str(int(y)), 0) + 1
    return {'n_bags': int(len(bag_ids)), 'n_patients': int(len(set(pats))),
            'n_patients_per_class': per_class}


# -------------------------------------------------------------- regimen tests

def regimen_tests(exps):
    from pipeline.evaluation_stage import _build_fold_val_sets
    from src.utils.config_manager import ConfigManager

    for exp in exps:
        jsons = {c: latest_stats_json(exp, c) for c in CANDIDATES}
        jsons = {c: v for c, v in jsons.items() if v[0] is not None}
        if not jsons:
            print(f'[{exp}] no slide-level stats JSON -- skipped')
            continue
        config = ConfigManager.load_config(exp).to_dict()
        stats_cfg = config.get('statistics', {})
        seed = int(stats_cfg.get('random_seed', SEED))
        print(f'\n[{exp}] building validation folds', flush=True)
        folds = _build_fold_val_sets(config)
        sig = _data_signature(folds)

        for cand, (src_path, js) in jsons.items():
            n_boot = int(js['n_bootstrap'])
            print(f'[{exp}] TRTR vs {cand}  (anchor: {src_path.name}, n_boot={n_boot})', flush=True)
            out_rows, meta = [], None
            for row in js['results']:
                if row['model'] == 'Classification_by_Reconstruction':
                    continue
                prefix = LABEL_PREFIX[row['model']]
                ids_a, pa, true_a = cached_bags(folds, sig, js['baseline_dir'], prefix)
                ids_b, pb, true_b = cached_bags(folds, sig, js['candidate_dir'], prefix)
                _check(np.array_equal(ids_a, ids_b) and np.array_equal(true_a, true_b),
                       f'{exp} {prefix}: bag sets differ between regimens')
                meta = meta or _patient_meta(ids_a, true_a)

                # anchor: the original slide-level bootstrap reproduces the stored row
                acc_a, acc_b = float((pa == true_a).mean()), float((pb == true_a).mean())
                _check(_close(acc_a, row['bag_acc_TRTR']) and _close(acc_b, row[f'bag_acc_{cand}']),
                       f'{exp} {cand} {prefix}: bag accuracy {acc_a},{acc_b} vs stored')
                _, _, d_slide = bag_level_bootstrap(pa, pb, true_a, n_boot=n_boot, seed=seed)
                s_slide = summarize_bootstrap_diffs(d_slide)
                for k in ('boot_mean_diff', 'ci_diff_lo', 'ci_diff_hi', 'p_bootstrap'):
                    _check(_close(s_slide[k], row[f'bag_{k}']), f'{exp} {cand} {prefix}: {k}')
                _check(s_slide['significant_95'] == row['bag_significant_95'], f'{exp} {cand} {prefix}: sig')

                _, _, d_pat = bag_level_patient_bootstrap(pa, pb, true_a, ids_a, n_boot=n_boot, seed=seed)
                s_pat = summarize_bootstrap_diffs(d_pat)
                width_slide = s_slide['ci_diff_hi'] - s_slide['ci_diff_lo']
                out_rows.append({
                    'model': row['model'], 'n_bags': len(ids_a),
                    'bag_acc_TRTR': row['bag_acc_TRTR'], f'bag_acc_{cand}': row[f'bag_acc_{cand}'],
                    'bag_delta': row['bag_delta'],
                    **{f'bag_{k}': v for k, v in s_pat.items()},
                    'slide': {k: row[f'bag_{k}'] for k in
                              ('ci_diff_lo', 'ci_diff_hi', 'p_bootstrap', 'significant_95')},
                    'ci_width_ratio': ((s_pat['ci_diff_hi'] - s_pat['ci_diff_lo']) / width_slide
                                       if width_slide > 0 else None),
                })

            cbr_out = None
            cbr_row = js.get('cbr') or next((r for r in js['results']
                                             if r['model'] == 'Classification_by_Reconstruction'), None)
            if cbr_row and js.get('cvae_weights_dir'):
                ids_c, pc, true_c = cached_bags(folds, sig, js['cvae_weights_dir'], 'cbr', config)
                _check(_close((pc == true_c).mean(), cbr_row['cbr_bag_accuracy']),
                       f'{exp} {cand} CbR: bag accuracy')
                boot_slide = bag_level_bootstrap_cbr(pc, true_c, n_boot=n_boot, seed=seed)
                _check(_close(np.percentile(boot_slide, 2.5), cbr_row['cbr_bag_ci_lo'])
                       and _close(np.percentile(boot_slide, 97.5), cbr_row['cbr_bag_ci_hi']),
                       f'{exp} {cand} CbR: CI')
                boot_pat = bag_level_patient_bootstrap_cbr(pc, true_c, ids_c, n_boot=n_boot, seed=seed)
                cbr_out = {
                    'model': 'Classification_by_Reconstruction', 'n_bags': len(ids_c),
                    'cbr_bag_accuracy': cbr_row['cbr_bag_accuracy'],
                    'cbr_bag_boot_mean': float(np.mean(boot_pat)),
                    'cbr_bag_ci_lo': float(np.percentile(boot_pat, 2.5)),
                    'cbr_bag_ci_hi': float(np.percentile(boot_pat, 97.5)),
                    'slide': {'cbr_bag_ci_lo': cbr_row['cbr_bag_ci_lo'],
                              'cbr_bag_ci_hi': cbr_row['cbr_bag_ci_hi']},
                }
            print(f'[{exp}] TRTR vs {cand}: anchor OK ({len(out_rows)} models'
                  f'{" + CbR" if cbr_out else ""})', flush=True)
            _write_json(OUT_ROOT / exp / f'regimen_TRTR_vs_{cand}.json', {
                'experiment': exp, 'baseline_label': 'TRTR', 'candidate_label': cand,
                'resample_unit': 'patient', 'stratified_by': 'class',
                'n_bootstrap': n_boot, 'seed': seed, **(meta or {}),
                'slide_level_source': src_path.relative_to(PROJECT_ROOT).as_posix(),
                'results': out_rows, 'cbr': cbr_out,
            })
        del folds


# ------------------------------------------------------------- pairwise tests

def _pairwise(rows, get_pair, n_boot_default):
    out = []
    for t in rows:
        pa, pb, truth, ids = get_pair(t)
        n_boot, seed = int(t.get('n_boot', n_boot_default)), int(t.get('seed', SEED))
        _check(_close((pa == truth).mean(), t['acc_baseline'])
               and _close((pb == truth).mean(), t['acc_candidate']),
               f'pairwise {t["comparison"]} {t["model"]}: accuracies')
        _, _, d_slide = bag_level_bootstrap(pa, pb, truth, n_boot=n_boot, seed=seed)
        s_slide = summarize_bootstrap_diffs(d_slide)
        _check(_close(s_slide['ci_diff_lo'], t['ci_lo']) and _close(s_slide['ci_diff_hi'], t['ci_hi'])
               and _close(s_slide['p_bootstrap'], t['p_bootstrap']),
               f'pairwise {t["comparison"]} {t["model"]} {t["regimen"]}: CI')
        _, _, d_pat = bag_level_patient_bootstrap(pa, pb, truth, ids, n_boot=n_boot, seed=seed)
        s = summarize_bootstrap_diffs(d_pat)
        out.append({**{k: t[k] for k in ('regimen', 'model', 'comparison', 'baseline_exp',
                                          'candidate_exp', 'acc_baseline', 'acc_candidate', 'delta')},
                    'ci_lo': s['ci_diff_lo'], 'ci_hi': s['ci_diff_hi'],
                    'p_bootstrap': s['p_bootstrap'], 'significant_95': s['significant_95'],
                    'n_bags': len(truth), 'n_boot': n_boot, 'seed': seed,
                    'slide': {k: t[k] for k in ('ci_lo', 'ci_hi', 'p_bootstrap', 'significant_95')}})
    return out


def pairwise_tests():
    lat = json.loads(LATENT_BOOT_JSON.read_text(encoding='utf-8'))
    ids = np.array(lat['bag_ids'])
    truth = np.array(lat['bag_true'])

    def latent_pair(t):
        code, prefix = REGIMEN_CODE[t['regimen']], LABEL_PREFIX[t['model']]
        return (np.array(lat['bag_pred'][f'{code}|{prefix}|{t["baseline_exp"]}']),
                np.array(lat['bag_pred'][f'{code}|{prefix}|{t["candidate_exp"]}']), truth, ids)

    rows = _pairwise(lat['tests'], latent_pair, 1000)
    print(f'latent-dim pairwise: anchor OK ({len(rows)} tests)')
    _write_json(OUT_ROOT / 'pairwise_latent_dim.json',
                {'resample_unit': 'patient', 'stratified_by': 'class', **_patient_meta(ids, truth),
                 'slide_level_source': LATENT_BOOT_JSON.relative_to(PROJECT_ROOT).as_posix(),
                 'tests': rows})

    # B8 stored bag_pred only, in sorted-bag order of the same processed file /
    # splits as B7 (asserted by B8 against exp_608); its own anchor below fails
    # unless that alignment holds.
    pool = json.loads(POOL_BOOT_JSON.read_text(encoding='utf-8'))
    lat608 = {p: lat['bag_pred'][f'TSTR|{p}|exp_608'] for p in LABEL_PREFIX.values()}
    for prefix, d in pool['bag_pred'].items():
        _check(d['exp_608'] == lat608[prefix], f'pool exp_608 {prefix} bag order differs from latent JSON')

    def pool_pair(t):
        d = pool['bag_pred'][LABEL_PREFIX[t['model']]]
        return np.array(d[t['baseline_exp']]), np.array(d[t['candidate_exp']]), truth, ids

    rows = _pairwise(pool['tests'], pool_pair, 1000)
    print(f'pool-size pairwise: anchor OK ({len(rows)} tests)')
    _write_json(OUT_ROOT / 'pairwise_pool_size.json',
                {'resample_unit': 'patient', 'stratified_by': 'class', **_patient_meta(ids, truth),
                 'slide_level_source': POOL_BOOT_JSON.relative_to(PROJECT_ROOT).as_posix(),
                 'bag_ids': ids.tolist(), 'tests': rows})


# ------------------------------------------------------------ paired MIL tests

def _slide_labels(exp):
    """Binary label per slide from the experiment's own key_instance_per_slide
    files (verified identical across files)."""
    labels = {}
    for f in sorted((RESULTS / exp / 'mil_eval' / 'key_instance_per_slide').glob('*.json')):
        p = json.loads(f.read_text(encoding='utf-8'))
        for s, y in zip(p['slides'], p['y_true']):
            _check(labels.setdefault(s, int(y)) == int(y), f'{exp}: label of {s} differs across files')
    return labels


def _paired_variants(d, slides, y, n_boot, seed):
    pat = paired_cluster_test(d, slides, y, n_boot=n_boot, seed=seed, cluster_unit='patient')
    strat = paired_cluster_test(d, slides, y, n_boot=n_boot, seed=seed, cluster_unit='slide')
    return pat, {k: strat[k] for k in ('ci_lo', 'ci_hi', 'p_signflip', 'n_units')}


def paired_mil_tests(exps):
    spatial_mod = _load_module('scripts/spatial_paired_real_vs_shuffled.py')
    ki_mod = _load_module('scripts/key_instance_paired_bootstrap.py')

    for exp in exps:
        labels = _slide_labels(exp)
        if not labels:
            print(f'[{exp}] no key_instance_per_slide labels -- paired MIL tests skipped')
            continue

        # ---- spatial (B11): replay the shared-RNG slide test, then patient test
        sp_path = RESULTS / exp / 'spatial_validation' / 'spatial_paired_real_vs_shuffled.json'
        if sp_path.exists():
            sp = json.loads(sp_path.read_text(encoding='utf-8'))
            n_boot, seed = int(sp['n_boot']), int(sp['seed'])
            _check(n_boot == spatial_mod.N_BOOT, f'{exp}: spatial n_boot')
            rng = np.random.default_rng(seed)
            fits = {}
            for key in sorted(sp['fits']):
                f = sp['fits'][key]
                res = {'stores': f['stores']}
                for name in ('morans_i', 'one_minus_gamma_over_var'):
                    r = f[name]
                    slides, d = r['slides'], np.array(r['per_slide_delta'])
                    rep = spatial_mod.paired_test(dict(zip(slides, d)), {s: 0.0 for s in slides}, rng)
                    _check(_close(rep['ci_lo'], r['ci_lo']) and _close(rep['ci_hi'], r['ci_hi'])
                           and _close(rep['p_signflip'], r['p_signflip']),
                           f'{exp} spatial {key} {name}: CI replay')
                    y = np.array([labels[s] for s in slides])
                    pat, strat = _paired_variants(d, slides, y, n_boot, seed)
                    res[name] = {
                        'n_matched': r['n_matched'], 'mean_real': r['mean_real'],
                        'mean_shuffled': r['mean_shuffled'], 'delta': r['delta'],
                        'ci_lo': pat['ci_lo'], 'ci_hi': pat['ci_hi'],
                        'p_signflip': pat['p_signflip'], 'p_wilcoxon': pat['p_wilcoxon'],
                        'frac_slides_real_higher': r['frac_slides_real_higher'],
                        'significant_95_real_higher': bool(pat['ci_lo'] > 0),
                        'n_patients': pat['n_clusters'], 'n_patients_per_class': pat['n_clusters_per_class'],
                        'slide_stratified': strat,
                        'slide': {k: r[k] for k in ('ci_lo', 'ci_hi', 'p_signflip',
                                                    'significant_95_real_higher')},
                    }
                fits[key] = res
            print(f'[{exp}] spatial: anchor OK ({len(fits)} fits)')
            _write_json(OUT_ROOT / exp / 'spatial_paired.json', {
                'lag': sp['lag'], 'n_boot': n_boot, 'seed': seed, 'resample_unit': 'patient',
                'stratified_by': 'class',
                'slide_level_source': sp_path.relative_to(PROJECT_ROOT).as_posix(), 'fits': fits})

        # ---- key-instance (B10): fresh RNG per test, as the original
        ki_path = RESULTS / exp / 'mil_eval' / 'key_instance_paired.json'
        if ki_path.exists():
            ki = json.loads(ki_path.read_text(encoding='utf-8'))
            n_boot, seed = int(ki['n_boot']), int(ki['seed'])
            _check(n_boot == ki_mod.N_BOOT and seed == ki_mod.SEED, f'{exp}: key-instance settings')
            tiers = {}
            for stem, t in ki['tiers'].items():
                p = json.loads((RESULTS / exp / 'mil_eval' / 'key_instance_per_slide' / t['source'])
                               .read_text(encoding='utf-8'))
                y = np.array(p['y_true'])
                correct = {arm: (np.array(p[arm]) == y).astype(int) for arm in ('selected', 'naive', 'random')}
                res = {'source': t['source']}
                for name, other in (('vs_naive', 'naive'), ('vs_random', 'random')):
                    rep = ki_mod.paired_test(correct['selected'], correct[other])
                    r = t[name]
                    _check(_close(rep['ci_lo'], r['ci_lo']) and _close(rep['ci_hi'], r['ci_hi']),
                           f'{exp} key-instance {stem} {name}: CI replay')
                    d = correct['selected'] - correct[other]
                    pat, strat = _paired_variants(d, p['slides'], y, n_boot, seed)
                    res[name] = {
                        'n_slides': r['n_slides'], 'acc_selected': r['acc_selected'],
                        'acc_other': r['acc_other'], 'delta': r['delta'],
                        'ci_lo': pat['ci_lo'], 'ci_hi': pat['ci_hi'],
                        'significant_95_selected_higher': bool(pat['ci_lo'] > 0),
                        'p_signflip': pat['p_signflip'],
                        'n_patients': pat['n_clusters'], 'n_patients_per_class': pat['n_clusters_per_class'],
                        'slide_stratified': strat,
                        'slide': {k: r[k] for k in ('ci_lo', 'ci_hi', 'significant_95_selected_higher')},
                    }
                tiers[stem] = res
            print(f'[{exp}] key-instance: anchor OK ({len(tiers)} tiers)')
            _write_json(OUT_ROOT / exp / 'key_instance_paired.json', {
                'n_boot': n_boot, 'seed': seed, 'resample_unit': 'patient', 'stratified_by': 'class',
                'slide_level_source': ki_path.relative_to(PROJECT_ROOT).as_posix(), 'tiers': tiers})


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--stage', choices=['cache', 'boot', 'all'], default='all',
                    help="cache: predictions + regimen tests; boot: pairwise + paired MIL "
                         "(no prediction); all: both, with before/after manifests")
    ap.add_argument('--exps', nargs='*', default=None, help='restrict to these experiments')
    args = ap.parse_args()

    before = manifest()
    if args.stage == 'all':
        _write_json(OUT_ROOT / 'manifest_before.json', before)

    if args.stage in ('cache', 'all'):
        regimen_tests([e for e in REGIMEN_EXPS if not args.exps or e in args.exps])
    if args.stage in ('boot', 'all'):
        if not args.exps:
            pairwise_tests()
        paired_mil_tests([e for e in MIL_EXPS if not args.exps or e in args.exps])

    after = manifest()
    if args.stage == 'all':
        _write_json(OUT_ROOT / 'manifest_after.json', after)
    if before != after:
        changed = sorted(set(before.items()) ^ set(after.items()))
        raise SystemExit(f'PROTECTED SLIDE-LEVEL RESULTS CHANGED: {changed[:10]}')
    print(f'\nProtected slide-level results unchanged ({len(after)} files hashed).')


if __name__ == '__main__':
    main()
