"""
Task pipeline for the MIL evaluation tests behind the per-configuration appendix tables
(spatial coherence and key-instance utility, $PAPER_DIR/results_mil_appendix_tables.tex, generated
by scripts/paper/gen_appendix_controlled_experiments.py) for the configurations other than exp_608.
Tasks are ordered from the cheapest to the most computationally expensive; a task runs only
after the tasks it needs, and is skipped when its output already exists or the state file
says it finished.

Cost ranks (static estimates -- the wall-clock of every run is stored in the state file, so
the estimates can be replaced by measurements):
    0  no compute          transcribe existing numbers into key_instance_utility_summary.json
    2  medium, local       key-instance utility: loads the data, refits 2 RFs x 5 folds
    4  heavy, Modal CPU    shuffled-label mi-SVM refit (~4 h sequential locally; fold-parallel
                           on Modal, billed) -- only with --include-modal
    9  finish              paired spatial tests on the new fits, regenerate the paper tables

Usage:
    python scripts/run_missing_mil_tests.py --list
    python scripts/run_missing_mil_tests.py --dry-run
    python scripts/run_missing_mil_tests.py --run --until-rank 2
    python scripts/run_missing_mil_tests.py --run --include-modal
    python scripts/run_missing_mil_tests.py --run --only ki:exp_607:misvm
"""
import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
LOGS = RESULTS / 'mil_missing_tests_logs'
STATE = RESULTS / 'mil_missing_tests_state.json'
PY = sys.executable

KI_EXPS = ['exp_607', 'exp_609', 'exp_2.607', 'exp_2.608', 'exp_4.608']   # exp_608 is complete
MISVM_SHUF_EXPS = ['exp_607', 'exp_609']

# Key-instance (selected / random-subset) RF accuracies of the CVAE tiers, transcribed from the
# appendix key-instance table; 'baseline' is each config's own naive MIL RF (TR/TR bag accuracy
# in its statistical_evaluation JSON). Selected-RF accuracy does not depend on the baseline.
KI_CVAE_TRANSCRIBED = {
    'attention':          {'exp_607': (0.8305, 0.7966), 'exp_609': (0.7797, 0.8136), 'exp_2.607': (0.7966, 0.8136),
                           'exp_2.608': (0.7966, 0.7797), 'exp_4.608': (0.7797, 0.7966)},
    'cbr_reconstruction': {'exp_607': (0.7627, 0.8136), 'exp_609': (0.7458, 0.7627), 'exp_2.607': (0.7797, 0.7966),
                           'exp_2.608': (0.7966, 0.7458), 'exp_4.608': (0.7797, 0.7797)},
    'cbr_encoder_head':   {'exp_607': (0.7966, 0.7966), 'exp_609': (0.8136, 0.8136), 'exp_2.607': (0.8305, 0.7966),
                           'exp_2.608': (0.8136, 0.7797), 'exp_4.608': (0.8305, 0.8136)},
}
KI_BASELINE = {'exp_607': 0.8136, 'exp_609': 0.8136, 'exp_2.607': 0.7797, 'exp_2.608': 0.7797, 'exp_4.608': 0.8136}
KI_TIERS = {   # summary key -> (--method, --variant)
    'misvm': ('misvm', 'mi_svm'),
}


def summary_path(exp):
    return RESULTS / exp / 'mil_eval' / 'key_instance_utility_summary.json'


def has_ki(exp, key):
    p = summary_path(exp)
    return p.exists() and key in json.loads(p.read_text(encoding='utf-8'))


@dataclass
class Task:
    id: str
    rank: int
    desc: str
    fills: int = 0                    # missing appendix-table results this task produces
    cmd: list = field(default_factory=list)
    fn: object = None                 # in-process callable instead of a subprocess
    needs: list = field(default_factory=list)
    done: object = None               # () -> bool, output already present
    modal: bool = False


# ---------------------------------------------------------------- in-process tasks
def transcribe_ki_cvae():
    for exp in KI_EXPS:
        p = summary_path(exp)
        d = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        d['_provenance'] = ('Key-instance utility, CVAE tiers: selected-RF and size-matched random-subset '
                            'accuracies transcribed from the appendix key-instance table (scripts/cvae_weighted_rf_vote.py; '
                            'median split on non-diagnostic mass). baseline_bag_acc is the config\'s own naive MIL RF '
                            '(TR/TR bag accuracy, statistical_evaluation JSON). mi-SVM entries are '
                            'parsed from key_instance_logs/ by scripts/run_missing_mil_tests.py.')
        d['baseline_bag_acc'] = KI_BASELINE[exp]
        for tier, per_exp in KI_CVAE_TRANSCRIBED.items():
            sel, rnd = per_exp[exp]
            d[tier] = {'key_instance_train': sel, 'random_subsample': rnd}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2), encoding='utf-8')
        print(f'  wrote {p}')


_RE_BASE = re.compile(r'plain majority vote \(baseline\): bag_acc=([\d.]+)')
_RE_RES = re.compile(r'(key_instance_train|random_subsample) \(.*?(\d+) bags fell back\): bag_acc=([\d.]+)\s+'
                     r'delta=([+-][\d.]+)\s+95%CI\[([+-][\d.]+),([+-][\d.]+)\]')


def collect_ki(exp, key):
    """Parse results/<exp>/mil_eval/key_instance_logs/<exp>_<method>_<variant>.log into the summary JSON
    in the format of exp_608's mi-SVM entry."""
    method, variant = KI_TIERS[key]
    log = (RESULTS / exp / 'mil_eval' / 'key_instance_logs' / f'{exp}_{method}_{variant}.log').read_text(encoding='utf-8',
                                                                                                           errors='replace')
    base = float(_RE_BASE.search(log).group(1))
    got = {m.group(1): m for m in _RE_RES.finditer(log)}
    assert set(got) == {'key_instance_train', 'random_subsample'}, f'{exp}/{key}: result lines not found in log'
    k = got['key_instance_train']
    entry = {'baseline_bag_acc': base, 'key_instance_train': float(k.group(3)),
             'random_subsample': float(got['random_subsample'].group(3)),
             'n_bags_fell_back': int(k.group(2)), 'delta_vs_baseline': float(k.group(4)),
             'ci95_vs_baseline': [float(k.group(5)), float(k.group(6))]}
    p = summary_path(exp)
    d = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    d[key] = entry
    p.write_text(json.dumps(d, indent=2), encoding='utf-8')
    print(f'  {exp}/{key}: selected {entry["key_instance_train"]:.4f}  random {entry["random_subsample"]:.4f}  '
          f'baseline {base:.4f}')


def refresh_tables():
    subprocess.run([PY, str(ROOT / 'scripts' / 'paper' / 'gen_appendix_controlled_experiments.py'), '--results-tables-only'],
                   check=True, cwd=ROOT)


# ---------------------------------------------------------------- task list
def build_tasks():
    T = []
    # rank 0 -- no compute
    T.append(Task('transcribe:ki_cvae', 0, 'Key-instance vs naive MIL for the three CVAE tiers (selected/random RF '
                  'accuracies from the appendix table, per-config naive RF baseline)', fills=15, fn=transcribe_ki_cvae,
                  done=lambda: all(has_ki(e, 'cbr_encoder_head') for e in KI_EXPS)))
    # rank 2 -- key-instance utility of mi-SVM (fresh RFs)
    for e in KI_EXPS:
        for key, (method, variant) in KI_TIERS.items():
            T.append(Task(f'ki:{e}:{key}', 2, f'{e}: key-instance utility, {key} (selected vs random vs naive)', fills=2,
                          cmd=[PY, 'scripts/cvae_weighted_rf_vote.py', '--config', e, '--cvae-config', e,
                               '--method', method, '--variant', variant],
                          done=lambda e=e, key=key: has_ki(e, key)))
    # rank 4 -- shuffled-label mi-SVM refits on Modal CPU, then their stage-5 runs
    for e in MISVM_SHUF_EXPS:
        store = RESULTS / e / 'latent' / 'latent_labels_shuffled_misvm.h5'
        out5 = RESULTS / e / 'spatial_validation' / 'stage5_misvm_mi_svm_shuffled.json'
        T.append(Task(f'fit:{e}:misvm_shuffled', 4, f'{e}: shuffled-bag-label refit of mi-SVM (Modal, fold-parallel)',
                      cmd=[PY, 'scripts/shuffled_label_control.py', '--config', e, '--method', 'misvm',
                           '--store-suffix', 'shuffled_misvm', '--backend', 'modal'],
                      done=store.exists, modal=True))
        T.append(Task(f'stage5:{e}:misvm_shuffled', 4, f'{e}: stage-5 statistics of the shuffled mi-SVM fit', fills=1,
                      cmd=[PY, 'scripts/spatial_validation.py', '--config', e, '--method', 'misvm', '--variant', 'mi_svm',
                           '--store-path', str(store), '--out-path', str(out5)],
                      needs=[f'fit:{e}:misvm_shuffled'], done=out5.exists, modal=True))
    # rank 9 -- paired spatial tests on the new fits, then the tables
    needs_by_exp = {e: [f'stage5:{e}:misvm_shuffled'] for e in MISVM_SHUF_EXPS}
    for e, needs in needs_by_exp.items():
        want = ['misvm_mi_svm']

        def paired_done(e=e, want=want):
            p = RESULTS / e / 'spatial_validation' / 'spatial_paired_real_vs_shuffled.json'
            return p.exists() and all(k in json.loads(p.read_text(encoding='utf-8'))['fits'] for k in want)
        T.append(Task(f'paired:{e}', 9, f'{e}: paired real-vs-shuffled spatial bootstrap, incl. the new fits',
                      fills=len(want), cmd=[PY, 'scripts/spatial_paired_real_vs_shuffled.py', '--config', e],
                      needs=needs, done=paired_done, modal=any(n.endswith('misvm_shuffled') for n in needs)))
    T.append(Task('finish:tables', 9, 'Regenerate the paper tables (paired spatial, key-instance)', fn=refresh_tables,
                  needs=[t.id for t in T if t.id.startswith('paired:')], done=lambda: False))
    return T


def order(tasks):
    """Cheapest first; a task never precedes the tasks it needs."""
    by_id = {t.id: t for t in tasks}
    out, seen = [], set()

    def visit(t):
        if t.id in seen:
            return
        for n in t.needs:
            if n in by_id:
                visit(by_id[n])
        seen.add(t.id)
        out.append(t)
    for t in sorted(tasks, key=lambda t: (t.rank, tasks.index(t))):
        visit(t)
    return out


# ---------------------------------------------------------------- runner
def load_state():
    return json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2), encoding='utf-8')


def is_done(t, st):
    return st.get(t.id, {}).get('status') == 'done' or (t.done is not None and t.done())


def run_task(t, st):
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f'\n=== {t.id}  [rank {t.rank}]  {t.desc}', flush=True)
    rc = 0
    if t.fn is not None:
        t.fn()
    else:
        m = re.match(r'ki:(.+?):(.+)', t.id)
        log = LOGS / (t.id.replace(':', '_') + '.log')
        with open(log, 'w', encoding='utf-8') as f:
            cmd = [t.cmd[0], '-u'] + t.cmd[1:]   # unbuffered, so the log shows progress
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if m and rc == 0:
            exp, key = m.groups()
            method, variant = KI_TIERS[key]
            keep = RESULTS / exp / 'mil_eval' / 'key_instance_logs' / f'{exp}_{method}_{variant}.log'
            keep.parent.mkdir(parents=True, exist_ok=True)
            keep.write_text(log.read_text(encoding='utf-8', errors='replace'), encoding='utf-8')
            collect_ki(exp, key)
        print(f'  exit {rc}  (log: {log})')
    # several runner processes may work concurrently (local + Modal): merge into the state on disk
    st.update(load_state())
    st[t.id] = {'status': 'done' if rc == 0 else 'failed', 'seconds': round(time.time() - t0, 1),
                'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'returncode': rc}
    save_state(st)
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--until-rank', type=int, default=9, help='run tasks up to this cost rank')
    ap.add_argument('--include-modal', action='store_true', help='allow the billed Modal tasks (rank 4)')
    ap.add_argument('--only', nargs='+', default=None, help='task ids (prefix match)')
    args = ap.parse_args()

    tasks = order(build_tasks())
    st = load_state()
    sel = [t for t in tasks if t.rank <= args.until_rank and (args.include_modal or not t.modal)
           and (args.only is None or any(t.id.startswith(o) for o in args.only))]

    if args.list or args.dry_run or not args.run:
        print(f'{"#":>2} {"rank":>4}  {"fills":>5}  {"status":8} {"task":34} description')
        for i, t in enumerate(tasks, 1):
            status = 'done' if is_done(t, st) else st.get(t.id, {}).get('status', 'pending')
            print(f'{i:>2} {t.rank:>4}  {t.fills:>5}  {status:8} {t.id:34} {t.desc[:90]}{"  [Modal]" if t.modal else ""}')
            if args.dry_run and t.cmd:
                print(f'{"":>26}$ {" ".join(map(str, t.cmd))}')
        pending = sum(t.fills for t in tasks if not is_done(t, st))
        print(f'\n{len(tasks)} tasks; {pending} appendix-table results still to produce')
        return

    by_id = {t.id: t for t in tasks}
    failed = set()
    for t in sel:
        if is_done(t, st):
            print(f'skip {t.id} (done)')
            continue
        if any(not is_done(by_id[n], st) for n in t.needs if n in by_id):
            print(f'skip {t.id} (a task it needs has not finished)')
            failed.add(t.id)
            continue
        if not run_task(t, st):
            failed.add(t.id)
    print('\nfailed:', sorted(failed) if failed else 'none')


if __name__ == '__main__':
    main()
