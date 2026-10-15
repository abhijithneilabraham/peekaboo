"""Reproduce "Vega learns from feedback" end to end, for Vega 0.8B and Vega 4B, and build the VegaML-style tables.

    python reproduce.py            # everything; resumable: finished steps are skipped
    python reproduce.py --check    # exit 0 complete, 1 not complete and not running (see STATUS.md), 2 still running
    python reproduce.py --quick    # a few rows of everything, 0.8B, ~15 min on a GPU: tests the code paths

What runs, per size (0.8B, then 4B):
  1. suite_cache.py   read the 43 public tasks, JevBench and DecisionBench through Vega.decide() exactly as
                      VegaML's benchmark notebook does; keep decide()'s records and the features it used (GPU)
  2. adapt.py         replay check, then every method on those features: VegaML's TTT head, a supervised adapter,
                      ours (feedback adapter), ours with KL 1, and a no-engine control; 3 seeds each
  3. bench_eval.py, bench_rl.py, figure.py, latency.py   the DecisionBench studies of the paper (Section 5.5-5.6, 6, 8)
  4. extract.py, vtrain.py, fitprobe.py                    Kuhn poker (Section 7)
Then table.py writes TABLES.md: VegaML's README tables with "this run" and "+ ours" columns for both sizes.

Every step writes a result file; a step whose file exists is skipped, so a stopped run continues where it was.
A failing step is retried 3 times, then recorded, and the run goes on with everything that does not depend on it.
STATUS.md (in the output folder) lists every step with its state and, for failures, the end of its log.
Output: ./reproduction (or VEGA_OUT). Device: VEGA_DEVICE, default cuda if available.
"""
import argparse, glob, json, os, platform, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = ['0.8b', '4b']
SUITES = ['public', 'jevbench', 'decisionbench']
METHODS = ['sup_ts', 'feedback', 'feedback_kl1', 'linear_fb']
SEEDS = [0, 1, 2]
TRIES = 3


class Step:
    def __init__(self, name, args, done, size=None, needs=(), always=False):
        self.name, self.args, self.done, self.size, self.needs, self.always = name, args, done, size, list(needs), always
        self.state, self.note = 'pending', ''


def plan(out, quick):
    seeds = [0] if quick else SEEDS
    q = ['--quick'] if quick else []
    steps = []
    for size in (['0.8b'] if quick else SIZES):
        sfx = '' if size == '0.8b' else '_4b'
        res = os.path.join(out, 'results' + sfx)
        f = lambda *p: os.path.join(res, *p)
        ex = lambda p: (lambda: os.path.exists(p))
        add = lambda name, args, done, needs=(), always=False: steps.append(Step(f'{size}/{name}', args, done, size, [f'{size}/{n}' for n in needs], always))
        # 1. read every suite through decide()
        for s in SUITES:
            add(f'cache/{s}', ['suite_cache.py', '--suite', s] + q, ex(f(f'cache_{s}.done')))
        # 2. methods
        for s in SUITES:
            add(f'replay/{s}', ['adapt.py', '--suite', s, '--method', 'replay'], ex(f('methods', s, 'replay_s0.json')), [f'cache/{s}'])
            if s != 'jevbench':
                add(f'ttt/{s}', ['adapt.py', '--suite', s, '--method', 'ttt'], ex(f('methods', s, 'ttt_s0.json')), [f'cache/{s}'])
            for m in METHODS:
                for sd in seeds:
                    add(f'{m}/{s}/s{sd}', ['adapt.py', '--suite', s, '--method', m, '--seed', str(sd)],
                        ex(f('methods', s, f'{m}_s{sd}.json')), [f'cache/{s}'])
        # 3. DecisionBench studies (bench_rl.py writes results*/db_<method>_<split>_s<seed><tag>.json)
        db = ['cache/decisionbench']
        add('db/variants', ['bench_eval.py', '1.5', '2', '3'], ex(f('decisionbench_variants.json')), db)
        big = ['--items', 'decisionbench_items_100.json'] + (['--n_test', '2'] if quick else ['--n_test', '40'])
        ns = [4] if quick else [5, 15, 30, 60]
        for sd in seeds:
            R = lambda tag, args, split='rows': add(f'db/{tag}/s{sd}', ['bench_rl.py'] + args + ['--seed', str(sd), '--tag', tag],
                                                   ex(f(f'db_{args[args.index("--method") + 1]}_{split}_s{sd}{tag}.json')), db)
            R('_big', big + ['--method', 'zero', '--n_train', str(ns[-1])])
            for n in ns:
                R(f'_big_n{n}_kl0.1', big + ['--method', 'bandit', '--kl', '0.1', '--n_train', str(n)])
                R(f'_big_n{n}_kl1', big + ['--method', 'bandit', '--kl', '1', '--n_train', str(n)])
                R(f'_big_n{n}_sup', big + ['--method', 'sup', '--kl', '0.1', '--n_train', str(n)])
            R(f'_big_n{ns[-1]}_kl0.1_online', big + ['--method', 'bandit', '--kl', '0.1', '--n_train', str(ns[-1]), '--epochs', '1'])
            R('_big_tasks_kl0.1', ['--items', 'decisionbench_items_100.json', '--method', 'bandit', '--kl', '0.1', '--split', 'tasks'], 'tasks')
            for sp in ('rows', 'tasks'):                     # the pilot where the KL weight was chosen (Section 8)
                small = ['--split', sp] + (['--n_test', '2', '--n_train', '2'] if quick else [])
                add(f'db/pilot_{sp}_zero/s{sd}', ['bench_rl.py', '--method', 'zero', '--seed', str(sd)] + small, ex(f(f'db_zero_{sp}_s{sd}.json')), db)
                for kl in ('0', '0.1', '1'):
                    add(f'db/pilot_{sp}_kl{kl}/s{sd}', ['bench_rl.py', '--method', 'bandit', '--kl', kl, '--seed', str(sd), '--tag', f'_kl{kl}'] + small,
                        ex(f(f'db_bandit_{sp}_s{sd}_kl{kl}.json')), db)
                add(f'db/pilot_{sp}_sup/s{sd}', ['bench_rl.py', '--method', 'sup', '--kl', '0.1', '--seed', str(sd), '--tag', '_kl0.1'] + small,
                    ex(f(f'db_sup_{sp}_s{sd}_kl0.1.json')), db)
        if not quick:
            add('db/figure', ['figure.py'], ex(os.path.join(out, 'figures', f'vega_feedback_curve{sfx}.png')),
                [s.name.split('/', 1)[1] for s in steps if s.name.startswith(f'{size}/db/_big')])
        add('db/latency', ['latency.py'], ex(f('compare', 'latency.json')), db)
        # 4. Kuhn poker
        add('kuhn/features', ['extract.py', 'kuhn'], ex(os.path.join(out, f'bank_kuhn{sfx}.pt')))
        K = lambda tag, args, brain: add(f'kuhn/{brain}{tag}', ['vtrain.py', '--env', 'kuhn', '--brain', brain] + args + (['--tag', tag] if tag else []),
                                         ex(f(f'kuhn_{brain}_pool1_s{args[args.index("--seed") + 1] if "--seed" in args else 0}{tag}.json')), ['kuhn/features'])
        up = ['--updates', '10'] if quick else []
        K('', ['--seed', '0'], 'vega0')
        for c in ([5] if quick else [3, 5, 10, 20]):
            K(f'_c{c}', ['--seed', '0', '--contrast', str(c)], 'vega0')
        for b in ('adapter', 'engine'):
            for lr in (['1e-4'] if quick else ['3e-4', '1e-4', '3e-5', '1e-5']):
                for sd in ([0] if quick else [0, 1]):
                    K(f'_lr{lr}', up + ['--lr', lr, '--seed', str(sd)], b)
        for sd in seeds:
            for brain, lr, extra, tag in [('engine', '1e-5', [], '_base'), ('adapter', '1e-4', ['--kl', '0.3'], '_kl0.3'),
                                          ('adapter', '1e-4', ['--kl', '1'], '_kl1'), ('engine', '1e-5', ['--ent', '0.1'], '_ent0.1'),
                                          ('engine', '1e-5', ['--contrast', '5'], '_c5'), ('adapter', '1e-4', ['--contrast', '5', '--kl', '0.3'], '_c5kl0.3'),
                                          ('engine', '1e-5', ['--contrast', '5', '--ent', '0.1'], '_c5ent0.1'),
                                          ('adapter', '1e-4', ['--contrast', '5', '--kl', '1', '--ent', '0.1'], '_c5kl1ent0.1')]:
                K(tag, up + ['--lr', lr, '--seed', str(sd)] + extra, brain)
        for sd in ([0] if quick else [0, 1, 2, 3, 4]):
            L = ['--updates', '10' if quick else '300', '--seed', str(sd)]
            for brain, extra, tag in [('adapter', ['--lr', '1e-4', '--kl', '1'], '_L_kl1'), ('adapter', ['--lr', '1e-4', '--kl', '3'], '_L_kl3'),
                                      ('adapter', ['--lr', '1e-4', '--kl', '1', '--magnet', '30'], '_L_kl1m30'),
                                      ('adapter', ['--lr', '1e-4', '--kl', '3', '--magnet', '30'], '_L_kl3m30'),
                                      ('table', [], '_L_plain'), ('table', ['--kl', '1'], '_L_kl1'), ('table', ['--kl', '1', '--magnet', '30'], '_L_kl1m30')]:
                K(tag, L + extra, brain)
        add('kuhn/fitprobe', ['fitprobe.py'], ex(f('fitprobe.txt')), ['kuhn/features'])
    steps.append(Step('tables', ['table.py'], lambda: False, None, [], always=True))
    return steps


# ───────────────────────────── running ─────────────────────────────
def free_mem_ok():
    """macOS only (the 8 GB laptop the paper used): wait until 30% of memory is free before a heavy step."""
    if platform.system() != 'Darwin':
        return
    while True:
        try:
            out = subprocess.run(['memory_pressure'], capture_output=True, text=True).stdout
            pct = int(out.split('free percentage:')[1].split('%')[0])
        except Exception:
            return
        if pct >= 30:
            return
        print(f'  waiting for memory: {pct}% free', flush=True); time.sleep(30)


def run_step(st, out, env):
    log = os.path.join(out, 'logs', st.name.replace('/', '__') + '.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    for attempt in range(1, TRIES + 1):
        free_mem_ok()
        t0 = time.time()
        print(f'\n==== {st.name}  (attempt {attempt}/{TRIES}, {time.strftime("%H:%M:%S")})', flush=True)
        with open(log, 'a') as fh:
            fh.write(f'\n==== attempt {attempt} {time.ctime()}  {" ".join(st.args)}\n'); fh.flush()
            p = subprocess.Popen([sys.executable, '-u'] + st.args, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
            for line in p.stdout:
                fh.write(line)
                if not line.lower().startswith(('warning', 'fetching')) and 'warn' not in line[:40].lower():
                    print('  ' + line.rstrip()[:240], flush=True)
            code = p.wait()
        if st.name.endswith('kuhn/fitprobe') and code == 0:       # fitprobe only prints; keep its output as the result
            res = os.path.join(out, 'results' + ('' if st.size == '0.8b' else '_4b'), 'fitprobe.txt')
            open(res, 'w').write(open(log).read().split('==== attempt')[-1])
        if code == 0 and (st.always or st.done()):
            st.state, st.note = 'done', f'{time.time() - t0:.0f}s'
            return
        st.note = f'exit {code}' + ('' if code else ', but its result file is missing')
        print(f'  FAILED: {st.note}; log {log}', flush=True)
    st.state = 'failed'


def heartbeat(out, state):
    """reproduction/RUNNING holds the current step and is touched every 30 s while a run is alive."""
    import threading
    path = os.path.join(out, 'RUNNING')

    def beat():
        while not state.get('stop'):
            open(path, 'w').write(json.dumps({'pid': os.getpid(), 'step': state.get('step'), 'since': state.get('since'),
                                              'beat': time.time()}))
            time.sleep(30)
    threading.Thread(target=beat, daemon=True).start()


def running(out):
    """The live run's heartbeat, or None (no run, or the last one died more than 2 minutes ago)."""
    try:
        hb = json.load(open(os.path.join(out, 'RUNNING')))
        return hb if time.time() - hb['beat'] < 120 else None
    except Exception:
        return None


def write_status(steps, out):
    by = {}
    for st in steps:
        by.setdefault(st.state, []).append(st)
    L = [f'# Reproduction status  ({time.ctime()})', '',
         ' · '.join(f'{k}: {len(v)}' for k, v in sorted(by.items())) + f'  (of {len(steps)} steps)', '']
    if any(st.state != 'done' for st in steps if st.name != 'tables'):
        L += ['NOT COMPLETE. Fix the cause of the first failed step (see its log), then run `python reproduce.py` again;',
              'finished steps are skipped. Do not change methods, settings, seeds, splits or metrics. See CLAUDE.md.', '']
    else:
        L += ['COMPLETE. The tables are in TABLES.md.', '']
    for st in steps:
        if st.state in ('failed', 'blocked', 'pending'):
            L.append(f'- **{st.state}** `{st.name}`  {st.note}  (`python {" ".join(st.args)}`)')
            if st.state == 'failed':
                log = os.path.join(out, 'logs', st.name.replace('/', '__') + '.log')
                tail = open(log, errors='replace').read().splitlines()[-25:] if os.path.exists(log) else []
                L += ['  ```'] + ['  ' + t for t in tail] + ['  ```']
    L += ['', '<details><summary>all steps</summary>', ''] + [f'- {st.state:8s} {st.name} {st.note}' for st in steps] + ['', '</details>']
    open(os.path.join(out, 'STATUS.md'), 'w').write('\n'.join(L) + '\n')


def ensure_packages():
    try:
        import vegaml, transformers, datasets, matplotlib  # noqa: F401
        assert vegaml.__version__ == '0.8.0'
    except Exception:
        print('installing requirements.txt into', sys.executable, flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', os.path.join(HERE, 'requirements.txt')])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true'); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = os.path.abspath(a.out or os.environ.get('VEGA_OUT') or os.path.join(HERE, 'reproduction_quick' if a.quick else 'reproduction'))
    os.makedirs(out, exist_ok=True)
    steps = plan(out, a.quick)
    for st in steps:
        if not st.always and st.done():
            st.state = 'done'
    live = running(out)
    if a.check:
        complete = all(st.state == 'done' for st in steps if not st.always)
        if live:
            print(f"RUNNING (pid {live['pid']}): step {live['step']} since {time.ctime(live['since'])}. "
                  f"{sum(st.state == 'done' for st in steps)}/{len(steps)} steps done. Log: reproduction/run.log")
            sys.exit(2)
        if os.path.exists(os.path.join(out, 'STATUS.md')):
            print(open(os.path.join(out, 'STATUS.md')).read())
        print('COMPLETE' if complete else f"NOT RUNNING, {sum(st.state != 'done' for st in steps if not st.always)} steps not done")
        sys.exit(0 if complete else 1)
    if live:
        sys.exit(f"another run is alive (pid {live['pid']}, step {live['step']}); wait for it, or check with --check")
    hb = {'step': 'setup', 'since': time.time()}
    heartbeat(out, hb)
    ensure_packages()
    try:
        import torch
        dev = os.environ.get('VEGA_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device {dev}' + (f' ({torch.cuda.get_device_name(0)})' if dev.startswith('cuda') else
              '  (no GPU: the 4B backbone on a CPU takes days; use a GPU with >= 12 GB)'), flush=True)
    except Exception:
        pass
    base_env = dict(os.environ, VEGA_OUT=out, PYTHONUNBUFFERED='1', HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false')
    names = {st.name: st for st in steps}
    t0 = time.time()
    for st in steps:
        if st.state == 'done':
            continue
        bad = [n for n in st.needs if names[n].state != 'done']
        if bad:
            st.state, st.note = 'blocked', 'needs ' + ', '.join(bad); continue
        env = dict(base_env, VEGA_SIZE=st.size or '0.8b')
        hb.update(step=st.name, since=time.time())
        run_step(st, out, env)
        write_status(steps, out)
    write_status(steps, out)
    left = [st for st in steps if st.state != 'done' and not st.always]
    print(f'\n{len(steps) - len(left)}/{len(steps)} steps done in {(time.time() - t0) / 3600:.1f} h. '
          + ('COMPLETE: ' + os.path.join(out, 'TABLES.md') if not left else f'{len(left)} not done: see {os.path.join(out, "STATUS.md")}'), flush=True)
    hb['stop'] = True
    try:
        os.remove(os.path.join(out, 'RUNNING'))
    except OSError:
        pass
    sys.exit(0 if not left else 1)


if __name__ == '__main__':
    main()
