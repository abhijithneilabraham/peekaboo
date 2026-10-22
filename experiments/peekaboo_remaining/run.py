"""The remaining Peekaboo experiments: only head-to-head results on Vega's and Laya's own benchmarks.

    python run.py              # everything below; resumable (finished steps are skipped); several steps at once
    python run.py --check      # exit 0 complete, 1 not complete and not running (see output/STATUS.md), 2 running
    python run.py --plan       # list the steps without running anything

| benchmark (theirs)                    | their model / their improvement (answer key)      | ours (right/wrong only) |
|---------------------------------------|---------------------------------------------------|-------------------------|
| 43 public tasks (Vega README)         | base Vega; VegaML's TTT head; Vega answer-key adapter | Vega + Peekaboo     |
| JevBench (Vega README)                | base Vega                                          | Vega + Peekaboo         |
| typed-decisions (Laya README)         | base Laya; laya-typed-decisions (Laya's fine-tune) | Laya + Peekaboo         |
| public tasks, JevBench, DecisionBench | base Laya; laya-typed-decisions                    | Laya + Peekaboo         |
| typed-decisions, same rows            | base Vega; VegaML's TTT head; Vega answer-key adapter | Vega + Peekaboo     |
Vega 0.8B and 4B, 3 seeds. DecisionBench (Vega README) is already done: experiments/vega_policy/results*.
Then table.py writes output/TABLES.md with each benchmark's table, base vs + Peekaboo vs their own improvement.

A failing step is retried 3 times, then recorded; everything that does not depend on it still runs.
Settings: --jobs N (parallel steps, default 4 on a GPU), VEGA_DEVICE (default cuda), HF_TOKEN (optional).
"""
import argparse, json, os, platform, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
VP = os.path.abspath(os.path.join(HERE, '..', 'vega_policy'))
SEEDS = [0, 1, 2]
TRIES = 3


class Step:
    def __init__(self, name, cwd, args, done, model, gpu, needs=()):
        self.name, self.cwd, self.args, self.done, self.model, self.gpu = name, cwd, args, done, model, gpu
        self.needs, self.state, self.note, self.tries, self.proc, self.t0 = list(needs), 'pending', '', 0, None, 0


SUITES = ('public', 'jevbench', 'decisionbench', 'typed', 'emotion', 'agnews')
VEGA = (('0.8b', 'results', 8, 2), ('4b', 'results_4b', 12, 4))   # (size, results dir, GB for the cache step, GB for a method step)


def plan(out):
    """Every step, in priority order (the scheduler starts ready steps in this order): read every suite through
    every model; the main comparison (base, their answer-key improvement, Peekaboo); pooled Peekaboo; the ablations."""
    steps = []
    ex = lambda *p: (lambda: os.path.exists(os.path.join(out, *p)))
    sfx = {'0.8b': '', '4b': '_4b', 'laya': '_laya', 'laya_td': '_laya_td'}
    # a reading counts as done only while its feature file exists (a new machine has the results but not the features)
    cached = lambda model, res, s: (lambda: os.path.exists(os.path.join(out, res, f'cache_{s}.done'))
                                    and os.path.exists(os.path.join(out, f'bank_{s}{sfx[model]}.pt')))
    V = lambda size, n, a, d, g, needs: steps.append(Step(f'vega-{size}/{n}', VP, a, d, size, g, needs))
    Lm = lambda model, n, a, d, g, needs: steps.append(Step(f'{model}/{n}', HERE, ['laya_peekaboo.py'] + a, d, model, g, needs))
    rows_from = lambda s: 'laya/cache/decisionbench' if s == 'decisionbench' else f'vega-0.8b/cache/{s}'   # one writer per row list
    # 1. read every suite once through every model
    for s in SUITES:
        for size, res, gc, _ in VEGA:
            needs = [] if (size == '0.8b' and s != 'decisionbench') else [rows_from(s)]
            V(size, f'cache/{s}', ['suite_cache.py', '--suite', s], cached(size, res, s), gc, needs)
        for model in ('laya', 'laya_td'):
            needs = [] if (model == 'laya' and s == 'decisionbench') else [rows_from(s)]
            Lm(model, f'cache/{s}', ['cache', '--suite', s], cached(model, f'results_{model}', s), 4, needs)
    for s in SUITES:
        for size, res, _, gr in VEGA:
            V(size, f'replay/{s}', ['adapt.py', '--suite', s, '--method', 'replay'], ex(res, 'methods', s, 'replay_s0.json'), gr, [f'vega-{size}/cache/{s}'])
        Lm('laya', f'replay/{s}', ['train', '--suite', s, '--method', 'replay'], ex('results_laya', 'methods', s, 'replay_s0.json'), 4, [f'laya/cache/{s}'])
    # 2. the main comparison: their answer-key improvement and Peekaboo, 3 seeds
    for s in SUITES:
        for size, res, _, gr in VEGA:
            need = [f'vega-{size}/cache/{s}']
            if s != 'jevbench':
                V(size, f'ttt/{s}', ['adapt.py', '--suite', s, '--method', 'ttt'], ex(res, 'methods', s, 'ttt_s0.json'), 0, need)
            for m in (['feedback'] if s == 'jevbench' else ['sup_ts', 'feedback']):
                for sd in SEEDS:
                    V(size, f'{m}/{s}/s{sd}', ['adapt.py', '--suite', s, '--method', m, '--seed', str(sd)], ex(res, 'methods', s, f'{m}_s{sd}.json'), gr, need)
        for m in (['feedback'] if s == 'jevbench' else ['sup_ts', 'feedback']):
            for sd in SEEDS:
                Lm('laya', f'{m}/{s}/s{sd}', ['train', '--suite', s, '--method', m, '--seed', str(sd)],
                   ex('results_laya', 'methods', s, f'{m}_s{sd}.json'), 8, ['laya/cache/' + s])
    # 3. pooled Peekaboo: one adapter per model from every suite's feedback, used on every suite (JevBench: transfer only)
    for sd in SEEDS:
        for size, res, _, gr in VEGA:
            V(size, f'pooled/s{sd}', ['adapt.py', '--suite', 'all', '--method', 'pooled', '--seed', str(sd)],
              ex(res, 'methods', f'pooled_s{sd}.done'), 2 * gr, [f'vega-{size}/cache/{x}' for x in SUITES])
        Lm('laya', f'pooled/s{sd}', ['train', '--suite', 'all', '--method', 'pooled', '--seed', str(sd)],
           ex('results_laya', 'methods', f'pooled_s{sd}.done'), 10, [f'laya/cache/{x}' for x in SUITES])
    # 4. ablation of Peekaboo: no leash (KL 0). (No honesty fix and the switch come free with every Peekaboo run:
    #    records_raw, records_switch.) KL 1 is not rerun: DecisionBench already shows a 10x tighter leash costs accuracy.
    for m in ('feedback_kl0',):
        for s in SUITES:
            for sd in SEEDS:
                for size, res, _, gr in VEGA:
                    V(size, f'{m}/{s}/s{sd}', ['adapt.py', '--suite', s, '--method', m, '--seed', str(sd)], ex(res, 'methods', s, f'{m}_s{sd}.json'), gr, [f'vega-{size}/cache/{s}'])
                Lm('laya', f'{m}/{s}/s{sd}', ['train', '--suite', s, '--method', m, '--seed', str(sd)],
                   ex('results_laya', 'methods', s, f'{m}_s{sd}.json'), 8, ['laya/cache/' + s])
    # 5. Peekaboo on Laya with the encoder unfrozen too: changes as much of Laya as laya-typed-decisions' fine-tune did
    for s in SUITES:
        for sd in SEEDS:
            Lm('laya', f'feedback_enc/{s}/s{sd}', ['train', '--suite', s, '--method', 'feedback_enc', '--seed', str(sd)],
               ex('results_laya', 'methods', s, f'feedback_enc_s{sd}.json'), 14, ['laya/cache/' + s])
    steps.append(Step('tables', HERE, ['table.py'], lambda: False, None, 0))
    return steps


REAL_SUITES = ('decisionbench', 'typed', 'emotion')              # realistic-feedback arms run on three benchmarks
REALISTIC = ('feedback_noise20', 'feedback_half', 'feedback_complaints', 'logged_argmax', 'logged_ips')


def plan_v2(out):
    """The v2 grid (after the review of the first paper). Same rows, models and metrics as v1; what changed:
      * every feedback method now has a one-pass arm (each row rated once, as a deployment would) next to the
        8-pass arm, and the answer-key arms (sup_ts / sup_enc) also report an early-stopped version (records_es)
      * Laya gets the matched answer-key fine-tune with the encoder unfrozen (sup_enc) and the KL term on sup_ts
      * the answer-key arms are cross-fitted on JevBench too, like Peekaboo
      * typed-decisions on Vega is also run with the shipped typed_decisions adapter off (_noship)
      * realistic feedback (noisy, missing, complaints only, logged argmax, logged sampled) on three benchmarks
      * TTT is scored under both vegaml rules (records / records_guard); the switch compares best answers
      * emotion and AG News pools keep the natural class mix (suite_cache.py)"""
    steps = []
    ex = lambda *p: (lambda: os.path.exists(os.path.join(out, *p)))
    sfx = {'0.8b': '', '4b': '_4b', 'laya': '_laya', 'laya_td': '_laya_td'}
    cached = lambda model, res, s: (lambda: os.path.exists(os.path.join(out, res, f'cache_{s}.done'))
                                    and os.path.exists(os.path.join(out, f'bank_{s}{sfx[model]}.pt')))
    V = lambda size, n, a, d, g, needs: steps.append(Step(f'vega-{size}/{n}', VP, a, d, size, g, needs))
    Lm = lambda model, n, a, d, g, needs: steps.append(Step(f'{model}/{n}', HERE, ['laya_peekaboo.py'] + a, d, model, g, needs))
    rows_from = lambda s: 'laya/cache/decisionbench' if s == 'decisionbench' else f'vega-0.8b/cache/{s}'
    vm = lambda size, s, m, sd, g, extra=(): V(size, f'{m}/{s}/s{sd}', ['adapt.py', '--suite', s, '--method', m, '--seed', str(sd)],
                                                ex(VEGA_RES[size], 'methods', s, f'{m}_s{sd}.json'), g, [f'vega-{size}/cache/{s}'] + list(extra))
    lm = lambda s, m, sd, g: Lm('laya', f'{m}/{s}/s{sd}', ['train', '--suite', s, '--method', m, '--seed', str(sd)],
                                ex('results_laya', 'methods', s, f'{m}_s{sd}.json'), g, ['laya/cache/' + s])
    # 1. read every suite once through every model; replay checks
    for s in SUITES:
        for size, res, gc, _ in VEGA:
            needs = [] if (size == '0.8b' and s != 'decisionbench') else [rows_from(s)]
            V(size, f'cache/{s}', ['suite_cache.py', '--suite', s], cached(size, res, s), gc, needs)
        for model in ('laya', 'laya_td'):
            needs = [] if (model == 'laya' and s == 'decisionbench') else [rows_from(s)]
            Lm(model, f'cache/{s}', ['cache', '--suite', s], cached(model, f'results_{model}', s), 4, needs)
    for s in SUITES:
        for size, res, _, gr in VEGA:
            V(size, f'replay/{s}', ['adapt.py', '--suite', s, '--method', 'replay'], ex(res, 'methods', s, 'replay_s0.json'), gr, [f'vega-{size}/cache/{s}'])
        Lm('laya', f'replay/{s}', ['train', '--suite', s, '--method', 'replay'], ex('results_laya', 'methods', s, 'replay_s0.json'), 4, [f'laya/cache/{s}'])
    # 2. Laya with the encoder unfrozen: the long pole, so it starts as soon as its cache exists
    for s in SUITES:
        for sd in SEEDS:
            for m in ('feedback_enc', 'sup_enc', 'feedback_enc_1pass'):
                lm(s, m, sd, 10)
    # 3. the main comparison: TTT (both rules), the answer-key add-on (+ early stopping), Peekaboo, both 8-pass and 1-pass
    for s in SUITES:
        for size, res, _, gr in VEGA:
            if s != 'jevbench':
                V(size, f'ttt/{s}', ['adapt.py', '--suite', s, '--method', 'ttt'], ex(res, 'methods', s, 'ttt_s0.json'), 0, [f'vega-{size}/cache/{s}'])
            for m in ('sup_ts', 'feedback', 'feedback_1pass', 'sup_1pass'):
                for sd in SEEDS:
                    vm(size, s, m, sd, gr)
        for m in ('sup_ts', 'feedback', 'feedback_1pass', 'sup_1pass'):
            for sd in SEEDS:
                lm(s, m, sd, 8)
    # 4. typed-decisions on Vega with the shipped typed_decisions adapter (fitted on this pool) switched off
    for size, res, _, gr in VEGA:
        V(size, 'noship_released/typed', ['adapt.py', '--suite', 'typed', '--method', 'noship_released'], ex(res, 'methods', 'typed', 'noship_released_s0.json'), gr, [f'vega-{size}/cache/typed'])
        for m in ('feedback_noship', 'sup_ts_noship', 'feedback_1pass_noship'):
            for sd in SEEDS:
                vm(size, 'typed', m, sd, gr)
    # 5. realistic feedback: noisy, missing, complaints only, and logged (argmax / sampled) answers learned off-policy
    for s in REAL_SUITES:
        for m in REALISTIC:
            for sd in SEEDS:
                for size, res, _, gr in VEGA:
                    vm(size, s, m, sd, gr)
                lm(s, m, sd, 8)
    # 6. ablation: no KL term; pooled Peekaboo
    for s in SUITES:
        for sd in SEEDS:
            for size, res, _, gr in VEGA:
                vm(size, s, 'feedback_kl0', sd, gr)
            lm(s, 'feedback_kl0', sd, 8)
    for sd in SEEDS:
        for size, res, _, gr in VEGA:
            V(size, f'pooled/s{sd}', ['adapt.py', '--suite', 'all', '--method', 'pooled', '--seed', str(sd)],
              ex(res, 'methods', f'pooled_s{sd}.done'), 2 * gr, [f'vega-{size}/cache/{x}' for x in SUITES])
        Lm('laya', f'pooled/s{sd}', ['train', '--suite', 'all', '--method', 'pooled', '--seed', str(sd)],
           ex('results_laya', 'methods', f'pooled_s{sd}.done'), 10, [f'laya/cache/{x}' for x in SUITES])
    steps.append(Step('tables', HERE, ['table.py'], lambda: False, None, 0))
    return steps


VEGA_RES = {'0.8b': 'results', '4b': 'results_4b'}


# ───────────────────────────── bookkeeping ─────────────────────────────
def log_path(out, st):
    return os.path.join(out, 'logs', st.name.replace('/', '__') + '.log')


def heartbeat(out, state):
    path = os.path.join(out, 'RUNNING')

    def beat():
        while not state.get('stop'):
            open(path, 'w').write(json.dumps({'pid': os.getpid(), 'running': state.get('running', []), 'beat': time.time()}))
            time.sleep(30)
    threading.Thread(target=beat, daemon=True).start()


def running(out):
    """The live run's heartbeat, or None. A heartbeat whose process is gone (killed) is stale, however recent."""
    try:
        hb = json.load(open(os.path.join(out, 'RUNNING')))
        if time.time() - hb['beat'] >= 120:
            return None
        if os.name == 'posix':
            os.kill(int(hb['pid']), 0)
        return hb
    except Exception:
        return None


def write_status(steps, out):
    by = {}
    for st in steps:
        by.setdefault(st.state, []).append(st)
    work = [st for st in steps if st.name != 'tables']
    L = [f'# Status of the remaining Peekaboo experiments  ({time.ctime()})', '',
         ' · '.join(f'{k}: {len(v)}' for k, v in sorted(by.items())) + f'  (of {len(steps)} steps)', '']
    if all(st.state == 'done' for st in work):
        L += ['COMPLETE. The tables are in TABLES.md.', '']
    else:
        L += ['NOT COMPLETE. Fix the cause of the first failed step (see its log), then run `python run.py` again;',
              'finished steps are skipped. Do not change methods, settings, seeds, splits or metrics. See CLAUDE.md.', '']
    for st in steps:
        if st.state in ('failed', 'blocked', 'pending', 'running'):
            L.append(f'- **{st.state}** `{st.name}`  {st.note}  (from `{os.path.relpath(st.cwd, HERE)}`: '
                     f'`{model_env(st)} python {" ".join(st.args)}`)')
            if st.state == 'failed' and os.path.exists(log_path(out, st)):
                tail = open(log_path(out, st), errors='replace').read().splitlines()[-25:]
                L += ['  ```'] + ['  ' + t for t in tail] + ['  ```']
    L += ['', '<details><summary>all steps</summary>', ''] + [f'- {st.state:8s} {st.name} {st.note}' for st in steps] + ['', '</details>']
    open(os.path.join(out, 'STATUS.md'), 'w').write('\n'.join(L) + '\n')


def model_env(st):
    return f'VEGA_OUT=<output> VEGA_SIZE={st.model}' if st.model else 'VEGA_OUT=<output>'


def ensure_packages():
    try:
        import vegaml, laya, transformers, datasets, matplotlib  # noqa: F401
        assert vegaml.__version__ == '0.8.0'
    except Exception:
        print('installing requirements.txt into', sys.executable, flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', os.path.join(HERE, 'requirements.txt')])


# ───────────────────────────── running ─────────────────────────────
def start(st, out, base_env):
    st.tries += 1; st.t0 = time.time(); st.state = 'running'
    env = dict(base_env, VEGA_SIZE=st.model or '0.8b')
    fh = open(log_path(out, st), 'a')
    fh.write(f'\n==== attempt {st.tries} {time.ctime()}  {" ".join(st.args)}\n'); fh.flush()
    st.proc = subprocess.Popen([sys.executable, '-u'] + st.args, cwd=st.cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    st.fh = fh
    print(f'[{time.strftime("%H:%M:%S")}] start  {st.name} (attempt {st.tries})', flush=True)


def finish(st):
    code = st.proc.returncode; st.fh.close(); st.proc = None
    if code == 0 and (st.name == 'tables' or st.done()):
        st.state, st.note = 'done', f'{time.time() - st.t0:.0f}s'
        print(f'[{time.strftime("%H:%M:%S")}] done   {st.name} ({st.note})', flush=True)
        return
    st.note = f'exit {code}' + ('' if code else ', but its result file is missing')
    st.state = 'failed' if st.tries >= TRIES else 'pending'
    print(f'[{time.strftime("%H:%M:%S")}] FAIL   {st.name}: {st.note} ({"gave up" if st.state == "failed" else "will retry"})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true'); ap.add_argument('--plan', action='store_true')
    ap.add_argument('--jobs', type=int, default=None); ap.add_argument('--out', default=None)
    ap.add_argument('--only', default=os.environ.get('ONLY', ''),
                    help='comma-separated step-name prefixes (e.g. vega-4b/); other steps are left alone and treated as satisfied')
    ap.add_argument('--grid', default='v2', choices=['v1', 'v2'], help='v1: the first paper\'s grid (output/); v2: the grid after the review (output_v2/)')
    a = ap.parse_args()
    out = os.path.abspath(a.out or os.environ.get('VEGA_OUT') or os.path.join(HERE, 'output' if a.grid == 'v1' else 'output_v2'))
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    if a.grid == 'v2':                               # the same rows as v1 for the four README benchmarks (emotion and AG News
        import shutil                                # pools are redrawn with the natural class mix, so they are rebuilt)
        for f in ('items_public.json', 'items_jevbench.json', 'items_decisionbench.json', 'items_typed.json', 'decisionbench_items.json', 'decisionbench_items_100.json'):
            src, dst = os.path.join(HERE, 'output', f), os.path.join(out, f)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)
    steps = plan(out) if a.grid == 'v1' else plan_v2(out)
    names = {st.name: st for st in steps}
    only = [p for p in a.only.split(',') if p]
    for st in steps:
        if st.name != 'tables' and st.done():
            st.state = 'done'
        elif only and st.name != 'tables' and not any(st.name.startswith(p) for p in only):
            st.state, st.note = 'done', 'not selected (--only)'
    if a.plan:
        for st in steps:
            print(f'{st.state:8s} {st.name:45s} gpu~{st.gpu:2d}GB  {model_env(st)} python {" ".join(st.args)}')
        print(len(steps), 'steps'); return
    live = running(out)
    work = [st for st in steps if st.name != 'tables']
    if a.check:
        if live:
            print(f"RUNNING: {', '.join(live['running']) or 'between steps'}; {sum(st.state == 'done' for st in work)}/{len(work)} steps done")
            sys.exit(2)
        write_status(steps, out); print(open(os.path.join(out, 'STATUS.md')).read())
        sys.exit(0 if all(st.state == 'done' for st in work) else 1)
    if live:
        sys.exit(f"another run is alive (pid {live['pid']}); check it with --check")
    ensure_packages()
    import torch
    gpu = torch.cuda.is_available() and os.environ.get('VEGA_DEVICE', 'cuda').startswith('cuda')
    budget = 0.9 * torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if gpu else 0
    jobs = a.jobs or (8 if gpu else 1)
    print(f'device: {torch.cuda.get_device_name(0) + f" ({budget / 0.9:.0f} GB)" if gpu else "CPU (the 4B will take days; use a GPU)"}; '
          f'{jobs} steps at a time; output {out}', flush=True)
    base_env = dict(os.environ, VEGA_OUT=out, PYTHONUNBUFFERED='1', HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false',
                    OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')   # without a cap, every step starts a thread per core and they thrash
    hb = {}; heartbeat(out, hb); t0 = time.time(); last_status = 0
    while True:
        active = [st for st in steps if st.state == 'running']
        for st in active:
            if st.proc.poll() is not None:
                finish(st)
        for st in steps:                                         # a step whose prerequisite gave up cannot run
            if st.state == 'pending' and any(names[n].state in ('failed', 'blocked') for n in st.needs):
                st.state, st.note = 'blocked', 'needs ' + ', '.join(n for n in st.needs if names[n].state != 'done')
        active = [st for st in steps if st.state == 'running']
        used = sum(st.gpu for st in active)
        for st in steps:
            if st.state != 'pending' or len(active) >= jobs or st.name == 'tables':
                continue
            if any(names[n].state != 'done' for n in st.needs):
                continue
            if active and gpu and used + st.gpu > budget:
                if st.gpu >= 12:                                 # a big step (a 4B cache) waits at the front of the queue and
                    break                                        # nothing passes it; small steps may pass each other
                continue
            start(st, out, base_env); active.append(st); used += st.gpu
        hb['running'] = [st.name for st in active]
        if time.time() - last_status > 60:
            write_status(steps, out); last_status = time.time()
        if not active and not any(st.state == 'pending' and st.name != 'tables' and all(names[n].state == 'done' for n in st.needs) for st in steps):
            break
        time.sleep(5)
    tables = names['tables']; start(tables, out, base_env); tables.proc.wait(); finish(tables)
    hb['stop'] = True
    try:
        os.remove(os.path.join(out, 'RUNNING'))
    except OSError:
        pass
    write_status(steps, out)
    left = [st for st in work if st.state != 'done']
    print(f'\n{len(work) - len(left)}/{len(work)} steps done in {(time.time() - t0) / 3600:.1f} h. '
          + (f'COMPLETE: {os.path.join(out, "TABLES.md")}' if not left else f'{len(left)} not done: see {os.path.join(out, "STATUS.md")}'), flush=True)
    sys.exit(0 if not left else 1)


if __name__ == '__main__':
    main()
