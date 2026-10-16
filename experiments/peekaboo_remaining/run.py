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


def plan(out):
    steps = []
    ex = lambda *p: (lambda: os.path.exists(os.path.join(out, *p)))
    for size, sfx, gpu_cache, gpu_run in (('0.8b', '', 8, 2), ('4b', '_4b', 24, 4)):
        res = 'results' + sfx
        add = lambda n, a, d, g, needs=(): steps.append(Step(f'vega-{size}/{n}', VP, a, d, size, g, [f'vega-{size}/{x}' for x in needs]))
        for s in ('public', 'jevbench', 'typed'):
            add(f'cache/{s}', ['suite_cache.py', '--suite', s], ex(res, f'cache_{s}.done'), gpu_cache)
            add(f'replay/{s}', ['adapt.py', '--suite', s, '--method', 'replay'], ex(res, 'methods', s, 'replay_s0.json'), gpu_run, [f'cache/{s}'])
            methods = ['feedback'] if s == 'jevbench' else ['sup_ts', 'feedback']
            if s != 'jevbench':
                add(f'ttt/{s}', ['adapt.py', '--suite', s, '--method', 'ttt'], ex(res, 'methods', s, 'ttt_s0.json'), 0, [f'cache/{s}'])
            for m in methods:
                for sd in SEEDS:
                    add(f'{m}/{s}/s{sd}', ['adapt.py', '--suite', s, '--method', m, '--seed', str(sd)],
                        ex(res, 'methods', s, f'{m}_s{sd}.json'), gpu_run, [f'cache/{s}'])
    for st in steps:                     # the 4B reads the row lists the 0.8B cache wrote, so the two never race
        if st.name.startswith('vega-4b/cache/'):
            st.needs.append('vega-0.8b/cache/' + st.name.split('/')[-1])
    # Laya on every benchmark: base Laya, its released fine-tune (laya-typed-decisions), and base Laya + Peekaboo
    L = lambda n, model, a, d, g, needs: steps.append(Step(f'{model}/{n}', HERE, ['laya_peekaboo.py'] + a, d, model, g, needs))
    for s in ('public', 'jevbench', 'decisionbench', 'typed'):
        rows = ['laya/cache/decisionbench'] if s == 'decisionbench' else [f'vega-0.8b/cache/{s}']   # one writer per row list
        L(f'cache/{s}', 'laya', ['cache', '--suite', s], ex('results_laya', f'cache_{s}.done'), 4, [] if s == 'decisionbench' else rows)
        L(f'cache/{s}', 'laya_td', ['cache', '--suite', s], ex('results_laya_td', f'cache_{s}.done'), 4, rows)
        L(f'replay/{s}', 'laya', ['train', '--suite', s, '--method', 'replay'], ex('results_laya', 'methods', s, 'replay_s0.json'), 4, [f'laya/cache/{s}'])
        for sd in SEEDS:
            L(f'feedback/{s}/s{sd}', 'laya', ['train', '--suite', s, '--method', 'feedback', '--seed', str(sd)],
              ex('results_laya', 'methods', s, f'feedback_s{sd}.json'), 8, [f'laya/cache/{s}'])
    steps.append(Step('tables', HERE, ['table.py'], lambda: False, None, 0))
    return steps


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
    try:
        hb = json.load(open(os.path.join(out, 'RUNNING')))
        return hb if time.time() - hb['beat'] < 120 else None
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
    a = ap.parse_args()
    out = os.path.abspath(a.out or os.environ.get('VEGA_OUT') or os.path.join(HERE, 'output'))
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    steps = plan(out)
    names = {st.name: st for st in steps}
    for st in steps:
        if st.name != 'tables' and st.done():
            st.state = 'done'
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
    jobs = a.jobs or (4 if gpu else 1)
    print(f'device: {torch.cuda.get_device_name(0) + f" ({budget / 0.9:.0f} GB)" if gpu else "CPU (the 4B will take days; use a GPU)"}; '
          f'{jobs} steps at a time; output {out}', flush=True)
    base_env = dict(os.environ, VEGA_OUT=out, PYTHONUNBUFFERED='1', HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false')
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
