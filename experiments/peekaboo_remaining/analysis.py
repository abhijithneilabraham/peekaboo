"""Post-hoc analysis of a finished grid (v1: output/, v2: output_v2/), from the per-row records only. Writes
<out>/analysis.json, which paper/make_figures.py reads. Everything here is CPU work on the stored probabilities.

  * gains and head-to-heads with intervals that resample training seeds as well as rows, rows resampled by
    cluster (public: task; DecisionBench: task; typed-decisions: case; JevBench: paraphrase group); Holm-adjusted
    p-values within each family of comparisons; non-inferiority at a 2-point margin
  * proper scores next to ECE: NLL, Brier, AURC, computed with VegaML's harness
  * the released model with the same feedback-fitted temperature (and with a label-fitted one): the control the
    calibration comparison needs
  * a label-prior baseline: per-task log-offsets on the released probabilities, trained with Peekaboo's own loop
    from right / wrong, and from the labels
  * how often the 8-pass protocol reveals the gold answer; the one-pass exploration cost against the base argmax
  * benchmark artifacts: the public menu without truthfulqa_mc1 (gold always first) and pubmedqa (majority class),
    DecisionBench without moderation-civil-comments (gold mostly first) and finqa-score (gold readable from the
    option structure); per-task DecisionBench gains
  * which rows a method fixes and breaks, by the base model's rank of the gold option; TTT on covered rows only

    VEGA_OUT=experiments/peekaboo_remaining/output_v2 python analysis.py
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VP = os.path.join(HERE, '..', 'vega_policy')
sys.path.insert(0, VP)
OUT = os.path.abspath(os.environ.get('VEGA_OUT') or os.path.join(HERE, 'output_v2'))
os.environ['VEGA_OUT'] = OUT
import paths  # noqa: E402
paths.harness()
import bench_common as bc  # noqa: E402

SUITES = ['public', 'jevbench', 'decisionbench', 'typed', 'emotion', 'agnews']
MODELS = ['0.8b', '4b', 'laya']
RES = {'0.8b': 'results', '4b': 'results_4b', 'laya': 'results_laya', 'laya_td': 'results_laya_td'}
TGRID = np.exp(np.linspace(np.log(0.25), np.log(32), 61))
REPS = int(os.environ.get('REPS', 2000))
ARTIFACT = {'public': ('truthfulqa_mc1', 'pubmedqa'), 'decisionbench': ('moderation-civil-comments', 'finqa-score')}
_items, _rel = {}, {}


# ───────────────────────────── loading ─────────────────────────────
def items(suite, split='test'):
    if suite not in _items:
        f = os.path.join(OUT, f'items_{suite}.json')
        _items[suite] = json.load(open(f))['items'] if os.path.exists(f) else []
    return [it for it in _items[suite] if it['split'] == split]


def released(model, suite):
    k = (model, suite)
    if k not in _rel:
        f = os.path.join(OUT, RES[model], f'released_{suite}.json')
        _rel[k] = json.load(open(f)) if os.path.exists(f) else None
    return _rel[k]


def run_files(model, suite, method):
    return sorted(glob.glob(os.path.join(OUT, RES[model], 'methods', suite, f'{method}_s[0-9].json')))


def runs(model, suite, method, key='records'):
    """List (one per seed) of record lists in test-row order; the released model is a single 'run'."""
    its = items(suite)
    if method == 'released':
        rel = released(model, suite)
        return [[rel[it['id']] for it in its]] if rel and all(it['id'] in rel for it in its) else []
    if method == 'laya_td':
        rel = released('laya_td', suite)
        return [[rel[it['id']] for it in its]] if rel and all(it['id'] in rel for it in its) else []
    out = []
    for f in run_files(model, suite, method):
        d = json.load(open(f))
        if key in d:
            out.append(d[key])
    return out


def infos(model, suite, method):
    return [json.load(open(f))['info'] for f in run_files(model, suite, method)]


def correct(recs):
    return np.array([float(r.get('pred') == r['gold']) for r in recs])


def clusters(suite):
    its = items(suite)
    if suite == 'public' or suite == 'decisionbench':
        key = lambda it: it['task']
    elif suite == 'typed':
        key = lambda it: it['id'].split(':')[0]
    elif suite == 'jevbench':
        key = lambda it: str(it.get('group') or it['id'])
    else:
        key = lambda it: it['id']
    groups = {}
    for i, it in enumerate(its):
        groups.setdefault(key(it), []).append(i)
    return [np.array(g) for g in groups.values()]


def task_groups(suite):
    its = items(suite)
    groups = {}
    for i, it in enumerate(its):
        groups.setdefault(it['task'], []).append(i)
    return {t: np.array(g) for t, g in groups.items()}


def stat(suite, d, mask=None):
    """The accuracy statistic over a per-row vector: macro over tasks on the public menu, micro elsewhere."""
    if mask is not None:
        d = np.where(mask, d, np.nan)
    if suite == 'public':
        return float(np.nanmean([np.nanmean(d[g]) for g in task_groups(suite).values() if np.isfinite(d[g]).any()]))
    return float(np.nanmean(d))


# ───────────────────────────── intervals ─────────────────────────────
def compare(suite, A, B, mask=None, reps=REPS, seed=0):
    """B minus A in accuracy points. A and B are lists of per-seed correctness vectors. The bootstrap resamples the
    seeds of each side (with replacement) and the row clusters; p is two-sided; Holm is applied later by family."""
    if not A or not B:
        return None
    A, B = [np.asarray(a) for a in A], [np.asarray(b) for b in B]
    rng = np.random.default_rng(seed)
    cl = clusters(suite)
    d_full = np.mean(B, 0) - np.mean(A, 0)
    est = 100 * stat(suite, d_full, mask)
    bs = []
    for _ in range(reps):
        a = np.mean([A[i] for i in rng.integers(len(A), size=len(A))], 0) if len(A) > 1 else A[0]
        b = np.mean([B[i] for i in rng.integers(len(B), size=len(B))], 0) if len(B) > 1 else B[0]
        d = b - a
        idx = np.concatenate([cl[j] for j in rng.integers(len(cl), size=len(cl))])
        if suite == 'public':           # macro over tasks: resample tasks, then rows within each task
            tg = task_groups(suite); keys = list(tg)
            vals = []
            for j in rng.integers(len(keys), size=len(keys)):
                g = tg[keys[j]]
                if mask is not None:
                    g = g[mask[g]]
                if len(g):
                    vals.append(d[rng.choice(g, len(g))].mean())
            bs.append(100 * float(np.mean(vals)))
        else:
            if mask is not None:
                idx = idx[mask[idx]]
            bs.append(100 * float(d[idx].mean()))
    bs = np.array(bs)
    p = 2 * min(float(np.mean(bs <= 0)), float(np.mean(bs >= 0)))
    return {'gain': est, 'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5)), 'p': min(1.0, p),
            'seeds': [len(A), len(B)], 'non_inferior_2': bool(np.percentile(bs, 2.5) > -2.0)}


def holm(family):
    """Holm step-down on a dict of name -> comparison (adds 'p_holm' and 'sig_holm')."""
    ps = [(k, v['p']) for k, v in family.items() if v]
    ps.sort(key=lambda x: x[1])
    m = len(ps); running = 0.0
    for i, (k, p) in enumerate(ps):
        adj = min(1.0, (m - i) * p); running = max(running, adj)
        family[k]['p_holm'] = running; family[k]['sig_holm'] = running < 0.05
    return family


# ───────────────────────────── metrics ─────────────────────────────
def probs_of(rec, labels):
    p = rec.get('probs')
    if not p:
        return None
    v = np.array([float(p.get(l, 0.0)) for l in labels], float)
    return v / v.sum() if v.sum() > 0 else None


def scores(suite, recs):
    """Accuracy, ECE, NLL, Brier, AURC with VegaML's harness (public: mean over tasks)."""
    its = items(suite)
    if suite == 'public':
        per = []
        for t, g in task_groups(suite).items():
            labels = its[g[0]]['labels']
            per.append(bc.score_metrics([recs[i] for i in g], labels, its[g[0]]['qtype'], 1.0 / max(len(labels), 2)))
        return {k: float(np.mean([p[k] for p in per if p.get(k) is not None])) for k in ('accuracy', 'ece', 'nll', 'brier', 'aurc')}
    m = bc.score_metrics(recs, None, its[0]['qtype'] if its else 'choice', None)
    return {k: m.get(k) for k in ('accuracy', 'ece', 'nll', 'brier', 'aurc')}


def balanced_accuracy(suite, recs):
    its = items(suite)
    by = {}
    for it, r in zip(its, recs):
        by.setdefault(it['gold'], []).append(float(r.get('pred') == it['gold']))
    return float(np.mean([np.mean(v) for v in by.values()]))


# ───────────────────────────── temperatures on the released model ─────────────────────────────
def temp(p, T):
    l = np.log(np.maximum(p, 1e-12)) / T
    e = np.exp(l - l.max()); return e / e.sum()


def cal_split(pool, cal_frac=0.2):
    """adapt.cal_split: per task, a fixed 20% of the pool held out (same permutation seed)."""
    rng = np.random.default_rng(0)
    train, cal = [], []
    for t in sorted({it['task'] for it in pool}):
        rows = [it for it in pool if it['task'] == t]
        rows = [rows[i] for i in rng.permutation(len(rows))]
        k = int(round(cal_frac * len(rows)))
        cal += rows[:k]; train += rows[k:]
    return train, cal


def fit_T(P, g, rng, how):
    """how='feedback': one answer sampled per row at T=1, Bernoulli likelihood of right / wrong; 'labels': NLL."""
    if how == 'labels':
        return float(min(TGRID, key=lambda T: -np.mean([np.log(max(temp(p, T)[gi], 1e-12)) for p, gi in zip(P, g)])))
    acts = [int(rng.choice(len(p), p=p)) for p in P]
    r = [float(a == gi) for a, gi in zip(acts, g)]

    def ll(T):
        tot = 0.0
        for p, a, ri in zip(P, acts, r):
            q = min(max(temp(p, T)[a], 1e-6), 1 - 1e-6); tot += ri * np.log(q) + (1 - ri) * np.log(1 - q)
        return tot
    return float(max(TGRID, key=ll))


def base_plus_T(model, suite, draws=10):
    """The released model's test records with a temperature fitted on its own calibration rows (20% of the pool):
    from right / wrong on sampled answers (mean over `draws` draws) and from the labels."""
    rel = released(model, suite)
    pool = [it for it in items(suite, 'pool') if rel and it['id'] in rel and rel[it['id']].get('probs')]
    its = items(suite)
    if not rel or not pool or not all(it['id'] in rel for it in its):
        return None
    _, cal = cal_split(pool)
    P = [probs_of(rel[it['id']], it['labels']) for it in cal]
    ok = [i for i, p in enumerate(P) if p is not None]
    P = [P[i] for i in ok]; g = [cal[i]['labels'].index(cal[i]['gold']) for i in ok]
    out = {}
    for how in ('feedback', 'labels'):
        Ts, ms = [], []
        for d in range(draws if how == 'feedback' else 1):
            T = fit_T(P, g, np.random.default_rng(d), how)
            recs = []
            for it in its:
                r = dict(rel[it['id']]); p = probs_of(r, it['labels'])
                if p is not None:
                    r['probs'] = {l: float(x) for l, x in zip(it['labels'], temp(p, T))}
                recs.append(r)
            Ts.append(T); ms.append(scores(suite, recs))
        out[how] = {'T': float(np.mean(Ts)), 'T_sd': float(np.std(Ts)), **{k: float(np.mean([m[k] for m in ms if m[k] is not None])) for k in ms[0]}}
    return out


# ───────────────────────────── label-prior baseline ─────────────────────────────
def prior_baseline(model, suite, seeds=(0, 1, 2)):
    """Per-task log-offsets on the released log-probabilities (one per (task, option text) seen in the pool), trained
    with Peekaboo's loop from right / wrong (REINFORCE, running baseline, KL 0.1 to the released probabilities, 8
    passes, batches of 32; the learning rate chosen on the calibration rows' right / wrong of the argmax answer) and,
    separately, from the labels (cross-entropy, same KL). No input features: this is what a label / position prior
    alone recovers. Returns per-seed correctness vectors on the test rows for both."""
    import torch
    rel = released(model, suite)
    its = items(suite)
    pool = [it for it in items(suite, 'pool') if rel and it['id'] in rel and rel[it['id']].get('probs')]
    if not rel or not pool or not all(it['id'] in rel and rel[it['id']].get('probs') for it in its):
        return None
    train, cal = cal_split(pool)
    keys = {}
    for it in pool:
        for lab in it['labels']:
            keys.setdefault((it['task'], lab), len(keys))
    K = max(len(it['labels']) for it in its + pool)

    def tens(rows):
        lp = torch.full((len(rows), K), -1e4); idx = torch.full((len(rows), K), -1, dtype=torch.long); g = torch.zeros(len(rows), dtype=torch.long)
        for i, it in enumerate(rows):
            p = probs_of(rel[it['id']], it['labels'])
            for j, lab in enumerate(it['labels']):
                lp[i, j] = float(np.log(max(p[j], 1e-12))); idx[i, j] = keys.get((it['task'], lab), -1)
            g[i] = it['labels'].index(it['gold'])
        return lp, idx, g
    lp_tr, idx_tr, g_tr = tens(train); lp_ca, idx_ca, g_ca = tens(cal); lp_te, idx_te, g_te = tens(its)

    def logits(b, lp, idx):
        off = torch.where(idx >= 0, b[idx.clamp(min=0)], torch.zeros_like(lp)); return torch.log_softmax(lp + off, -1)

    def train_prior(how, lr, seed):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        b = torch.zeros(len(keys), requires_grad=True); opt = torch.optim.Adam([b], lr=lr); base = 0.5
        for ep in range(8):
            order = rng.permutation(len(train))
            for s in range(0, len(train), 32):
                sel = torch.as_tensor(order[s:s + 32])
                lq = torch.log_softmax(lp_tr[sel], -1); lp = logits(b, lp_tr[sel], idx_tr[sel]); ar = torch.arange(len(sel))
                if how == 'labels':
                    loss = -lp[ar, g_tr[sel]].mean()
                else:
                    with torch.no_grad():
                        a = torch.multinomial(lp.exp(), 1).squeeze(1); r = (a == g_tr[sel]).float()
                    loss = -((r - base) * lp[ar, a]).mean(); base = 0.9 * base + 0.1 * float(r.mean())
                loss = loss + 0.1 * (lp.exp() * (lp - lq)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return b.detach()
    out = {}
    for how in ('feedback', 'labels'):
        per_seed, lrs = [], []
        for seed in seeds:
            best = None
            for lr in (1e-3, 1e-2, 3e-2, 1e-1):
                b = train_prior(how, lr, seed)
                cal_acc = float((logits(b, lp_ca, idx_ca).argmax(-1) == g_ca).float().mean())   # argmax rated right / wrong
                if best is None or cal_acc > best[0]:
                    best = (cal_acc, lr, b)
            pred = logits(best[2], lp_te, idx_te).argmax(-1)
            per_seed.append((pred == g_te).float().numpy().tolist()); lrs.append(best[1])
        out[how] = {'correct': per_seed, 'lr': lrs, 'params': len(keys)}
    return out


# ───────────────────────────── reveal statistics, exploration cost ─────────────────────────────
def reveal(model, suite, method='feedback'):
    rel = released(model, suite)
    inf = infos(model, suite, method)
    if not inf:
        return None
    curves = [f.get('reward_curve') for i in inf for f in i.get('folds', []) if f.get('reward_curve')]
    curves = [[x for x in c if x is not None] for c in curves]
    curves = [c for c in curves if c]
    out = {}
    if curves:
        out['reveals_per_row'] = float(np.mean([sum(c) for c in curves]))          # expected reward-1 events per training row
        out['max_pass_reward'] = float(np.mean([max(c) for c in curves]))          # a lower bound on the share of rows revealed
        out['pass1_reward'] = float(np.mean([c[0] for c in curves]))
        out['passes'] = int(np.mean([len(c) for c in curves]))
    pool = [it for it in items(suite, 'pool') if rel and it['id'] in rel and rel[it['id']].get('probs')]
    if pool:
        train, _ = cal_split(pool)
        pg = np.array([probs_of(rel[it['id']], it['labels'])[it['labels'].index(it['gold'])] for it in train])
        out['base_sampled_acc_train'] = float(pg.mean())
        out['base_argmax_acc_train'] = float(np.mean([rel[it['id']]['pred'] == it['gold'] for it in train]))
        out['p_reveal_8_base_policy'] = float(np.mean(1 - (1 - pg) ** 8))
        out['p_reveal_1_base_policy'] = float(pg.mean())
        wrong = np.array([rel[it['id']]['pred'] != it['gold'] for it in train])
        out['p_reveal_8_base_wrong_rows'] = float(np.mean(1 - (1 - pg[wrong]) ** 8)) if wrong.any() else None
        out['binary_share'] = float(np.mean([len(it['labels']) == 2 for it in train]))
    return out


# ───────────────────────────── fixes and breaks, TTT coverage ─────────────────────────────
def fix_break(model, suite, method, key='records'):
    rel = released(model, suite); its = items(suite)
    R = runs(model, suite, method, key)
    if not rel or not R:
        return None
    base = correct([rel[it['id']] for it in its])
    ranks = []
    for it in its:
        p = probs_of(rel[it['id']], it['labels'])
        ranks.append(int((p > p[it['labels'].index(it['gold'])]).sum()) + 1 if p is not None else 99)
    ranks = np.array(ranks)
    c = np.mean([correct(r) for r in R], 0)
    out = {'fix_rate': float(np.mean(c[base == 0])) if (base == 0).any() else None, 'break_rate': float(1 - np.mean(c[base == 1])) if (base == 1).any() else None}
    for name, m in (('rank2', ranks == 2), ('rank3', ranks == 3), ('rank4plus', ranks >= 4)):
        m = m & (base == 0)
        out[f'fix_rate_{name}'] = float(np.mean(c[m])) if m.any() else None
    return out


def ttt_coverage(model, suite, key='records'):
    its = items(suite)
    T = runs(model, suite, 'ttt', key)
    if not T:
        return None
    covered = np.array([r.get('readout') == 'ttt' for r in T[0]])
    P = [correct(r) for r in runs(model, suite, 'feedback')]
    S = [correct(r) for r in runs(model, suite, 'sup_ts')]
    out = {'rows_covered': int(covered.sum()), 'rows': len(its)}
    if P:
        out['peekaboo_vs_ttt_covered'] = compare(suite, [correct(T[0])], P, mask=covered)
        out['peekaboo_vs_ttt_uncovered'] = compare(suite, [correct(T[0])], P, mask=~covered)
    if S:
        out['adapter_vs_ttt_covered'] = compare(suite, [correct(T[0])], S, mask=covered)
    return out


# ───────────────────────────── main ─────────────────────────────
def main():
    N = {'compare': {}, 'scores': {}, 'base_T': {}, 'prior': {}, 'reveal': {}, 'fixbreak': {}, 'ttt': {}, 'tasks_db': {}, 'switch': {}, 'balanced': {}}
    C = lambda m, s, a, b, ka='records', kb='records', mask=None: compare(s, [correct(r) for r in runs(m, s, a, ka)], [correct(r) for r in runs(m, s, b, kb)], mask)
    # every method against base, and the named head-to-heads; artifact-free versions on public and DecisionBench
    methods = {m: sorted({os.path.basename(f).rsplit('_s', 1)[0] for s in SUITES for f in glob.glob(os.path.join(OUT, RES[m], 'methods', s, '*_s[0-9].json'))}) for m in MODELS}
    def keys_of(m, s, meth):
        ks = [k for k in ('records', 'records_raw', 'records_switch', 'records_es', 'records_guard') if runs(m, s, meth, k)]
        if 'records_switch' in ks and not any(f.get('switch') for i in infos(m, s, meth) for f in i.get('folds', [])):
            ks.remove('records_switch')              # answer-key runs carry a copy of their records under this key; no switch ran
        return ks
    for m in MODELS:
        for s in SUITES:
            rel_ok = bool(runs(m, s, 'released'))
            for meth in [x for x in methods[m] if x != 'replay'] + (['laya_td'] if m == 'laya' else []):
                for k in (keys_of(m, s, meth) if meth != 'laya_td' else ['records']):
                    R = runs(m, s, meth, k)
                    if not R:
                        continue
                    tag = f'{m}|{meth}|{k}|{s}'
                    sc = [scores(s, r) for r in R]
                    N['scores'][tag] = {kk: [x[kk] for x in sc] for kk in sc[0]}
                    if s in ('emotion', 'public'):
                        N['balanced'][tag] = [balanced_accuracy(s, r) for r in R]
                    if rel_ok:
                        N['compare'][f'{tag}|vs|released'] = C(m, s, 'released', meth, 'records', k)
                        if s in ARTIFACT:
                            mask = np.array([not any(a in it['task'] or a in it['id'] for a in ARTIFACT[s]) for it in items(s)])
                            N['compare'][f'{tag}|vs|released|noartifact'] = C(m, s, 'released', meth, 'records', k, mask)
            if rel_ok:
                N['scores'][f'{m}|released|records|{s}'] = {kk: [v] for kk, v in scores(s, runs(m, s, 'released')[0]).items()}
                if s in ('emotion', 'public'):
                    N['balanced'][f'{m}|released|records|{s}'] = [balanced_accuracy(s, runs(m, s, 'released')[0])]
            # head-to-heads: Peekaboo (8-pass, 1-pass) against the answer-key arms and their own method
            theirs = 'laya_td' if m == 'laya' else 'ttt'
            pairs = [('feedback', theirs, 'records', 'records'), ('feedback', theirs, 'records', 'records_guard'), ('feedback', 'sup_ts', 'records', 'records'),
                     ('feedback', 'sup_ts', 'records', 'records_es'), ('feedback_1pass', 'sup_1pass', 'records', 'records'), ('feedback_1pass', 'feedback', 'records', 'records'),
                     ('feedback', 'feedback_kl0', 'records', 'records'), ('feedback', 'pooled', 'records', 'records'),
                     ('feedback_enc', 'sup_enc', 'records', 'records'), ('feedback_enc', 'sup_enc', 'records', 'records_es'), ('feedback_enc', theirs, 'records', 'records'),
                     ('feedback_enc_1pass', 'feedback_enc', 'records', 'records'), ('feedback_enc', 'feedback', 'records', 'records'),
                     ('feedback_noship', 'sup_ts_noship', 'records', 'records'), ('feedback_noship', 'noship_released', 'records', 'records'),
                     ('sup_ts_noship', 'noship_released', 'records', 'records'), ('feedback_1pass_noship', 'noship_released', 'records', 'records'),
                     ('logged_argmax', 'feedback_1pass', 'records', 'records'), ('logged_ips', 'feedback_1pass', 'records', 'records'),
                     ('feedback_noise20', 'feedback', 'records', 'records'), ('feedback_half', 'feedback', 'records', 'records'), ('feedback_complaints', 'feedback', 'records', 'records'),
                     ('feedback', 'feedback', 'records_switch', 'records'),
                     # (a, b, ka, kb): a minus b
                     ('sup_enc', 'laya_td', 'records', 'records'), ('sup_enc', 'laya_td', 'records_es', 'records'), ('sup_enc', 'feedback_enc_1pass', 'records', 'records'),
                     ('sup_ts', 'ttt', 'records', 'records_guard'), ('sup_ts', 'ttt', 'records_es', 'records_guard'), ('feedback_1pass', theirs, 'records', 'records_guard'),
                     ('logged_argmax', 'sup_1pass', 'records', 'records'), ('logged_argmax', 'feedback', 'records', 'records'), ('sup_1pass', 'sup_ts', 'records', 'records'),
                     ('feedback_enc_1pass', 'laya_td', 'records', 'records'), ('feedback_enc_1pass', 'sup_enc', 'records', 'records'), ('sup_ts_noship', 'sup_ts', 'records', 'records')]
            for a, b, ka, kb in pairs:
                v = C(m, s, b, a, kb, ka)       # a minus b
                if v:
                    N['compare'][f'{m}|{a}|{ka}|{s}|vs|{b}|{kb}'] = v
            # the released model with the same feedback-fitted temperature; the prior baseline
            if rel_ok and s != 'jevbench':
                bt = base_plus_T(m, s)
                if bt:
                    N['base_T'][f'{m}|{s}'] = bt
                pr = prior_baseline(m, s)
                if pr:
                    N['prior'][f'{m}|{s}'] = {h: {'acc': [float(np.mean(c)) for c in v['correct']], 'lr': v['lr'], 'params': v['params'],
                                                 'vs_released': compare(s, [correct(runs(m, s, 'released')[0])], v['correct']),
                                                 'peekaboo_vs': compare(s, v['correct'], [correct(r) for r in runs(m, s, 'feedback')]),
                                                 'adapter_vs': compare(s, v['correct'], [correct(r) for r in runs(m, s, 'sup_ts')])} for h, v in pr.items()}
            for meth in ('feedback', 'feedback_1pass', 'feedback_enc', 'feedback_enc_1pass'):
                rv = reveal(m, s, meth)
                if rv:
                    N['reveal'][f'{m}|{meth}|{s}'] = rv
            for meth in ('feedback', 'sup_ts', 'feedback_1pass', 'feedback_enc', 'sup_enc'):
                fb = fix_break(m, s, meth)
                if fb:
                    N['fixbreak'][f'{m}|{meth}|{s}'] = fb
            if m != 'laya' and s != 'jevbench':
                tc = ttt_coverage(m, s)
                if tc:
                    N['ttt'][f'{m}|{s}'] = tc
            for meth in methods[m]:
                sw = [f.get('switch') for i in infos(m, s, meth) for f in i.get('folds', []) if f.get('switch')]
                if sw:
                    N['switch'][f'{m}|{meth}|{s}'] = {'on': float(np.mean([x['on'] for x in sw])), 'n': len(sw),
                                                      'sampled_on': float(np.mean([x['sampled']['on'] for x in sw if x.get('sampled')])) if any(x.get('sampled') for x in sw) else None}
    # per-task DecisionBench gains (Peekaboo and the answer-key adapter), both Vega sizes
    for m in ('0.8b', '4b'):
        rel = released(m, 'decisionbench')
        if not rel:
            continue
        its = items('decisionbench'); base = correct([rel[it['id']] for it in its])
        for meth in ('feedback', 'sup_ts', 'ttt'):
            R = runs(m, 'decisionbench', meth)
            if not R:
                continue
            c = np.mean([correct(r) for r in R], 0)
            N['tasks_db'][f'{m}|{meth}'] = {t: {'base': float(base[g].mean()), 'acc': float(c[g].mean()), 'n': int(len(g))} for t, g in task_groups('decisionbench').items()}
    # Holm within the families a reader will count
    fam = {}
    for k, v in N['compare'].items():
        if not v:
            continue
        parts = k.split('|')
        if '|vs|' in k and parts[1] in ('feedback', 'feedback_enc') and parts[2] == 'records' and len(parts) >= 7 and parts[5] in ('ttt', 'laya_td', 'sup_ts', 'sup_enc') and parts[6] == 'records':
            fam.setdefault(f'{parts[0]}|{parts[1]}|vs|{parts[5]}', {})[k] = v
        if parts[-1] == 'released' and parts[1] in ('feedback', 'feedback_1pass', 'feedback_enc') and parts[2] == 'records':
            fam.setdefault(f'{parts[0]}|{parts[1]}|vs|released', {})[k] = v
        if '|vs|feedback_kl0|' in k:
            fam.setdefault('kl', {})[k] = v
    for name, f in fam.items():
        holm(f)
    N['families'] = {name: sorted(f) for name, f in fam.items()}
    json.dump(N, open(os.path.join(OUT, 'analysis.json'), 'w'), indent=0)
    print('wrote', os.path.join(OUT, 'analysis.json'), {k: len(v) for k, v in N.items()})


if __name__ == '__main__':
    main()
