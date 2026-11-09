"""Figures, LaTeX tables and number macros for the paper, from a finished grid and its analysis.json
(experiments/peekaboo_remaining/analysis.py). Every number in the paper comes from here.

    VEGA_OUT=experiments/peekaboo_remaining/output_v2 python paper/make_figures.py
"""
import json, os, sys
import numpy as np
import matplotlib, matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PR = os.path.join(REPO, 'experiments', 'peekaboo_remaining')
os.environ.setdefault('VEGA_OUT', os.path.join(PR, 'output_v2'))
sys.path.insert(0, PR)
import analysis as A   # noqa: E402

OUT = A.OUT
N = json.load(open(os.path.join(OUT, 'analysis.json')))
FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
SUITES = A.SUITES
SNAME = {'public': '43 public tasks', 'jevbench': 'JevBench', 'decisionbench': 'DecisionBench', 'typed': 'typed-decisions', 'emotion': 'emotion', 'agnews': 'AG News'}
SSHORT = {'public': 'public', 'jevbench': 'JevBench', 'decisionbench': 'DecisionB.', 'typed': 'typed-dec.', 'emotion': 'emotion', 'agnews': 'AG News'}
MODELS = [('0.8b', 'Vega 0.8B'), ('4b', 'Vega 4B'), ('laya', 'Laya')]
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})
C = {'base': '#9e9e9e', 'theirs': '#e07b39', 'sup': '#4c72b0', 'sup_es': '#7fa6d8', 'pk': '#2a9d3f', 'pk1': '#8fd19e', 'raw': '#b5d99c',
     'kl0': '#c44e52', 'pooled': '#8172b2', 'enc': '#1b5e20', 'enc1': '#66bb6a', 'prior': '#b58900', 'logged': '#6c757d'}
MARK = {'theirs': 'D', 'sup': 's', 'sup_es': 's', 'pk': 'o', 'pk1': 'o', 'enc': 'o', 'enc1': 'o', 'prior': '^', 'logged': 'v', 'pooled': 's', 'kl0': 'x'}


PK = lambda model: 'feedback_enc' if model == 'laya' else 'feedback'          # the main Peekaboo arm per model in this grid
SUP = lambda model: 'sup_enc' if model == 'laya' else 'sup_ts'


# ───────────────────────────── access ─────────────────────────────
def cmp(model, meth, key, suite, vs='released', vkey='records', suffix=''):
    k = f'{model}|{meth}|{key}|{suite}|vs|{vs}' + ('' if vs == 'released' else f'|{vkey}') + suffix
    return N['compare'].get(k)


def sc(model, meth, key, suite, what):
    v = N['scores'].get(f'{model}|{meth}|{key}|{suite}')
    return None if not v or not v.get(what) or v[what][0] is None else v[what]


def fg(g, ci=True, mark=True):
    if not g:
        return '--'
    s = f'{g["gain"]:+.1f}'
    if ci:
        s += f' {{\\scriptsize[{g["lo"]:+.1f}, {g["hi"]:+.1f}]}}'
    if mark and g.get('sig_holm'):
        s += '$^{\\dagger}$'
    return s


def f3(v):
    return '--' if not v else f'{np.mean(v):.3f}' + (f' {{\\scriptsize$\\pm${np.std(v, ddof=1):.3f}}}' if len(v) > 1 else '')


# ───────────────────────────── figures ─────────────────────────────
# Figure sizes are close to the printed page width so labels remain readable.
# Missing observations stay distinct from numerical zero throughout these plots.
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
import textwrap

PALETTE = {'feedback': '#16734A', 'labels': '#3165A8', 'vendor': '#B46816',
           'prior': '#8154A1', 'base': '#676767', 'raw': '#BF4D36'}
GAIN_CMAP = LinearSegmentedColormap.from_list('gain', ['#D7986C', '#FFFFFF', '#3D9776'])
LOWER_CMAP = LinearSegmentedColormap.from_list('error', ['#FFFFFF', '#F2C7AC', '#A84127'])
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'legend.fontsize': 8.5, 'axes.titlepad': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def save_figure(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def signed(value):
    return f'{value:+.1f}' if abs(value) >= 0.05 else '0.0'


def clean_axis(ax, ylabels, zero=True):
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    ax.set_ylim(len(ylabels) - 0.5, -0.5)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0, pad=6)
    ax.grid(axis='x', color='#E3E3E3', lw=0.6); ax.set_axisbelow(True)
    if zero:
        ax.axvline(0, color='#555555', lw=0.9)


def interval(ax, estimate, y, color, marker='o', filled=True):
    if estimate is None:
        return
    ax.errorbar(estimate['gain'], y,
                xerr=[[estimate['gain'] - estimate['lo']], [estimate['hi'] - estimate['gain']]],
                fmt=marker, ms=5, mfc=color if filled else 'white', mec=color,
                color=color, ecolor=color, lw=1.1, capsize=2, zorder=3)


def matrix(ax, values, row_labels, col_labels, norm, cmap, fmt):
    values = np.array(values, dtype=float)
    cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cm.set_bad('#E8E8E8')
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cm, norm=norm, aspect='auto')
    for (row, col), value in np.ndenumerate(values):
        label = fmt(value) if np.isfinite(value) else '—'
        rgba = cm(norm(value)) if np.isfinite(value) else (0.9, 0.9, 0.9, 1)
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        ax.text(col, row, label, va='center', ha='center', fontsize=9,
                color='white' if lum < .52 else '#202020')
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8.2)
    ax.tick_params(length=0, pad=7)
    ax.set_xticks(np.arange(-.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, values.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', lw=2); ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    return im


def fig_training():
    fig, ax = plt.subplots(figsize=(7.2, 4.7)); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    def box(x, y, w, h, title, subtitle, train=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008',
                     facecolor='#E2F0E8' if train else '#F1F3F5', edgecolor='#65717C', lw=.9))
        ax.text(x+w/2, y+h*.68, title, ha='center', va='center', fontsize=9.2, weight='bold')
        ax.text(x+w/2, y+h*.28, subtitle, ha='center', va='center', fontsize=8.1)
    def arrow(a, b, label=None, offset=(0, 0)):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11,
                                    color='#475763', lw=1.2))
        if label:
            ax.text((a[0]+b[0])/2+offset[0], (a[1]+b[1])/2+offset[1], label,
                    ha='center', va='center', fontsize=7.6, backgroundcolor='white')
    ax.text(0, .98, 'Vega training from correctness feedback', fontsize=12, weight='bold', va='top')
    ax.text(0, .91, 'Gray: fixed computation or data. Green: learned adjustment and its loss.', fontsize=8.5)
    box(.01, .69, .17, .15, 'Text input', 'State, question\nand options')
    box(.23, .69, .19, .15, 'Backbone', 'Frozen features\ncan be cached')
    box(.48, .69, .25, .15, 'Decision engine', 'Frozen weights +\ntrainable adapter / gate', True)
    box(.80, .69, .18, .15, 'Sample', 'Draw from option\nprobabilities')
    arrow((.18,.765),(.23,.765)); arrow((.42,.765),(.48,.765)); arrow((.73,.765),(.80,.765))
    box(.23,.39,.19,.15,'Reference','New adapter off;\nfixed probabilities')
    box(.48,.39,.25,.15,'Training loss','REINFORCE + KL\nUpdate adapter / gate', True)
    box(.80,.39,.18,.15,'Verdict','Right = 1\nWrong = 0')
    arrow((.325,.69),(.325,.54)); arrow((.42,.465),(.48,.465),'KL anchor',(0,.105))
    arrow((.89,.69),(.89,.54)); arrow((.80,.465),(.73,.465))
    arrow((.605,.54),(.605,.69),'gradient',(.09,0))
    ax.text(.89,.32,'Gold answer simulates\nthe external rater',ha='center',va='center',fontsize=8)
    arrow((.89,.36),(.89,.39))
    ax.text(.01,.20,'One pass',weight='bold',fontsize=9)
    ax.text(.21,.20,'One new sampled answer and verdict per training question.',fontsize=8.5)
    ax.text(.01,.13,'Eight passes',weight='bold',fontsize=9)
    ax.text(.21,.13,'Repeat each training question with fresh samples and verdicts.',fontsize=8.5)
    ax.text(.01,.03,'After training: held-out verdicts → temperature fitting → optional model-level switch.',fontsize=8.5)
    save_figure(fig,'training')


def fig_gains():
    """One interval per benchmark: directly answer which method performs better."""
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.7), sharex=True, sharey=True)
    comparisons = [('sup_ts', 'records_es', 'labelled adapter'),
                   ('ttt', 'records_guard', 'TTT head (0.12.3)')]
    limits = []
    for row, (other, key, label) in enumerate(comparisons):
        for col, (model, name) in enumerate(MODELS[:2]):
            ax=axes[row,col]; clean_axis(ax,[SNAME[s] for s in SUITES])
            for y,s in enumerate(SUITES):
                g=cmp(model,'feedback','records',s,other,key)
                if g:
                    limits.extend([g['lo'],g['hi']]); interval(ax,g,y,PALETTE['feedback'])
                else:
                    ax.text(0,y,'not evaluated',ha='center',va='center',fontsize=8,color='#666666')
            ax.set_title(f'{name}\nPeekaboo minus {label}',fontsize=9.5)
            if col: ax.tick_params(labelleft=False)
    lo=min(limits)-1; hi=max(limits)+1
    for ax in axes.flat: ax.set_xlim(lo,hi)
    for ax in axes[-1]: ax.set_xlabel('Accuracy difference (percentage points)')
    fig.subplots_adjust(left=.21,right=.98,top=.82,bottom=.12,hspace=.65,wspace=.17)
    fig.text(.60,.99,'Eight fresh-feedback passes versus labelled methods',ha='center',va='top',weight='bold',fontsize=11)
    fig.text(.60,.025,'← Comparator better                       Peekaboo better →',ha='center',fontsize=9)
    save_figure(fig,'gains')


def fig_passes():
    """Explicit column labels avoid interpreting line endpoints as uncertainty bars."""
    fig,axes=plt.subplots(1,3,figsize=(7.3,3.5),gridspec_kw={'width_ratios':[4,4,2]})
    norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=35)
    for ax,(model,name) in zip(axes,MODELS):
        methods=['feedback_1pass','feedback','sup_1pass','sup_ts'] if model!='laya' else ['feedback_enc_1pass','feedback_enc']
        values=[[cmp(model,m,'records',s)['gain'] if cmp(model,m,'records',s) else np.nan for m in methods] for s in SUITES]
        cols=['1 pass\nverdicts','8 passes\nverdicts','1 pass\nlabels','8 passes\nlabels'][:len(methods)]
        matrix(ax,values,[SNAME[s] for s in SUITES] if model=='0.8b' else ['']*6,cols,norm,GAIN_CMAP,signed)
        ax.set_title(name+('\nencoder trained' if model=='laya' else ''))
    fig.subplots_adjust(left=.18,right=.99,bottom=.26,top=.79,wspace=.14)
    fig.text(.58,.98,'Mean accuracy gain over the released model (percentage points)',ha='center',va='top',weight='bold',fontsize=10)
    fig.text(.58,.035,'Positive = improved accuracy. Orange = loss. Gray / — = not run.',ha='center',fontsize=8.5)
    save_figure(fig,'passes')


def fig_realistic():
    arms=[('feedback','Fresh feedback, 8 passes'),('feedback_1pass','Fresh feedback, 1 pass'),
          ('logged_argmax','Logged best answer'),('logged_ips','Logged sampled answer'),
          ('feedback_noise20','20% wrong verdicts'),('feedback_half','50% of answers rated'),
          ('feedback_complaints','All errors + 20% of correct\nanswers rated')]
    suites=['decisionbench','typed','emotion']
    fig,axes=plt.subplots(2,3,figsize=(7.4,5.7),sharex=True,sharey=True)
    limits=[]
    for row,(model,name) in enumerate(MODELS[:2]):
        for col,s in enumerate(suites):
            ax=axes[row,col]; clean_axis(ax,[a[1] for a in arms])
            for y,(meth,_) in enumerate(arms):
                g=cmp(model,meth,'records',s)
                if g is None: continue
                limits.extend([g['lo'],g['hi']])
                failed=(model=='4b' and s=='emotion' and meth=='feedback_half')
                interval(ax,g,y,PALETTE['raw'] if failed else PALETTE['feedback'],
                         marker='x' if failed else 'o',filled=g['seeds'][1]>1)
                if failed: ax.annotate('failed run',(g['gain'],y),xytext=(6,0),textcoords='offset points',va='center',fontsize=7.5,color=PALETTE['raw'])
            ax.set_title(f'{name}\n{SNAME[s]}',fontsize=9.5)
            if col: ax.tick_params(labelleft=False)
            ax.tick_params(axis='x',labelsize=8)
    lo=np.floor(min(limits)/5)*5;hi=np.ceil(max(limits)/5)*5
    for ax in axes.flat: ax.set_xlim(lo,hi);ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    for ax in axes[-1]: ax.set_xlabel('Accuracy gain (pp)',fontsize=9)
    fig.subplots_adjust(left=.29,right=.98,bottom=.14,top=.80,wspace=.20,hspace=.50)
    fig.text(.63,.99,'Feedback collection conditions',ha='center',va='top',fontsize=11,weight='bold')
    fig.text(.63,.94,'Zero = released model; lines = 95% intervals',ha='center',fontsize=8.5)
    fig.text(.63,.035,'Open circles: one training seed. Filled circles: three seeds.',ha='center',fontsize=8.5)
    save_figure(fig,'realistic')


def fig_anatomy():
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.1),gridspec_kw={'height_ratios':[1,1.25]})
    ranks=['fix_rate_rank2','fix_rate_rank3','fix_rate_rank4plus']
    for col,(model,name) in enumerate(MODELS[:2]):
        ax=axes[0,col]; x=np.arange(3); w=.34
        for offset,meth,label,color in [(-.5,'feedback','Feedback',PALETTE['feedback']),(.5,'sup_ts','Labels',PALETTE['labels'])]:
            vals=[]
            for rank in ranks:
                present=[N['fixbreak'].get(f'{model}|{meth}|{s}',{}).get(rank) for s in SUITES if s!='jevbench']
                vals.append(100*np.mean([v for v in present if v is not None]))
            bars=ax.bar(x+offset*w,vals,w,color=color,label=label)
            ax.bar_label(bars,fmt='%.0f%%',fontsize=8,padding=2)
        ax.set_xticks(x);ax.set_xticklabels(['2nd','3rd','4th or lower'])
        ax.set_ylim(0,60);ax.set_title(f'{name}: which errors are fixed',fontsize=9.5)
        ax.set_xlabel('Correct answer’s rank before training',fontsize=8.5)
        ax.set_ylabel('Initially wrong rows fixed (%)',fontsize=8.5)
        ax.legend(frameon=False,fontsize=8,loc='upper right');ax.grid(axis='y',color='#EEEEEE');ax.set_axisbelow(True)
        ax=axes[1,col]; suites=[s for s in SUITES if N['prior'].get(f'{model}|{s}')]
        clean_axis(ax,[SNAME[s] for s in suites])
        for y,s in enumerate(suites):
            prior=N['prior'][f'{model}|{s}']['feedback'].get('vs_released')
            full=cmp(model,'feedback','records',s)
            if prior and full:
                ax.plot([prior['gain'],full['gain']],[y,y],color='#B8B8B8',lw=1.4)
                ax.plot(prior['gain'],y,'^',color=PALETTE['prior'],ms=6)
                ax.plot(full['gain'],y,'o',color=PALETTE['feedback'],ms=5)
        ax.set_xlim(-3,14);ax.set_xlabel('Accuracy gain over released model (pp)',fontsize=8.5)
        ax.set_title(f'{name}: prior correction versus full adapter',fontsize=9.2)
    fig.subplots_adjust(left=.17,right=.98,top=.95,bottom=.13,hspace=.65,wspace=.65)
    handles=[Line2D([],[],marker='^',color=PALETTE['prior'],ls='',label='No-input prior from verdicts'),
             Line2D([],[],marker='o',color=PALETTE['feedback'],ls='',label='Full Peekaboo adapter')]
    fig.legend(handles=handles,frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.59,.015),fontsize=8.3)
    save_figure(fig,'anatomy')


def fig_calibration():
    bars=[('released','records','Base'),('baseT','feedback','Base\n+ T'),
          ('feedback','records_raw','Pk raw'),('feedback','records','Pk\n+ T'),('sup_ts','records','Labels\n+ T')]
    fig,axes=plt.subplots(3,2,figsize=(7.3,7.0))
    for col,what in enumerate(('ece','aurc')):
        data=[]
        for model,name in MODELS:
            rows=[]
            for s in SUITES:
                vals=[]
                for meth,key,label in bars:
                    if meth=='baseT': v=(N['base_T'].get(f'{model}|{s}') or {}).get(key,{}).get(what)
                    else:
                        meth={'feedback':PK(model),'sup_ts':SUP(model)}.get(meth,meth)
                        v=sc(model,meth,key,s,what);v=float(np.mean(v)) if v else None
                    vals.append(v if v is not None else np.nan)
                rows.append(vals)
            data.append(rows)
        vmax=np.nanmax(np.array(data));norm=Normalize(0,vmax)
        for row,((model,name),values) in enumerate(zip(MODELS,data)):
            ax=axes[row,col]
            matrix(ax,values,[SNAME[s] for s in SUITES] if col==0 else ['']*6,[b[2] for b in bars],norm,LOWER_CMAP,lambda v:f'{v:.3f}')
            ax.set_title(f'{name} — '+('calibration error (ECE)' if what=='ece' else 'selective risk (AURC)'),fontsize=9.6)
    fig.subplots_adjust(left=.19,right=.99,bottom=.10,top=.90,hspace=.65,wspace=.13)
    fig.text(.59,.985,'Confidence quality: lower numbers are better',ha='center',va='top',fontsize=11,weight='bold')
    fig.text(.59,.025,'Pk = Peekaboo; T = temperature. Gray / — = unavailable. Darker = larger error within each metric.',ha='center',fontsize=8)
    save_figure(fig,'calibration')


def fig_reliability():
    fig,axes=plt.subplots(2,3,figsize=(7.3,5.5),sharex=True,sharey=True)
    for row,s in enumerate(('decisionbench','typed')):
        for col,(model,name) in enumerate(MODELS):
            ax=axes[row,col]
            ax.plot([0,1],[0,1],'--',color='#888888',lw=.9)
            for meth,key,label,color,marker in [('released','records','Released model',PALETTE['base'],'s'),
                (PK(model),'records_raw','Feedback before T',PALETTE['raw'],'^'),
                (PK(model),'records','Feedback after T',PALETTE['feedback'],'o')]:
                runs=A.runs(model,s,meth,key)
                pairs=[(max(r['probs'].values()),float(r.get('pred')==r['gold'])) for recs in runs for r in recs if r.get('probs')]
                if not pairs:continue
                conf,corr=np.array(pairs).T
                bins=np.clip(np.digitize(conf,np.linspace(0,1,11))-1,0,9)
                pts=[(conf[bins==i].mean(),corr[bins==i].mean()) for i in range(10) if (bins==i).sum()>=10]
                ax.plot([p[0] for p in pts],[p[1] for p in pts],marker=marker,ms=4,lw=1.2,color=color,label=label)
            ax.set(xlim=(0,1.02),ylim=(0,1.02));ax.set_aspect('equal',adjustable='box')
            ax.set_title(f'{name}\n{SNAME[s]}',fontsize=9.5)
            ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            ax.grid(color='#EEEEEE',lw=.6)
            if row==1:ax.set_xlabel('Predicted confidence',fontsize=8.5)
            if col==0:ax.set_ylabel('Observed accuracy',fontsize=8.5)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.54,.045),fontsize=8.2)
    fig.subplots_adjust(left=.10,right=.99,bottom=.18,top=.89,hspace=.48,wspace=.15)
    fig.text(.55,.99,'Calibration check: points on the diagonal match stated confidence',ha='center',va='top',weight='bold',fontsize=10)
    fig.text(.55,.015,'Below the dashed diagonal = overconfident. Above = underconfident.',ha='center',fontsize=8.5)
    save_figure(fig,'reliability')


def fig_switch_T():
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.9),gridspec_kw={'width_ratios':[3,6]})
    temperatures=[]; switches=[]
    for s in SUITES:
        ts=[];sw=[]
        for model,name in MODELS:
            values=[f['T'] for info in A.infos(model,s,PK(model)) for f in info.get('folds',[]) if f.get('T') is not None]
            ts.append(np.mean(values) if values else np.nan)
            info=N['switch'].get(f'{model}|{PK(model)}|{s}') or {}
            for key in ('on','sampled_on'):
                value=info.get(key);sw.append(value if value is not None else np.nan)
        temperatures.append(ts);switches.append(sw)
    matrix(axes[0],np.log2(temperatures),[SNAME[s] for s in SUITES],['Vega\n0.8B','Vega\n4B','Laya'],
           TwoSlopeNorm(vmin=-2,vcenter=0,vmax=5),'BrBG',lambda v:f'{2**v:.1f}')
    matrix(axes[1],switches,['']*6,['Best','Sampled']*3,Normalize(0,1),'Blues',lambda v:f'{100*v:.0f}%')
    axes[0].set_title('Fitted temperature T',fontsize=10,pad=29)
    axes[1].set_title('How often the switch retains the adapter',fontsize=10,pad=29)
    for x,name in zip((.165,.5,.835),('Vega 0.8B','Vega 4B','Laya')):
        axes[1].text(x,1.025,name,transform=axes[1].transAxes,ha='center',fontsize=8.5)
    for x in (1.5,3.5):axes[1].axvline(x,color='white',lw=6)
    fig.subplots_adjust(left=.18,right=.99,bottom=.28,top=.76,wspace=.13)
    fig.text(.38,.11,'T > 1 softens confidence.\nT < 1 sharpens confidence.',ha='center',fontsize=8.5)
    fig.text(.77,.11,'Best / Sampled = answer type compared.\n0% = always discard; 100% = always keep.',ha='center',fontsize=8.5)
    fig.text(.59,.025,'Laya: encoder trained. Gray / — = not evaluated. Switch fractions pool seeds and cross-validation folds.',ha='center',fontsize=8)
    save_figure(fig,'switch_T')


def fig_laya():
    """Use named training configurations and accuracy to make Laya's improvement explicit."""
    arms=[('released','records','Released Laya',PALETTE['base']),
          ('feedback_1pass','records','Head only\n1 feedback pass','#6FAB89'),
          ('feedback_enc_1pass','records','Encoder + head\n1 feedback pass','#6FAB89'),
          ('feedback_enc','records','Encoder + head\n8 feedback passes',PALETTE['feedback']),
          ('sup_enc','records_es','Encoder + head\nLabelled, early-stopped',PALETTE['labels']),
          ('laya_td','records','Released typed-decisions\ncheckpoint',PALETTE['vendor'])]
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.5),sharex=True,sharey=True)
    for col,s in enumerate(('typed','emotion')):
        ax=axes[col];clean_axis(ax,[a[2] for a in arms],zero=False)
        for y,(meth,key,label,color) in enumerate(arms):
            # The released fine-tune is in-domain only on typed-decisions.
            if meth=='laya_td' and s!='typed':
                ax.text(4,y,'transfer checkpoint;\nnot a matched comparator',va='center',fontsize=8,color='#666666');continue
            values=sc('laya',meth,key,s,'accuracy')
            if not values:
                ax.text(4,y,'not run',va='center',fontsize=8,color='#666666');continue
            value=100*np.mean(values)
            ax.barh(y,value,height=.52,color=color)
            ax.text(value+1.4,y,f'{value:.1f}%',va='center',fontsize=9)
        ax.set_xlim(0,103);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Test accuracy (%)')
        ax.set_title(SNAME[s],fontsize=11)
        if col:ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=.30,right=.98,bottom=.19,top=.80,wspace=.15)
    fig.text(.63,.98,'What changes when Laya is adapted?',ha='center',va='top',weight='bold',fontsize=11)
    fig.text(.63,.035,'Encoder-training results here use one seed per task. Bars show point estimates, not uncertainty.',ha='center',fontsize=8)
    save_figure(fig,'laya')


def fig_tasks_db():
    from collections import Counter
    d=N['tasks_db'];key='0.8b|feedback'
    if key not in d:return
    tasks=sorted(d[key],key=lambda t:d[key][t]['acc']-d[key][t]['base'])
    leaf_counts=Counter(t.split('/')[-1] for t in tasks)
    allg=[100*(d[f'{m}|feedback'][t]['acc']-d[f'{m}|feedback'][t]['base']) for m,_ in MODELS[:2] for t in tasks]
    limits=(np.floor(min(allg)/10)*10-3,np.ceil(max(allg)/10)*10+3)
    for chunk,(start,end) in enumerate(((0,22),(22,len(tasks)))):
        selected=tasks[start:end]
        names=[]
        for t in selected:
            name=t.split('/')[-1].replace('-',' ').replace('_',' ')
            if leaf_counts[t.split('/')[-1]] > 1:
                parts=t.split('/')
                qualifier=parts[0] if parts[-1]=='route' else parts[-2]
                name=qualifier.replace('_',' ')+' / '+name
            if any(a in t for a in ('contains_threat','toxicity_severity','finqa')):name+=' *'
            names.append(textwrap.fill(name,30))
        fig,ax=plt.subplots(figsize=(7.2,6.6));clean_axis(ax,names)
        for y,t in enumerate(selected):
            for dy,(m,name),color,marker in zip((-.13,.13),MODELS[:2],(PALETTE['feedback'],PALETTE['labels']),('o','s')):
                val=100*(d[f'{m}|feedback'][t]['acc']-d[f'{m}|feedback'][t]['base'])
                ax.plot([0,val],[y+dy,y+dy],color=color,lw=1.4,alpha=.65)
                ax.plot(val,y+dy,marker,color=color,ms=4)
        ax.set_xlim(*limits);ax.set_xlabel('Accuracy gain over released Vega (percentage points)')
        ax.set_title(f'DecisionBench tasks {start+1}–{end} of {len(tasks)}\nSorted by Vega 0.8B gain',fontsize=11)
        ax.tick_params(axis='y',labelsize=8.2)
        handles=[Line2D([],[],marker=mk,color=c,ls='',label=label) for (m,label),c,mk in zip(MODELS[:2],(PALETTE['feedback'],PALETTE['labels']),('o','s'))]
        ax.legend(handles=handles,frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.095))
        fig.subplots_adjust(left=.37,right=.98,bottom=.10,top=.85)
        save_figure(fig,'tasks_db' if chunk==0 else 'tasks_db_more')


def fig_benchmarks():
    import collections
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),sharey=True)
    y=np.arange(len(SUITES));counts={s:collections.Counter(it['question'].get('type') for it in A.items(s)) for s in SUITES}
    labels=[f'{SNAME[s]}\n(n = {len(A.items(s)):,})' for s in SUITES]
    specifications=[([('choice','Pick one','#5A82AA'),('boolean','Yes / no','#7FB295'),('score','Rating','#D1AD78')],'Question format'),
                    ([(2,'2 options','#5A82AA'),((3,4),'3–4 options','#7FB295'),((5,1000000),'5+ options','#D1AD78')],'Number of options')]
    for col,(parts,title) in enumerate(specifications):
        ax=axes[col];left=np.zeros(len(SUITES))
        for value,label,color in parts:
            vals=[]
            for s in SUITES:
                its=A.items(s)
                if col==0:v=counts[s].get(value,0)/len(its)
                else:
                    lo,hi=value if isinstance(value,tuple) else (value,value)
                    v=np.mean([lo<=len(it['labels'])<=hi for it in its])
                vals.append(100*v)
            vals=np.array(vals);ax.barh(y,vals,left=left,height=.65,color=color,label=label,edgecolor='white')
            for yi,v,l in zip(y,vals,left):
                if v>=13:ax.text(l+v/2,yi,f'{v:.0f}%',ha='center',va='center',fontsize=8.5)
            left+=vals
        ax.set(yticks=y,xlim=(0,100),ylim=(5.6,-.6))
        if col==0: ax.set_yticklabels(labels)
        if col:ax.tick_params(labelleft=False)
        ax.set_xticks([0,50,100]);ax.set_xlabel('Share of test questions (%)')
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.set_title(title,fontsize=10)
        ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.12),fontsize=7.8,handlelength=1,columnspacing=.7)
    fig.subplots_adjust(left=.20,right=.99,bottom=.16,top=.76,wspace=.13)
    save_figure(fig,'benchmarks')


# ───────────────────────────── tables ─────────────────────────────
def table(lines, name):
    open(os.path.join(HERE, name), 'w').write('\n'.join(lines) + '\n')


def num(x):
    return f'{x:,}'.replace(',', '{,}')


def tab_bench():
    pools = {s: len(A.items(s, 'pool')) for s in SUITES}
    rows = [('43 public tasks', 'Vega', num(len(A.items('public'))), num(pools['public']), "43 public datasets as typed questions, up to 60 test rows each; feedback: VegaML's 50 fit rows per task, drawn class by class; macro-averaged over tasks"),
            ('JevBench', 'Vega', num(len(A.items('jevbench'))), '(5-fold)', 'rules, policies and judgment calls in 18 families, paraphrase groups; no training split, every method cross-fits'),
            ('DecisionBench', 'Vega', num(len(A.items('decisionbench'))), num(pools['decisionbench']), "43 tasks in 28 domains; the README's 25-row slice per task is the test set, other rows of the same tasks the pool"),
            ('typed-decisions', 'Laya', num(len(A.items('typed'))), num(pools['typed']), "400 test and 1,200 training cases, five decisions each; the pool is the public training split, also the training data of Vega's shipped adapter"),
            ('emotion', '(ours)', num(len(A.items('emotion'))), num(pools['emotion']), 'six emotions of a tweet; pool: training rows drawn at random (natural class mix)'),
            ('AG News', '(ours)', num(len(A.items('agnews'))), num(pools['agnews']), 'four news sections; pool: training rows drawn at random')]
    L = ['\\begin{table}[t]\\centering\\small', '\\caption{The six benchmarks. ``Feedback rows\'\' are the rows whose verdicts a method may learn from (80\\%) and calibrate on (20\\%); no test row is ever used for learning.}',
         '\\label{tab:bench}', '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{llrrp{7.4cm}}', '\\toprule', 'benchmark & reported by & test rows & feedback rows & rows and pool \\\\', '\\midrule']
    for r in rows:
        L.append(' & '.join(r) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    table(L, 'tables_bench.tex')


def tab_main():
    rows_v = [('released', 'records', 'released model', 'none'), ('ttt', 'records', "TTT head, 0.8.0 rule", 'answer key'), ('ttt', 'records_guard', "TTT head, 0.12.3 rule", 'answer key'),
              ('sup_ts', 'records', 'answer-key adapter, same recipe', 'answer key'), ('sup_ts', 'records_es', 'answer-key adapter, early-stopped', 'answer key'), ('sup_1pass', 'records', 'answer-key adapter, one pass', 'answer key'),
              ('feedback', 'records', '\\textbf{Peekaboo}, eight passes', 'verdicts'), ('feedback_1pass', 'records', '\\textbf{Peekaboo}, one pass', 'verdicts'),
              ('logged_argmax', 'records', 'logged best answers, rated once', 'verdicts'), ('logged_ips', 'records', 'logged sampled answers, rated once', 'verdicts'), ('feedback_kl0', 'records', 'Peekaboo, no KL term', 'verdicts')]
    rows_l = [('released', 'records', 'released model', 'none'), ('laya_td', 'records', "laya-typed-decisions (fine-tune on typed-decisions)", 'answer key'),
              ('sup_ts', 'records', 'answer-key head, same recipe', 'answer key'), ('sup_ts', 'records_es', 'answer-key head, early-stopped', 'answer key'), ('sup_1pass', 'records', 'answer-key head, one pass', 'answer key'),
              ('feedback', 'records', '\\textbf{Peekaboo}, head only, eight passes', 'verdicts'), ('feedback_1pass', 'records', '\\textbf{Peekaboo}, head only, one pass', 'verdicts'),
              ('sup_enc', 'records', 'answer-key fine-tune, encoder unfrozen', 'answer key'), ('sup_enc', 'records_es', 'answer-key fine-tune, encoder unfrozen, early-stopped', 'answer key'),
              ('feedback_enc', 'records', '\\textbf{Peekaboo}, encoder unfrozen, eight passes', 'verdicts'), ('feedback_enc_1pass', 'records', '\\textbf{Peekaboo}, encoder unfrozen, one pass', 'verdicts'),
              ]
    for what, caption, label in (('accuracy', 'Accuracy, mean over seeds ($\\pm$ s.d.; three seeds except where Section~\\ref{sec:limitations} lists fewer)', 'tab:acc'),
                                 ('gain', 'Gain over the released model in accuracy points; $\\dagger$ marks differences that survive Holm adjustment within the method\'s family. The 95\\% intervals are in Figure~\\ref{fig:gains} and Table~\\ref{tab:h2h}', 'tab:gain')):
        L = ['\\begin{table}[tbp]\\centering\\scriptsize', f'\\caption{{{caption}. JevBench has no training split: every method learns by 5-fold cross-fitting there.}}', f'\\label{{{label}}}',
             '\\setlength{\\tabcolsep}{2.5pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llr' + 'r' * len(SUITES) + '}', '\\toprule',
             'model & method & learns from & ' + ' & '.join(SSHORT[s] for s in SUITES) + ' \\\\', '\\midrule']
        for model, mname in MODELS:
            first = True
            for meth, key, name, src in (rows_l if model == 'laya' else rows_v):
                cells = []
                for s in SUITES:
                    if what == 'accuracy':
                        v = sc(model, meth, key, s, 'accuracy'); cells.append(f3(v) if v else '$\\cdot$')
                    else:
                        g = cmp(model, meth, key, s); cells.append('--' if meth == 'released' else (fg(g, ci=False) if g else '$\\cdot$'))
                if all(c == '$\\cdot$' for c in cells):
                    continue
                L.append(f'{mname if first else ""} & {name} & {src} & ' + ' & '.join(cells) + ' \\\\'); first = False
            if model != 'laya':
                L.append('\\addlinespace')
        L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
        table(L, f'tables_{"acc" if what == "accuracy" else "gain"}.tex')


def tab_h2h():
    L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{Head to head, in accuracy points with 95\\% intervals (seeds and row clusters resampled). Positive means the first method is better. '
         '$\\dagger$: survives Holm adjustment within the row\'s family; NI: exploratory non-inferiority at a 2-point margin using an unadjusted interval (lower bound above $-2$); these marks are not multiplicity-adjusted. TTT is scored under the vegaml 0.12.3 rule; for Laya ``their method\'\' is the typed-decisions fine-tune, in-domain only on typed-decisions. The Laya head-only one-pass arms barely move from the released model (Section~\\ref{sec:laya}), so no NI marks are given for them.}',
         '\\label{tab:h2h}', '\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{ll' + 'r' * len(SUITES) + '}', '\\toprule',
         'model & comparison & ' + ' & '.join(SSHORT[s] for s in SUITES) + ' \\\\']
    rows = [('feedback', 'records', 'ttt', 'records_guard', 'Peekaboo (8 passes) vs. their method'), ('feedback', 'records', 'sup_ts', 'records', 'Peekaboo (8) vs. answer key, same recipe (8)'),
            ('feedback', 'records', 'sup_ts', 'records_es', 'Peekaboo (8) vs. answer key, early-stopped'), ('feedback_1pass', 'records', 'sup_1pass', 'records', 'Peekaboo (1 pass) vs. answer key (1 pass)'),
            ('feedback_1pass', 'records', 'feedback', 'records', 'Peekaboo, 1 pass vs. 8 passes'), ('logged_ips', 'records', 'feedback_1pass', 'records', 'logged sampled answers vs. Peekaboo (1 pass)'),
            ('logged_ips', 'records', 'feedback', 'records', 'logged sampled answers vs. Peekaboo (8)'), ('logged_ips', 'records', 'sup_ts', 'records_es', 'logged sampled answers vs. answer key, early-stopped'),
            ('logged_argmax', 'records', 'sup_ts', 'records_es', 'logged best answers vs. answer key, early-stopped'),
            ('feedback', 'records', 'feedback_kl0', 'records', 'KL term vs. none'), ('feedback', 'records_switch', 'feedback', 'records', 'with the switch vs. without')]
    rows_l = [('feedback', 'records', 'laya_td', 'records', 'Peekaboo, head (8) vs. their fine-tune'), ('feedback', 'records', 'sup_ts', 'records_es', 'Peekaboo, head (8) vs. answer-key head, early-stopped'),
              ('feedback_1pass', 'records', 'sup_1pass', 'records', 'Peekaboo, head (1) vs. answer-key head (1)'),
              ('feedback_enc', 'records', 'laya_td', 'records', 'Peekaboo, encoder (8) vs. their fine-tune'), ('feedback_enc', 'records', 'sup_enc', 'records', 'Peekaboo, encoder (8) vs. answer-key encoder, same recipe'),
              ('feedback_enc', 'records', 'sup_enc', 'records_es', 'Peekaboo, encoder (8) vs. answer-key encoder, early-stopped'), ('feedback_enc_1pass', 'records', 'feedback_enc', 'records', 'Peekaboo, encoder: 1 pass vs. 8 passes'),
              ('feedback_enc', 'records', 'feedback', 'records', 'encoder unfrozen vs. head only (8 passes)')]
    for model, mname in MODELS:
        L += ['\\midrule', f'\\multicolumn{{{2 + len(SUITES)}}}{{l}}{{\\textit{{{mname}}}}} \\\\']
        for a, ka, b, kb, name in (rows_l if model == 'laya' else rows):
            cells = []
            for s in SUITES:
                g = N['compare'].get(f'{model}|{a}|{ka}|{s}|vs|{b}|{kb}')
                cells.append((fg(g) + (' NI' if g.get('non_inferior_2') and 'answer' in name and 'head' not in name else '')) if g else '--')
            if any(c != '--' for c in cells):
                L.append(f' & {name} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_h2h.tex')


def tab_realistic():
    arms = [('feedback', 'eight passes, clean verdicts'), ('feedback_1pass', 'one pass, clean verdicts'), ('logged_argmax', "logged: the released model's best answers, rated once"),
            ('logged_ips', "logged: the released model's sampled answers, rated once (clipped IPS)"), ('feedback_noise20', '20\\% of verdicts flipped (eight passes)'),
            ('feedback_half', 'half of the answers rated (eight passes)'), ('feedback_complaints', 'every wrong answer reported, 20\\% of right ones (eight passes)'), ('sup_1pass', 'answer key, one pass'), ('sup_ts', 'answer key, eight passes')]
    suites = ['decisionbench', 'typed', 'emotion']
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{Feedback as a deployment delivers it: gain over the released model in points (95\\% intervals), on the three benchmarks with the realistic-feedback arms.}', '\\label{tab:realistic}',
         '\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{l' + 'r' * (3 * len(suites)) + '}', '\\toprule',
         'feedback & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{SNAME[s]}}}' for s in suites) + ' \\\\', ' & ' + ' & '.join('Vega 0.8B & Vega 4B & Laya (head)' for _ in suites) + ' \\\\', '\\midrule']
    for meth, name in arms:
        cells = [fg(cmp(m, meth, 'records', s), ci=True, mark=False) for s in suites for m, _ in MODELS]
        if any(c != '--' for c in cells):
            L.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_realistic.tex')


def tab_calibration():
    L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{Calibration, proper scores and selective risk (lower is better; mean over seeds). ``+ T\'\' rows apply one temperature to the released model, fitted on its own calibration rows from verdicts on sampled answers (mean over 10 draws) or from labels; JevBench has no calibration pool for them.}',
         '\\label{tab:cal}', '\\setlength{\\tabcolsep}{2.5pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{ll' + 'rrrr' * len(SUITES) + '}', '\\toprule',
         'model & method & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{SSHORT[s]}}}' for s in SUITES) + ' \\\\', ' & & ' + ' & '.join('ECE & NLL & Brier & AURC' for _ in SUITES) + ' \\\\']
    rows = [('released', 'records', 'released model'), ('baseT', 'feedback', 'released + T from verdicts'), ('baseT', 'labels', 'released + T from labels'), ('feedback', 'records_raw', 'Peekaboo before the honesty fix'),
            ('feedback', 'records', 'Peekaboo'), ('feedback_1pass', 'records', 'Peekaboo, one pass'), ('sup_ts', 'records', 'answer key + label temperature'), ('feedback_enc', 'records', 'Peekaboo, encoder unfrozen'), ('sup_enc', 'records', 'answer-key encoder + label temperature')]
    for model, mname in MODELS:
        L += ['\\midrule', f'\\multicolumn{{{2 + 4 * len(SUITES)}}}{{l}}{{\\textit{{{mname}}}}} \\\\']
        for meth, key, name in rows:
            cells = []
            for s in SUITES:
                if meth == 'baseT':
                    v = (N['base_T'].get(f'{model}|{s}') or {}).get(key)
                    cells += [f'{v[w]:.3f}' if v and v.get(w) is not None else '--' for w in ('ece', 'nll', 'brier', 'aurc')]
                else:
                    cells += [f'{np.mean(sc(model, meth, key, s, w)):.3f}' if sc(model, meth, key, s, w) else '--' for w in ('ece', 'nll', 'brier', 'aurc')]
            if any(c != '--' for c in cells):
                L.append(f' & {name} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_cal.tex')


def tab_anatomy():
    L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{What the gain is made of. Left: gain over the released model of a per-task label prior with no input features (log-offsets on the released probabilities), trained from verdicts with Peekaboo\'s loop and from labels; and Peekaboo minus that prior. '
         'Right: how often eight on-policy passes reveal the gold answer: reward-1 events per training row (each one names the gold answer), the share of rows the base policy would reveal in 8 draws, the share of two-option rows, and the first pass\'s reward against the released model\'s best-answer accuracy on the same rows.}',
         '\\label{tab:anatomy}', '\\setlength{\\tabcolsep}{2.5pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrr|rrrrr}', '\\toprule',
         'model & benchmark & prior (verdicts) & prior (labels) & Peekaboo $-$ prior & params & reveals / row & $P$(revealed, 8 draws) & 2-option rows & pass-1 reward & base best-answer acc. \\\\']
    for model, mname in MODELS:
        L += ['\\midrule']
        first = True
        for s in SUITES:
            p = N['prior'].get(f'{model}|{s}'); r = N['reveal'].get(f'{model}|feedback|{s}')
            if not p and not r:
                continue
            c = [fg(p['feedback']['vs_released'], ci=False, mark=False) if p else '--', fg(p['labels']['vs_released'], ci=False, mark=False) if p else '--',
                 fg(p['feedback']['peekaboo_vs'], ci=True, mark=False) if p and p['feedback'].get('peekaboo_vs') else '--', str(p['feedback']['params']) if p else '--',
                 f"{r['reveals_per_row']:.1f}" if r and r.get('reveals_per_row') else '--', f"{r['p_reveal_8_base_policy']:.2f}" if r and r.get('p_reveal_8_base_policy') is not None else '--',
                 f"{100 * r['binary_share']:.0f}\\%" if r and r.get('binary_share') is not None else '--', f"{r['pass1_reward']:.2f}" if r and r.get('pass1_reward') else '--',
                 f"{r['base_argmax_acc_train']:.2f}" if r and r.get('base_argmax_acc_train') is not None else '--']
            L.append(f'{mname if first else ""} & {SNAME[s]} & ' + ' & '.join(c) + ' \\\\'); first = False
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_anatomy.tex')
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{Which rows change. Share of the released model\'s wrong rows that a method fixes, by the released model\'s rank of the gold option, and share of its right rows the method breaks (mean over benchmarks with a training split and over seeds).}',
         '\\label{tab:fixbreak}', '\\begin{tabular}{llrrrrr}', '\\toprule', 'model & method & fixed, all & gold 2nd & gold 3rd & gold 4th+ & broken \\\\', '\\midrule']
    for model, mname in MODELS:
        for meth, name in (('feedback', 'Peekaboo (8 passes)'), ('feedback_1pass', 'Peekaboo (1 pass)'), ('sup_ts', 'answer key (8 passes)'), ('feedback_enc', 'Peekaboo, encoder (8)'), ('sup_enc', 'answer-key encoder (8)')):
            vals = [N['fixbreak'].get(f'{model}|{meth}|{s}') for s in SUITES if s != 'jevbench']
            vals = [v for v in vals if v]
            if not vals:
                continue
            m = lambda k: f"{100 * np.nanmean([v[k] if v.get(k) is not None else np.nan for v in vals]):.0f}\\%"
            L.append(f'{mname} & {name} & {m("fix_rate")} & {m("fix_rate_rank2")} & {m("fix_rate_rank3")} & {m("fix_rate_rank4plus")} & {m("break_rate")} \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    table(L, 'tables_fixbreak.tex')
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{VegaML\'s TTT head answers a row only where a head was kept for its task and the option set matches (``covered\'\'); elsewhere its output is the released engine\'s. Eight-pass Peekaboo minus TTT, and the answer-key adapter minus TTT, on covered and uncovered rows, under the vegaml 0.12.3 rule used in the main comparison; rows covered under the 0.8.0 rule for reference.}',
         '\\label{tab:ttt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrrr}', '\\toprule', 'model & benchmark & covered (0.12.3) & covered (0.8.0) & Peekaboo $-$ TTT, covered & uncovered & answer key $-$ TTT, covered \\\\', '\\midrule']
    for model, mname in MODELS[:2]:
        for s in SUITES:
            t = N['ttt'].get(f'{model}|{s}'); t0 = N['ttt'].get(f'{model}|{s}|v080') or {}
            if t:
                L.append(f'{mname} & {SNAME[s]} & {t["rows_covered"]} / {t["rows"]} & {t0.get("rows_covered", "--")} & {fg(t.get("peekaboo_vs_ttt_covered"), mark=False)} & {fg(t.get("peekaboo_vs_ttt_uncovered"), mark=False)} & {fg(t.get("adapter_vs_ttt_covered"), mark=False)} \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_ttt.tex')
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{Benchmark artifacts. Gain over the released model with every task, and without the five tasks with answer-position, option-structure or class-prior shortcuts (public: truthfulqa\\_mc1, pubmedqa; DecisionBench: contains\\_threat, toxicity\\_severity, finqa-numerical-reasoning).}',
         '\\label{tab:artifacts}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrr}', '\\toprule', 'model & method & public, all 43 & public, 41 tasks & DecisionBench, all 43 & DecisionBench, 40 tasks \\\\', '\\midrule']
    for model, mname in MODELS:
        for meth, key, name in (('feedback', 'records', 'Peekaboo (8)'), ('feedback_1pass', 'records', 'Peekaboo (1)'), ('sup_ts', 'records', 'answer key'), ('ttt', 'records_guard', 'TTT (0.12.3)'), ('feedback_enc', 'records', 'Peekaboo, encoder'), ('laya_td', 'records', 'their fine-tune')):
            c = [fg(cmp(model, meth, key, 'public'), mark=False), fg(cmp(model, meth, key, 'public', suffix='|noartifact'), mark=False),
                 fg(cmp(model, meth, key, 'decisionbench'), mark=False), fg(cmp(model, meth, key, 'decisionbench', suffix='|noartifact'), mark=False)]
            if any(x != '--' for x in c):
                L.append(f'{mname} & {name} & ' + ' & '.join(c) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_artifacts.tex')


def tab_noship():
    L = ['\\begin{table}[tbp]\\centering\\small', '\\caption{typed-decisions on Vega with the shipped \\texttt{typed\\_decisions} adapter, which was fitted with labels on this feedback pool, switched off. Accuracy (mean over seeds) and gain over the corresponding released model.}',
         '\\label{tab:noship}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrr}', '\\toprule', 'model & method & accuracy, adapters on & gain & accuracy, adapters off & gain \\\\', '\\midrule']
    for model, mname in MODELS[:2]:
        for on, off, name in (('released', 'noship_released', 'released engine'), ('sup_ts', 'sup_ts_noship', 'answer-key adapter (8)'), ('feedback', 'feedback_noship', 'Peekaboo (8)'), ('feedback_1pass', 'feedback_1pass_noship', 'Peekaboo (1)')):
            a_on = sc(model, on, 'records', 'typed', 'accuracy'); a_off = sc(model, off, 'records', 'typed', 'accuracy')
            g_on = cmp(model, on, 'records', 'typed') if on != 'released' else None
            g_off = N['compare'].get(f'{model}|{off}|records|typed|vs|noship_released|records') if off != 'noship_released' else None
            if a_on or a_off:
                L.append(f'{mname} & {name} & {f3(a_on)} & {fg(g_on, mark=False) if g_on else "--"} & {f3(a_off)} & {fg(g_off, mark=False) if g_off else "--"} \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_noship.tex')


def tab_tasks():
    cols = [('0.8b', 'released', 'records', 'base'), ('0.8b', 'ttt', 'records_guard', 'TTT'), ('0.8b', 'sup_ts', 'records_es', 'key'), ('0.8b', 'feedback', 'records', 'Pk8'), ('0.8b', 'feedback_1pass', 'records', 'Pk1'),
            ('4b', 'released', 'records', 'base'), ('4b', 'ttt', 'records_guard', 'TTT'), ('4b', 'sup_ts', 'records_es', 'key'), ('4b', 'feedback', 'records', 'Pk8'), ('4b', 'feedback_1pass', 'records', 'Pk1'),
            ('laya', 'released', 'records', 'base'), ('laya', 'laya_td', 'records', 'f-t'), ('laya', 'feedback', 'records', 'Pk8'), ('laya', 'sup_enc', 'records_es', 'key-enc'), ('laya', 'feedback_enc', 'records', 'Pk-enc')]
    tg = A.task_groups('public')
    acc = {}
    for m, meth, key, lab in cols:
        R = A.runs(m, 'public', meth, key)
        if R:
            c = np.mean([A.correct(r) for r in R], 0); acc[(m, meth, key)] = {t: float(c[g].mean()) for t, g in tg.items()}
    cols = [c for c in cols if (c[0], c[1], c[2]) in acc]
    L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{Accuracy on each of the 43 public tasks (mean over seeds). key: the early-stopped answer-key adapter / head; Pk8, Pk1: Peekaboo with eight passes, one pass; f-t: laya-typed-decisions; -enc: encoder unfrozen. $^*$ marks the two tasks with a construction artifact. Bold: the best method for that model on that task.}',
         '\\label{tab:tasks}', '\\setlength{\\tabcolsep}{2pt}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}{l' + 'r' * len(cols) + '}', '\\toprule',
         'task & ' + ' & '.join(f'\\multicolumn{{{sum(1 for c in cols if c[0] == m)}}}{{c}}{{{mn}}}' for m, mn in MODELS if any(c[0] == m for c in cols)) + ' \\\\', ' & ' + ' & '.join(c[3] for c in cols) + ' \\\\', '\\midrule']
    for t in sorted(tg):
        row = []
        for m, _ in MODELS:
            vals = [acc[(c[0], c[1], c[2])].get(t) for c in cols if c[0] == m]
            best = max(v for v in vals if v is not None) if any(v is not None for v in vals) else None
            row += [('' if v is None else (f'\\textbf{{{v:.2f}}}' if abs(v - best) < 5e-4 else f'{v:.2f}')) for v in vals]
        L.append(t.replace('_', '\\_') + ('$^*$' if t in A.ARTIFACT['public'] else '') + ' & ' + ' & '.join(row) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    table(L, 'tables_tasks.tex')
    d = N['tasks_db']
    if '0.8b|feedback' in d:
        L = ['\\begin{table}[tbp]\\centering\\scriptsize', '\\caption{DecisionBench per task (25 test rows each): accuracy of the released model and gain in points of Peekaboo (8 passes), the answer-key adapter and TTT (0.8.0 rule). $^*$: tasks with a construction artifact.}',
             '\\label{tab:tasks_db}', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{lrrrrrrrr}', '\\toprule', 'task & \\multicolumn{4}{c}{Vega 0.8B} & \\multicolumn{4}{c}{Vega 4B} \\\\', ' & base & Pk & key & TTT & base & Pk & key & TTT \\\\', '\\midrule']
        for t in sorted(d['0.8b|feedback']):
            c = []
            for m in ('0.8b', '4b'):
                b = d.get(f'{m}|feedback', {}).get(t, {}).get('base')
                c.append(f'{b:.2f}' if b is not None else '')
                for meth in ('feedback', 'sup_ts', 'ttt'):
                    v = d.get(f'{m}|{meth}', {}).get(t)
                    c.append(f'{100 * (v["acc"] - v["base"]):+.0f}' if v else '')
            star = '$^*$' if any(a in t for a in ('contains_threat', 'toxicity_severity', 'finqa')) else ''
            L.append(t.split('/')[-1].replace('_', '\\_') + star + ' & ' + ' & '.join(c) + ' \\\\')
        L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
        table(L, 'tables_tasks_db.tex')


def tex_numbers():
    tr = {'0.8b': 'Small', '4b': 'Big', 'laya': 'Laya', 'released': 'Base', 'ttt': 'Ttt', 'sup_ts': 'Sup', 'sup_1pass': 'SupOne', 'sup_enc': 'SupEnc', 'feedback': 'Pk', 'feedback_1pass': 'PkOne',
          'feedback_enc': 'PkEnc', 'feedback_enc_1pass': 'PkEncOne', 'pooled': 'Pooled', 'feedback_kl0': 'Klzero', 'laya_td': 'Td', 'logged_argmax': 'LogArg', 'logged_ips': 'LogIps',
          'feedback_noise20': 'Noise', 'feedback_half': 'Half', 'feedback_complaints': 'Compl', 'noship_released': 'NoshipBase', 'feedback_noship': 'PkNoship', 'sup_ts_noship': 'SupNoship',
          'feedback_1pass_noship': 'PkOneNoship', 'public': 'Pub', 'jevbench': 'Jev', 'decisionbench': 'Db', 'typed': 'Typed', 'emotion': 'Emo', 'agnews': 'Ag',
          'records': '', 'records_switch': 'Sw', 'records_raw': 'Raw', 'records_es': 'Es', 'records_guard': 'Guard', 'noartifact': 'Clean'}
    nm = lambda *p: ''.join(tr.get(x, x.title()) for x in p)
    L = ['% generated by paper/make_figures.py; every number in the text comes from here']
    seen = set()

    def add(name, val):
        if name not in seen:
            seen.add(name); L.append(f'\\newcommand{{\\{name}}}{{{val}}}')
    for k, v in N['compare'].items():
        if not v:
            continue
        parts = k.split('|')
        model, meth, key, s = parts[:4]
        if parts[5] == 'released':
            base = nm(model, meth, key, s) + ('Clean' if parts[-1] == 'noartifact' else '')
        else:
            base = nm(model, meth, key, s) + 'Vs' + nm(parts[5], parts[6])
        add('gain' + base, f'{v["gain"]:+.1f}'); add('ci' + base, f'[{v["lo"]:+.1f}, {v["hi"]:+.1f}]'); add('p' + base, f'{v["p"]:.3f}')
        add('abs' + base, f'{abs(v["gain"]):.1f}')
    for k, v in N['scores'].items():
        model, meth, key, s = k.split('|')
        for w in ('accuracy', 'ece', 'nll', 'aurc', 'brier'):
            if v.get(w) and v[w][0] is not None:
                add(('acc' if w == 'accuracy' else w) + nm(model, meth, key, s), f'{np.mean(v[w]):.3f}')
    for k, v in N['base_T'].items():
        model, s = k.split('|')
        for how in ('feedback', 'labels'):
            add('eceBaseT' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'{v[how]["ece"]:.3f}'); add('TBaseT' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'{v[how]["T"]:.1f}')
    for k, v in N['prior'].items():
        model, s = k.split('|')
        for how in ('feedback', 'labels'):
            add('gainPrior' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'{v[how]["vs_released"]["gain"]:+.1f}')
            if v[how].get('peekaboo_vs'):
                add('gainPkVsPrior' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'{v[how]["peekaboo_vs"]["gain"]:+.1f}')
                add('ciPkVsPrior' + ('Fb' if how == 'feedback' else 'Lab') + nm(model, s), f'[{v[how]["peekaboo_vs"]["lo"]:+.1f}, {v[how]["peekaboo_vs"]["hi"]:+.1f}]')
    for k, v in N['reveal'].items():
        model, meth, s = k.split('|')
        for w, name in (('reveals_per_row', 'reveals'), ('p_reveal_8_base_policy', 'preveal'), ('pass1_reward', 'passone'), ('base_argmax_acc_train', 'baseargmax'), ('binary_share', 'binary'), ('max_pass_reward', 'maxpass')):
            if v.get(w) is not None:
                add(name + nm(model, meth, s), f'{v[w]:.2f}')
    for k, v in N['ttt'].items():
        parts = k.split('|'); model, s = parts[:2]; old = 'Vzero' if len(parts) == 3 else ''
        add('tttCovered' + old + nm(model, s), str(v['rows_covered']))
        if old:
            continue
        for w in ('peekaboo_vs_ttt_covered', 'peekaboo_vs_ttt_uncovered', 'adapter_vs_ttt_covered'):
            if v.get(w):
                add('gain' + ''.join(x.title() for x in w.split('_')) + nm(model, s), f'{v[w]["gain"]:+.1f}'); add('ci' + ''.join(x.title() for x in w.split('_')) + nm(model, s), f'[{v[w]["lo"]:+.1f}, {v[w]["hi"]:+.1f}]')
    for k, v in N['switch'].items():
        model, meth, s = k.split('|')
        add('swOn' + nm(model, meth, s), f'{100 * v["on"]:.0f}')
        if v.get('sampled_on') is not None:
            add('swOnSampled' + nm(model, meth, s), f'{100 * v["sampled_on"]:.0f}')
    for k, v in N['fixbreak'].items():
        model, meth, s = k.split('|')
        for w, name in (('fix_rate', 'Fix'), ('break_rate', 'Break'), ('fix_rate_rank2', 'FixRankTwo'), ('fix_rate_rank3', 'FixRankThree'), ('fix_rate_rank4plus', 'FixRankFour')):
            if v.get(w) is not None:
                add(name + nm(model, meth, s), f'{100 * v[w]:.0f}')
    for model in ('0.8b', '4b', 'laya'):
        for meth in ('feedback', 'feedback_1pass', 'sup_ts', 'feedback_enc', 'sup_enc'):
            vals = [N['fixbreak'].get(f'{model}|{meth}|{s}') for s in SUITES if s != 'jevbench']
            vals = [x for x in vals if x]
            if vals:
                for w, name in (('fix_rate', 'Fix'), ('break_rate', 'Break'), ('fix_rate_rank2', 'FixRankTwo'), ('fix_rate_rank4plus', 'FixRankFour')):
                    add(name + 'Mean' + nm(model, meth), f'{100 * np.nanmean([x[w] if x.get(w) is not None else np.nan for x in vals]):.0f}')
    if N.get('t_match'):
        add('TmatchSame', str(N['t_match']['same'])); add('TmatchNear', str(N['t_match']['within_one_step'])); add('TmatchRuns', str(N['t_match']['runs']))
    if N.get('T_range'):
        add('Tmin', f'{N["T_range"]["min"]:.1f}'); add('Tmax', f'{N["T_range"]["max"]:.1f}'); add('Tmedian', f'{N["T_range"]["median"]:.1f}')
    for k, v in N['balanced'].items():
        model, meth, key, s = k.split('|')
        add('bal' + nm(model, meth, key, s), f'{np.mean(v):.3f}')
    # a few numbers from the first version's grid (output/analysis.json), prefixed vone: Laya head-only eight-pass arms
    v1 = os.path.join(PR, 'output', 'analysis.json')
    if os.path.exists(v1):
        N1 = json.load(open(v1))
        for k, v in N1['compare'].items():
            parts = k.split('|')
            if v and parts[0] == 'laya' and parts[1] in ('feedback', 'sup_ts', 'feedback_kl0', 'pooled') and parts[2] == 'records' and parts[5] == 'released' and len(parts) == 6:
                add('gainVone' + nm(*parts[:4]), f'{v["gain"]:+.1f}'); add('ciVone' + nm(*parts[:4]), f'[{v["lo"]:+.1f}, {v["hi"]:+.1f}]')
            if v and parts[0] == 'laya' and parts[1] == 'feedback' and parts[2] == 'records' and len(parts) == 7 and parts[5] in ('sup_ts', 'laya_td') and parts[6] == 'records':
                add('gainVone' + nm(*parts[:4]) + 'Vs' + nm(parts[5]), f'{v["gain"]:+.1f}'); add('ciVone' + nm(*parts[:4]) + 'Vs' + nm(parts[5]), f'[{v["lo"]:+.1f}, {v["hi"]:+.1f}]')
        for k, v in N1['prior'].items():
            model, s = k.split('|')
            if model == 'laya':
                add('gainVonePriorFb' + nm(model, s), f'{v["feedback"]["vs_released"]["gain"]:+.1f}')
        for k, v in N1['scores'].items():
            model, meth, key, s = k.split('|')
            if model == 'laya' and meth in ('feedback', 'sup_ts') and key == 'records' and v.get('accuracy'):
                add('accVone' + nm(model, meth, key, s), f'{np.mean(v["accuracy"]):.3f}')
    open(os.path.join(HERE, 'numbers.tex'), 'w').write('\n'.join(L) + '\n')
    return len(L)


def tab_seeds():
    """Report observed run coverage, and reject grouping arms with different seed sets."""
    groups = []
    for model, label in MODELS[:2]:
        groups += [
            (model, label, 'feedback and labelled, 8 / 1 passes', ('feedback', 'feedback_1pass', 'sup_ts', 'sup_1pass')),
            (model, label, 'logged and damaged feedback', ('logged_ips', 'logged_argmax', 'feedback_noise20', 'feedback_half', 'feedback_complaints')),
            (model, label, 'no KL', ('feedback_kl0',)),
            (model, label, 'shipped adapters off', ('feedback_noship', 'feedback_1pass_noship', 'sup_ts_noship')),
        ]
    groups += [
        ('laya', 'Laya', 'encoder, feedback 8 / 1 and labelled 8', ('feedback_enc', 'feedback_enc_1pass', 'sup_enc')),
        ('laya', 'Laya', 'head, feedback and labelled 1 pass', ('feedback_1pass', 'sup_1pass')),
    ]
    L = [r'\begin{table}[H]\centering\small',
         r'\caption{Independent training runs available in the current grid, counted from per-seed result files. Grouped methods have identical seed sets. A dash means not run. Early stopping reuses the labelled run; it is not another seed. Released models, TTT and historical head-only eight-pass runs are not included.}',
         r'\label{tab:seeds}', r'\setlength{\tabcolsep}{4pt}',
         r'\resizebox{\linewidth}{!}{\begin{tabular}{llrrrrrr}', r'\toprule',
         r'model & arms & public & JevBench & DecisionB. & typed-dec. & emotion & AG News \\', r'\midrule']
    for model, label, desc, methods in groups:
        cells = []
        for suite in SUITES:
            seeds = [{os.path.basename(f).rsplit('_s', 1)[1] for f in A.run_files(model, suite, method)} for method in methods]
            if any(s != seeds[0] for s in seeds[1:]):
                raise ValueError(f'Cannot group different seed sets: {model}, {suite}, {methods}')
            cells.append(str(len(seeds[0])) if seeds[0] else '--')
        L.append(label + ' & ' + desc + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    table(L, 'tables_seeds.tex')


if __name__ == '__main__':
    fig_training(); fig_benchmarks(); fig_gains(); fig_laya(); fig_passes(); fig_realistic(); fig_anatomy(); fig_calibration(); fig_reliability(); fig_switch_T(); fig_tasks_db()
    tab_bench(); tab_main(); tab_h2h(); tab_realistic(); tab_calibration(); tab_anatomy(); tab_noship(); tab_tasks(); tab_seeds()
    n = tex_numbers()
    print('wrote figures, tables and', n, 'macros from', OUT)
