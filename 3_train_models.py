# -*- coding: utf-8 -*-
"""
3_train_models.py   (formerly train_models.py; pipeline step 3)
================
ANN baseline + PINN + ablation + the reviewer-question experiments in one script.

Run:
    python 3_train_models.py              # main ablation (ANN vs PINN, 8 configs, bootstrap CI) + final model
    python 3_train_models.py --robust     # [Q1] misspecified-prior test (answer to the circularity concern)
    python 3_train_models.py --realval    # [Q2] leave-one-group-out validation on the real Liu specimens
    python 3_train_models.py --all        # all three
    python 3_train_models.py --quick      # fast smoke test (fewer seeds/epochs)
    python 3_train_models.py --final-only # final model only (Results/pinn_bundle.pt)
    python 3_train_models.py --figs-only  # redraw Figures 4-6 from the CSVs without retraining

Optional overrides (defaults unchanged when omitted):
    --fractions 0.3,0.8                   # run only a subset of the training fractions
    --seeds N                             # fewer/more seeds for the ablation and the robustness run

Reviewer questions -> where they are answered:
  Q1 circularity            -> --robust  (Results/prior_robustness.csv, Figure 5)
  Q2 validation on real data -> --realval (Results/real_validation.csv, Figure 6)
  Q3 corrosion factor estimate -> 2_build_dataset.py --sensitivity / --factor
  Q4 FEA adds no new information -> FEA VERIFICATION in 2_build_dataset.py + 1_fem_pipeline.py --mesh-study
  Q5 R^2 limit              -> oracle R^2, irreducible error, R^2(vs law) and R^2/oracle in every table of this file

File outputs: Results/ablation_results.csv, Results/prior_robustness.csv,
             Results/real_validation.csv, Results/pinn_bundle.pt,
             Figures/fig4_ablation.*, Figures/fig5_prior_robustness.*, Figures/fig6_real_validation.*
             + Figures/fig_prior_robustness.* and Figures/fig_real_validation.* under the
               summary-figure file names (fig_sn_fits / fig_logo_pred_obs come from 4_reanalysis.py,
               fig_fem_verification from 1_fem_pipeline.py)
Dependencies: torch, numpy, matplotlib (figures only)
"""
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# 2_build_dataset.py starts with a digit (pipeline order) -- hence imported from the path
import importlib.util as _ilu
_BD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2_build_dataset.py')
_bd_spec = _ilu.spec_from_file_location('build_dataset', _BD_PATH)
bd = _ilu.module_from_spec(_bd_spec)
sys.modules['build_dataset'] = bd
_bd_spec.loader.exec_module(bd)

# On Windows, redirecting the output (python 3_train_models.py > log.txt) can raise
# UnicodeEncodeError for non-ASCII log text -- this is only that guard; no effect in a normal console.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# =====================================================================
# CONFIG
# =====================================================================
FEATURES = ['corrosion_depth_mm', 'applied_stress_mpa']
TARGET = 'log10_fatigue_life'
CLEAN = 'log10N_clean'

N_SEEDS = 8
TRAIN_FRACTIONS = (0.3, 0.8)
TEST_FRAC = 0.2
EPOCHS = 2000
LR = 0.01
HIDDEN = 32
N_COLLOC = 200
VIOL_TOL = 1e-3
N_BOOT = 5000
PRACTICAL_TOL = 0.003            # log10; differences smaller than this (~0.7% life) count as practically 'the same' even when the CI excludes zero

WEIGHT_DECAY = 1e-3
PATIENCE = 300
VAL_FRAC = 0.2

# (name, lambda_monotonic, lambda_point_physics, tuned?)  tuned = weight decay + validation early stopping
# -- the physics-based configs also have tuned versions so the comparison is fair (the tuned ANN is a strong baseline).
CONFIGS = [
    ('ANN (plain)', 0.0, 0.0, False),
    ('ANN (tuned)', 0.0, 0.0, True),
    ('Monotonicity (plain)', 0.1, 0.0, False),
    ('Monotonicity (tuned)', 0.1, 0.0, True),
    ('Law-prior (plain)', 0.0, 1.0, False),
    ('Law-prior (tuned)', 0.0, 1.0, True),
    ('Full PINN (plain)', 0.1, 1.0, False),
    ('Full PINN (tuned)', 0.1, 1.0, True),
]
BASELINE = 'ANN (tuned)'
FINAL_CONFIG = 'Full PINN (tuned)'
FINAL_SEED = 0

ROBUST_SEEDS = 5
ROBUST_LAMBDAS = (1.0, 0.1)     # mitigation: what happens when the physics weight is lowered to limit the damage of a wrong prior
REALVAL_SEEDS = 3

# depending on the 2_build_dataset.py version, RESULTS_DIR may or may not exist -- hence getattr
RESULTS_DIR = getattr(bd, 'RESULTS_DIR', 'Results')
BUNDLE = os.path.join(RESULTS_DIR, 'pinn_bundle.pt')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'ablation_results.csv')
ROBUST_CSV = os.path.join(RESULTS_DIR, 'prior_robustness.csv')
REALVAL_CSV = os.path.join(RESULTS_DIR, 'real_validation.csv')


def _ensure_dirs():
    """Create the Results/ and Figures/ folders. This used to call bd._ensure_dirs(), but that
    function does not exist in 2_build_dataset.py -- the cause of the earlier AttributeError; hence it is done here."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('Figures', exist_ok=True)


# ---------------------------------------------------------------------
# Figure-style and save helpers (previously in a separate figstyle.py -- now inlined here;
# the separate style file was dropped to keep the script self-contained)
# ---------------------------------------------------------------------
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')

_RC_PIPELINE = {   # pipeline figures (fig4/5/6) -- exactly the earlier figstyle.py settings
    'font.size': 9.5,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'axes.titlepad': 6,
    'legend.fontsize': 7.5,
    'legend.title_fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.4,
    'lines.markersize': 5,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.35,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'savefig.dpi': 300,
    'figure.constrained_layout.use': True,
    'axes.unicode_minus': False,
}
_RC_SUMMARY = {      # summary-figure style (same settings as the earlier fig_*.pdf files)
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.4,
    'savefig.dpi': 300,
    'figure.constrained_layout.use': True,
    'axes.unicode_minus': False,
}
_RC_SUMMARY_SMALL = dict(_RC_SUMMARY, **{'font.size': 9, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
                                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8})


def _fig_style(summary=False, small=False):
    """Apply the figure style and return plt (idempotent). summary=True gives the summary-figure
    style (same family as the pipeline figures); small=True gives the compact (two-panel) version."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if summary:
        matplotlib.rcParams.update(_RC_SUMMARY_SMALL if small else _RC_SUMMARY)
    else:
        matplotlib.rcParams.update(_RC_PIPELINE)
    if not os.path.isdir(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
    return plt


def _fig_save(fig, *names):
    """Save Figures/<name>.png (300 dpi) + Figures/<name>.pdf for each name."""
    out = []
    for name in names:
        for ext in ('.png', '.pdf'):
            p = os.path.join(FIGURES_DIR, name + ext)
            fig.savefig(p)
            out.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def cfg_of(name):
    return [c for c in CONFIGS if c[0] == name][0]


# =====================================================================
# Data, network
# =====================================================================
def load_dataset(path=None):
    if path is None:
        path = getattr(bd, 'OUT_CSV', 'training_dataset.csv')
        if not os.path.exists(path):      # if not found at the bd path, search the old/new version locations too
            for alt in ('training_dataset.csv', os.path.join(RESULTS_DIR, 'training_dataset.csv')):
                if os.path.exists(alt):
                    path = alt
                    break
    if not os.path.exists(path):
        raise SystemExit("'%s' missing -- run `python 2_build_dataset.py` first." % path)
    with open(path, 'r') as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    y = np.array([float(r[TARGET]) for r in rows])
    yc = np.array([float(r[CLEAN]) for r in rows])
    return X, y, yc


class SimpleNet(nn.Module):
    def __init__(self, input_dim=len(FEATURES), hidden=HIDDEN):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


# =====================================================================
# Training: a prior is any object providing .log10N(rho_A, S_eq)
# =====================================================================
def train_one(Xtr, ytr, lam_mono, lam_point, seed, prior, lo, hi, epochs=EPOCHS, tuned=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-12
    y_mean, y_std = float(ytr.mean()), float(ytr.std() + 1e-12)

    Xs = torch.tensor((Xtr - x_mean) / x_std, dtype=torch.float32)
    ys = torch.tensor(((ytr - y_mean) / y_std).reshape(-1, 1), dtype=torch.float32)
    net = SimpleNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY if tuned else 0.0)
    mse = nn.MSELoss()
    use_phys = (lam_mono > 0) or (lam_point > 0)
    floor_log = math.log10(getattr(bd, 'HCF_MIN_CYCLES', 100))

    if tuned:
        idx = rng.permutation(len(Xs))
        nv = max(3, int(round(VAL_FRAC * len(Xs))))
        va_i, tr_i = idx[:nv], idx[nv:]
        Xva, yva, Xs, ys = Xs[va_i], ys[va_i], Xs[tr_i], ys[tr_i]
        best, best_state, wait = float('inf'), None, 0

    for _ in range(epochs):
        opt.zero_grad()
        loss = mse(net(Xs), ys)
        if use_phys:
            c_raw = rng.uniform(lo, hi, size=(N_COLLOC, len(FEATURES)))
            z = torch.tensor((c_raw - x_mean) / x_std, dtype=torch.float32, requires_grad=True)
            pred_c = net(z)
            if lam_mono > 0:
                g = torch.autograd.grad(pred_c.sum(), z, create_graph=True)[0]
                loss = loss + lam_mono * (torch.relu(g) ** 2).mean(dim=0).sum()
            if lam_point > 0:
                tgt = prior.log10N(bd.rho_A_from_depth(c_raw[:, 0]), bd.equivalent_amplitude_R01(c_raw[:, 1]))
                tgt = np.maximum(tgt, floor_log)
                tgt_s = torch.tensor(((tgt - y_mean) / y_std).reshape(-1, 1), dtype=torch.float32)
                loss = loss + lam_point * mse(pred_c, tgt_s)
        loss.backward()
        opt.step()
        if tuned:
            with torch.no_grad():
                v = float(mse(net(Xva), yva))
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
    if tuned and best_state is not None:
        net.load_state_dict(best_state)
    return net, (x_mean, x_std, y_mean, y_std)


def predict(net, scal, X):
    x_mean, x_std, y_mean, y_std = scal
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor((np.asarray(X) - x_mean) / x_std, dtype=torch.float32)).numpy().reshape(-1)
    return p * y_std + y_mean


def violation_rate(net, scal, lo, hi, n=25):
    d = np.linspace(lo[0], hi[0], n)
    s = np.linspace(lo[1], hi[1], n)
    D, S = np.meshgrid(d, s, indexing='ij')
    P = predict(net, scal, np.column_stack([D.ravel(), S.ravel()])).reshape(n, n)
    bad = (np.diff(P, axis=0) > VIOL_TOL).sum() + (np.diff(P, axis=1) > VIOL_TOL).sum()
    return float(bad) / (np.diff(P, axis=0).size + np.diff(P, axis=1).size)


def metrics(net, scal, Xte, yte, cte, lo, hi):
    p = predict(net, scal, Xte)
    sse = float(((yte - p) ** 2).sum())
    sst = float(((yte - yte.mean()) ** 2).sum())
    sse_c = float(((cte - p) ** 2).sum())
    sst_c = float(((cte - cte.mean()) ** 2).sum())
    return dict(r2=1.0 - sse / sst, r2_clean=1.0 - sse_c / sst_c,
                rmse=math.sqrt(sse / len(yte)), rmse_clean=math.sqrt(sse_c / len(yte)),
                viol=violation_rate(net, scal, lo, hi))


def split_indices(n, seed, frac):
    n_test = int(round(TEST_FRAC * n))
    perm = np.random.default_rng(seed).permutation(n)
    te, rest = perm[:n_test], perm[n_test:]
    return rest[:min(int(round(frac * n)), len(rest))], te


def verdict(dm, dl, dh):
    """Tag combining the CI and practical relevance (negative Delta = better).
    'n.s.' = the two-sided 95% bootstrap CI includes zero (alpha = 0.05)."""
    if dl <= 0.0 <= dh:
        return "n.s."
    if abs(dm) < PRACTICAL_TOL:
        return "same (tiny)"
    return "BETTER" if dh < 0 else "WORSE"


def paired_ci(a, b, n_boot=N_BOOT, seed=0):
    """d = a - b (per seed); bootstrap 95% CI. Returns (mean, lo, hi)"""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion (two-sided, alpha = 0.05).
    Returns (lo, hi) as percentages; n = 0 gives NaN."""
    if n <= 0:
        return float('nan'), float('nan')
    p = float(k) / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (100.0 * (c - h) / d, 100.0 * (c + h) / d)


# =====================================================================
# Pipeline figures: Figure 4 (ablation), 5 (prior robustness), 6 (real validation)
# rows = list of dicts following the CSV schema (strings or numbers both accepted); --figs-only uses the same path
# =====================================================================
def _sig_marker(ci_lo, ci_hi, tol=PRACTICAL_TOL):
    """Marker combining the CI and practical relevance: '+' better, '-' worse, '~' not significant/tiny"""
    lo, hi = float(ci_lo), float(ci_hi)
    if lo <= 0.0 <= hi:
        return '~'
    if abs((lo + hi) / 2.0) < tol:
        return '~'
    return '+' if hi < 0 else '-'


def draw_fig4(rows, floor=None, oracle_r2=None):
    if not rows:
        return
    try:
        plt = _fig_style()
    except Exception as e:
        print("(matplotlib missing -- Figure 4 skipped: %s)" % e)
        return
    fracs = sorted(set(float(r['train_fraction']) for r in rows), reverse=True)
    configs = [c[0] for c in CONFIGS]
    fig, axs = plt.subplots(1, len(fracs), figsize=(4.2 * len(fracs), 3.9), squeeze=False,
                            sharey=True)
    tops = []
    for ax, frac in zip(axs[0], fracs):
        xs, ys, es, marks = [], [], [], []
        for i, cname in enumerate(configs):
            sub = [r for r in rows if float(r['train_fraction']) == frac and r['config'] == cname]
            if not sub:
                continue
            r0 = sub[0]
            xs.append(i)
            ys.append(float(r0['rmse_vs_law_mean']))
            es.append(float(r0['rmse_vs_law_std']))
            marks.append(_sig_marker(r0.get('ci_lo', 0.0), r0.get('ci_hi', 0.0))
                         if float(r0.get('delta_rmse_vs_baseline', 0.0) or 0.0) != 0.0 else ' ')
        colors = ['#888888' if 'ANN' in c else ('#3b6db4' if 'Full PINN' in c else '#6aaed6')
                  for c in configs[:len(xs)]]
        ax.bar(xs, ys, yerr=es, width=0.62, color=colors, edgecolor='k', linewidth=0.5,
               error_kw=dict(lw=1.0, capsize=2.5), zorder=2)
        for x, y, m in zip(xs, ys, marks):
            if m != ' ':
                ax.text(x, y * 1.06, m, ha='center', va='bottom', fontsize=10,
                        color='#b2182b' if m == '+' else '#2166ac')
        ax.axhline(floor, color='k', ls=':', lw=1.0)
        base = [r for r in rows if float(r['train_fraction']) == frac and r['config'] == BASELINE]
        if base:
            ax.axhline(float(base[0]['rmse_vs_law_mean']), color='#888888', ls='--', lw=1.0)
        tops.append(max((y + e) for y, e in zip(ys, es)) if ys else 0.1)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=40, ha='right', fontsize=8.5)
        ax.set_title('Training fraction = %d%%' % round(100 * frac), loc='left')
        ax.grid(True, axis='y')
    top_all = max(tops) if tops else 0.1
    if floor is not None:
        top_all = max(top_all, float(floor))
    for ax in axs[0]:
        ax.set_ylim(0.0, 1.30 * top_all)
    axs[0][0].set_ylabel('RMSE vs true law (log$_{10}$ N)')
    if floor is not None:
        from matplotlib.transforms import blended_transform_factory
        _tr = blended_transform_factory(axs[0][0].transAxes, axs[0][0].transData)
        axs[0][0].text(0.99, float(floor), 'irreducible error (label noise)', transform=_tr,
                       va='bottom', ha='right', fontsize=7.5)
    axs[0][0].text(0.97, 0.97, 'Error bars: s.d. across seeds\n+ / \u2212 : better / worse\n~ : not significant\n(95% CI vs ANN (tuned))',
                   transform=axs[0][0].transAxes, ha='right', va='top', fontsize=7.5,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.9))
    try:
        paths = _fig_save(fig, 'fig4_ablation')
        print("Figure saved: %s" % ", ".join(os.path.basename(p) for p in paths))
    except Exception as e:
        print("(fig4_ablation could not be saved -- training results untouched: %s)" % e)


def draw_fig5(rows):
    if not rows:
        return
    try:
        plt = _fig_style()
    except Exception as e:
        print("(matplotlib missing -- Figure 5 skipped: %s)" % e)
        return
    fracs = sorted(set(float(r['train_fraction']) for r in rows), reverse=True)
    priors = []
    for r in rows:
        p = r['prior'].split(' | lambda=')[0]
        if p not in priors:
            priors.append(p)
    lam_vals = sorted(set(float(r['prior'].split(' | lambda=')[1]) for r in rows
                          if ' | lambda=' in r['prior']))
    fig, axs = plt.subplots(1, len(fracs), figsize=(6.0 * len(fracs), 4.8), squeeze=False, sharey=True)
    mk = {lam_vals[0] if lam_vals else 1.0: ('o', '#3b6db4'),
          (lam_vals[1] if len(lam_vals) > 1 else 0.1): ('s', '#d16127')}
    for ax, frac in zip(axs[0], fracs):
        for i, p in enumerate(priors):
            for lam in lam_vals:
                sub = [r for r in rows if float(r['train_fraction']) == frac
                       and r['prior'].startswith(p + ' | lambda=%.1f' % lam)]
                if not sub:
                    continue
                r0 = sub[0]
                d, lo, hi = float(r0['delta']), float(r0['ci_lo']), float(r0['ci_hi'])
                m, c = mk.get(lam, ('^', '#4d4d4d'))
                ax.errorbar(i, d, yerr=[[max(0.0, d - lo)], [max(0.0, hi - d)]],
                            fmt=m, color=c, ms=6, lw=1.1, capsize=3, zorder=3)
        ax.axhline(0.0, color='k', lw=0.9)
        ax.set_xticks(range(len(priors)))
        ax.set_xticklabels(priors, rotation=40, ha='right', fontsize=9)
        ax.set_title('training fraction = %d%%' % round(100 * frac), loc='left')
        ax.grid(True, axis='y')
    axs[0][0].set_ylabel('$\\Delta$ RMSE vs true law (PINN $-$ tuned ANN)')
    hs = [plt.Line2D([], [], ls='none', marker=mk[l][0], color=mk[l][1], ms=6,
                     label='$\\lambda_{point}$ = %.1f' % l) for l in lam_vals]
    axs[0][-1].legend(handles=hs, loc='upper left', fontsize=9)
    fig.suptitle('95% bootstrap CI; below 0: PINN more accurate than the tuned ANN; above 0: the prior degrades accuracy', fontsize=10)
    try:
        paths = _fig_save(fig, 'fig5_prior_robustness')
        print("Figure saved: %s" % ", ".join(os.path.basename(p) for p in paths))
    except Exception as e:
        print("(fig5_prior_robustness could not be saved -- training results untouched: %s)" % e)


def draw_fig6(rows):
    if not rows:
        return
    try:
        plt = _fig_style()
    except Exception as e:
        print("(matplotlib missing -- Figure 6 skipped: %s)" % e)
        return
    ind = [r for r in rows if int(float(r['in_domain'])) == 1]
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    lo_v = min(float(r['log10N_real']) for r in rows) - 0.25
    hi_v = max(float(r['log10N_real']) for r in rows) + 0.25
    band = [lo_v, hi_v]
    ax.fill_between(band, [l - math.log10(2) for l in band], [l + math.log10(2) for l in band],
                    color='0.85', zorder=0, label='within $\\times$2')
    ax.plot(band, band, 'k-', lw=0.9, zorder=1)
    allr = [r for r in rows if not math.isnan(float(r['law_pred']))]
    ax.scatter([float(r['log10N_real']) for r in allr], [float(r['law_pred']) for r in allr],
               marker='s', s=16, facecolors='none', edgecolors='#7f7f7f', lw=0.8,
               label='S-N law (all specimens, n=%d)' % len(allr), zorder=2)
    series = [('ann_pred', 'o', '#3b6db4', 'ANN (tuned)'), ('pinn_pred', '^', '#d16127', 'PINN')]
    ylab = []
    for key, mkc, col, lab in series:
        pts = [(float(r['log10N_real']), float(r[key])) for r in ind
               if r.get(key) not in (None, '', 'nan') and not math.isnan(float(r[key]))]
        if not pts:
            continue
        xs, ys = zip(*pts)
        err = [yy - xx for xx, yy in pts]
        w2 = 100.0 * sum(1 for e in err if abs(e) <= math.log10(2)) / len(err)
        rmse = math.sqrt(sum(e * e for e in err) / len(err))
        ax.scatter(xs, ys, marker=mkc, s=20, color=col, alpha=0.85,
                   label='%s (in-domain n=%d):\nRMSE %.2f, %.0f%% within $\\times$2' % (lab, len(pts), rmse, w2),
                   zorder=3)
        ylab.append((lab, rmse, w2))
    ax.set_xlim(lo_v, hi_v)
    ax.set_ylim(lo_v, hi_v)
    ax.set_aspect('equal')
    ax.set_xlabel('Observed log$_{10}$N')
    ax.set_ylabel('LOGO predicted log$_{10}$N\n(unseen corrosion group)')
    ax.set_title('Real-specimen validation (Liu et al. 2021)', loc='left')
    ax.grid(True)
    hs6, ls6 = ax.get_legend_handles_labels()
    fig.legend(hs6, ls6, loc='outside lower center', ncol=2, fontsize=7.5, framealpha=0.95)
    try:
        paths = _fig_save(fig, 'fig6_real_validation')
        print("Figure saved: %s" % ", ".join(os.path.basename(p) for p in paths))
    except Exception as e:
        print("(fig6_real_validation could not be saved -- training results untouched: %s)" % e)


# =====================================================================
# Summary figures (fig_prior_robustness, fig_real_validation) -- same style and
# save scheme as the pipeline figures; implemented directly in this script.
# =====================================================================
# prior robustness: summary-figure order + colours (CSV prior name | display name | colour)
PRIOR_ORDER = [
    ('exact (= data law; main result)',   'Verified law',                 '#1b7837'),
    ('linear-in-rho refit (structural)',  'Verified refit (linear)',      '#57a05f'),
    ('half-sample refit',                 'Verified refit (half-sample)', '#9fcc9f'),
    ('bias +0.15 log10',                  'Biased prior +0.15',           '#e8973a'),
    ('bias -0.15 log10',                  'Biased prior \u22120.15',      '#e8973a'),
    ('bias +0.30 log10',                  'Biased prior +0.30',           '#c04a3a'),
    ('bias -0.30 log10',                  'Biased prior \u22120.30',      '#c04a3a'),
    ('published Eq.5 (unverified)',       'Published Eq. 5 (unverified)', '#7a1f1f'),
]


def draw_summary_prior(rows):
    """Summary figure fig_prior_robustness: 8 priors, lambda=1.0 bar + lambda=0.1 circle (log-x).
    rows = list of dicts following the return schema of run_robustness() (directly or from Results/prior_robustness.csv)."""
    if not rows:
        return
    try:
        plt = _fig_style(summary=True, small=True)
    except Exception as e:
        print("(matplotlib missing -- fig_prior_robustness skipped: %s)" % e)
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 4.2))
    for ax, frac, tag in zip(axs, (0.3, 0.8), ('(a) 30% training data', '(b) 80% training data')):
        sub = [r for r in rows if abs(float(r['train_fraction']) - frac) < 1e-9]
        if not sub:
            continue
        base = float(sub[0]['baseline_rmse'])
        ys, labels = [], []
        for i, (key, disp, col) in enumerate(PRIOR_ORDER):
            r1 = [r for r in sub if r['prior'] == '%s | lambda=1.0' % key]
            r0 = [r for r in sub if r['prior'] == '%s | lambda=0.1' % key]
            if not r1:
                continue
            v1 = float(r1[0]['rmse_vs_law_mean'])
            v0 = float(r0[0]['rmse_vs_law_mean']) if r0 else float('nan')
            ax.barh(i, v1, height=0.68, color=col, zorder=3)
            if v0 == v0:
                ax.scatter([v0], [i], marker='o', s=26, facecolors='none', edgecolors='k',
                           linewidths=1.2, zorder=5)
            vtxt = max(v1, v0) if v0 == v0 else v1
            ax.text(vtxt * 1.32, i, '%.3f' % v1, va='center', fontsize=9, zorder=6)
            ys.append(i)
            labels.append(disp)
        ax.axvline(base, ls='--', color='0.45', lw=1.0, zorder=4)
        ax.set_xscale('log')
        ax.set_xlim(8e-3, 1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(tag, fontweight='bold', pad=5)
        ax.set_xlabel('Held-out RMSE (log$_{10}N$)', fontsize=9)
    try:
        paths = _fig_save(fig, 'fig_prior_robustness')
        print("Figure saved: %s" % ", ".join(os.path.basename(p) for p in paths))
    except Exception as e:
        print("(fig_prior_robustness could not be saved -- results untouched: %s)" % e)


def draw_summary_realval(rows):
    """Summary figure fig_real_validation: real-specimen predicted-vs-observed, factor-of-two band,
    Verified law / Tuned PINN / Tuned ANN plus out-of-domain marks.
    rows = list of dicts following the return schema of run_real_validation()."""
    if not rows:
        return
    try:
        plt = _fig_style(summary=True)
    except Exception as e:
        print("(matplotlib missing -- fig_real_validation skipped: %s)" % e)
        return

    def _fv(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('nan')

    real = np.array([_fv(r['log10N_real']) for r in rows])
    law = np.array([_fv(r['law_pred']) for r in rows])
    ind = np.array([int(float(r['in_domain'])) == 1 for r in rows])
    pinn = np.array([_fv(r['pinn_pred']) for r in rows])
    ann = np.array([_fv(r['ann_pred']) for r in rows])
    ok = ~(np.isnan(real) | np.isnan(law))
    real, law, ind, pinn, ann = real[ok], law[ok], ind[ok], pinn[ok], ann[ok]
    if not len(real):
        return
    fig, ax = plt.subplots(figsize=(3.9, 3.9))
    lo = min(real.min(), law.min()) - 0.15
    hi = max(real.max(), law.max()) + 0.15
    xb = np.array([lo, hi])
    ax.fill_between(xb, xb - math.log10(2), xb + math.log10(2), color='#cfe0f2',
                    zorder=0, label='factor-of-two band')
    ax.plot(xb, xb, '--', color='0.3', lw=1.2, zorder=1)
    ax.scatter(law[~ind], real[~ind], marker='x', s=30, color='0.45', linewidths=1.3,
               zorder=3, label='out-of-domain (%d)' % int((~ind).sum()))
    ax.scatter(law[ind], real[ind], marker='o', s=30, facecolors='none', edgecolors='#1f77b4',
               linewidths=1.5, zorder=4, label='Verified law')
    ok_p = ind & (pinn == pinn)
    ok_a = ind & (ann == ann)
    ax.scatter(pinn[ok_p], real[ok_p], marker='s', s=20, color='#ff7f0e', zorder=5, label='Tuned PINN')
    ax.scatter(ann[ok_a], real[ok_a], marker='^', s=26, color='#2ca02c', zorder=4, label='Tuned ANN')
    w2 = int(np.sum(np.abs(law[ind] - real[ind]) <= math.log10(2)))
    fig.suptitle('in-domain: %d/%d within \u00d72 (%.1f%%)' % (
        w2, int(ind.sum()), 100.0 * w2 / max(1, int(ind.sum()))), fontsize=9.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('Predicted log$_{10}N$')
    ax.set_ylabel('Observed log$_{10}N$')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.95, borderaxespad=0.3, handletextpad=0.4)
    try:
        paths = _fig_save(fig, 'fig_real_validation')
        print("Figure saved: %s" % ", ".join(os.path.basename(p) for p in paths))
    except Exception as e:
        print("(fig_real_validation could not be saved -- results untouched: %s)" % e)


# =====================================================================
# Main ablation  [Q5: oracle/ceiling, R2(vs law), bootstrap CI]
# =====================================================================
def run_experiments(X, y, yc, law, lo, hi, n_seeds, fractions, epochs):
    n = len(y)
    res, t0 = {}, time.time()
    for frac in fractions:
        for seed in range(n_seeds):
            tr, te = split_indices(n, seed, frac)
            for name, lm, lp, tn in CONFIGS:
                net, scal = train_one(X[tr], y[tr], lm, lp, seed, law, lo, hi, epochs, tuned=tn)
                res.setdefault((frac, name), []).append(metrics(net, scal, X[te], y[te], yc[te], lo, hi))
            print("  frac=%.1f seed %d/%d done (%.0fs)" % (frac, seed + 1, n_seeds, time.time() - t0))
    return res


def summarize(res, fractions, floor, oracle_r2):
    rows = []
    print("\n" + "=" * 128)
    print("EVALUATION TARGETS (stated explicitly):")
    print("  - noise floor (irreducible RMSE) = %.3f: the labels carry Gaussian noise with sigma = the Table 4 fit" % floor)
    print("    residual, so no model that sees only the noisy labels can reach a lower RMSE on the noisy target.")
    print("  - ORACLE R^2 = %.3f: the R^2 that the noise-free law itself achieves against the noisy labels." % oracle_r2)
    print("  - RMSE(noisy): model vs the noisy labels -- the practical prediction target.")
    print("  - RMSE(vs law): model vs the noise-free trend (target recovery). The same fitted law generated the")
    print("    labels and serves as the physics prior; the prior-robustness experiment (--robust) quantifies what")
    print("    happens when the prior is wrong, which is the explicit check against circularity.")
    print("=> R^2(noisy) can never exceed the oracle value; hence also report 'R2/oracle' and 'R2 vs law'.")
    base_by_frac = {f: [m['rmse_clean'] for m in res[(f, BASELINE)]] for f in fractions}
    for frac in fractions:
        print("\n--- training fraction = %.0f%% of dataset ---" % (100 * frac))
        print("%-22s %-14s %-9s %-14s %-14s %-8s %s" % (
            "config", "R2(noisy)", "R2/oracle", "R2(vs law)", "RMSE(vs law)", "viol%",
            "Δ RMSE(vs law) vs %s  [95%% CI]  wins" % BASELINE))
        for name, lm, lp, tn in CONFIGS:
            ms = res[(frac, name)]
            g = lambda k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
            r2, r2c, rm, rc, vi = g('r2'), g('r2_clean'), g('rmse'), g('rmse_clean'), g('viol')
            cur = [m['rmse_clean'] for m in ms]
            if name == BASELINE:
                dtxt, dm, dl, dh, wins = "(baseline)", 0.0, 0.0, 0.0, ''
            else:
                dm, dl, dh = paired_ci(cur, base_by_frac[frac])
                wins = sum(1 for a, b in zip(cur, base_by_frac[frac]) if a < b)
                tag = verdict(dm, dl, dh)
                dtxt = "%+.3f [%+.3f, %+.3f] %s  %d/%d" % (dm, dl, dh, tag, wins, len(cur))
            print("%-22s %5.3f+/-%-6.3f %-9.3f %5.3f+/-%-6.3f %5.3f+/-%-6.3f %5.1f%%   %s"
                  % (name, r2[0], r2[1], r2[0] / oracle_r2, r2c[0], r2c[1], rc[0], rc[1], 100 * vi[0], dtxt))
            rows.append(dict(train_fraction=frac, config=name, lam_mono=lm, lam_point=lp, tuned=int(tn),
                             r2_noisy_mean=r2[0], r2_noisy_std=r2[1], r2_over_oracle=r2[0] / oracle_r2,
                             r2_vs_law_mean=r2c[0], r2_vs_law_std=r2c[1],
                             rmse_noisy_mean=rm[0], rmse_vs_law_mean=rc[0], rmse_vs_law_std=rc[1],
                             violation_rate_mean=vi[0], delta_rmse_vs_baseline=dm, ci_lo=dl, ci_hi=dh,
                             oracle_r2=oracle_r2, noise_floor_rmse=floor, n_seeds=len(ms)))
    print("=" * 128)
    print("(CI = two-sided 95% bootstrap of the paired per-seed differences, alpha = 0.05; negative Delta = better.")
    print(" n.s. = the CI includes zero; 'same (tiny)' = |Delta| < %.3f log10, i.e. practically equivalent)" % PRACTICAL_TOL)
    with open(RESULTS_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved: %s" % RESULTS_CSV)
    draw_fig4(rows, floor=floor, oracle_r2=oracle_r2)


# =====================================================================
# [Q1] misspecified-prior test -- what if the prior differs from the data-generating law?
# =====================================================================
class ShiftedPrior(object):
    def __init__(self, base, delta):
        self.base, self.delta = base, delta

    def log10N(self, rho_A, S_eq):
        return self.base.log10N(rho_A, S_eq) + self.delta


def make_priors(table_rows, exact_law, seed):
    rho = [r['rho'] for r in table_rows]; S = [r['S'] for r in table_rows]; N = [r['N'] for r in table_rows]
    priors = [('exact (= data law; main result)', exact_law)]
    rng = np.random.default_rng(10000 + seed)
    for _ in range(50):                       # refit on half the specimens (realistic estimation error)
        idx = rng.permutation(len(rho))[:max(len(rho) // 2, 10)]
        try:
            law_h, _, _ = bd.fit_liu_law([rho[i] for i in idx], [S[i] for i in idx], [N[i] for i in idx])
            priors.append(('half-sample refit', law_h))
            break
        except SystemExit:
            continue
    law_l, _, _ = bd.fit_liu_law(rho, S, N, quadratic=False)
    priors.append(('linear-in-rho refit (structural)', law_l))
    for d in (0.15, -0.15, 0.30, -0.30):
        priors.append(('bias %+.2f log10' % d, ShiftedPrior(exact_law, d)))
    pub = getattr(bd, 'PUBLISHED_COEFFS_UNVERIFIED', None)
    if pub:
        priors.append(('published Eq.5 (unverified)',
                       bd.LiuLaw.from_percent_coeffs(pub['a'], pub['b'])))
    return priors


def run_robustness(X, y, yc, law, table_rows, lo, hi, n_seeds, fractions, epochs):
    _, lm, lp, tn = cfg_of(FINAL_CONFIG)
    bl = cfg_of(BASELINE)
    n = len(y)
    store, base_store, t0 = {}, {}, time.time()
    for frac in fractions:
        for seed in range(n_seeds):
            tr, te = split_indices(n, seed, frac)
            net, scal = train_one(X[tr], y[tr], bl[1], bl[2], seed, law, lo, hi, epochs, tuned=bl[3])
            base_store.setdefault(frac, []).append(metrics(net, scal, X[te], y[te], yc[te], lo, hi))
            for pname, prior in make_priors(table_rows, law, seed):
                for lam in ROBUST_LAMBDAS:
                    net, scal = train_one(X[tr], y[tr], lm, lam, seed, prior, lo, hi, epochs, tuned=tn)
                    store.setdefault((frac, "%s | lambda=%.1f" % (pname, lam)), []).append(
                        metrics(net, scal, X[te], y[te], yc[te], lo, hi))
            print("  [robust] frac=%.1f seed %d/%d (%.0fs)" % (frac, seed + 1, n_seeds, time.time() - t0))
    rows = []
    print("\n" + "=" * 112)
    print("PRIOR ROBUSTNESS -- model = %s; baseline = %s. Values: RMSE against the true (data-generating) law." % (FINAL_CONFIG, BASELINE))
    for frac in fractions:
        b = [m['rmse_clean'] for m in base_store[frac]]
        print("\n--- training fraction = %.0f%% ---   baseline %s: RMSE(vs law) = %.3f" % (100 * frac, BASELINE, np.mean(b)))
        print("%-52s %-15s %s" % ("prior | physics-weight", "RMSE(vs law)", "Δ vs baseline [95% CI]   wins"))
        for pname in [k[1] for k in store if k[0] == frac]:
            cur = [m['rmse_clean'] for m in store[(frac, pname)]]
            dm, dl, dh = paired_ci(cur, b[:len(cur)])
            tag = verdict(dm, dl, dh)
            wins = sum(1 for a, c in zip(cur, b) if a < c)
            print("%-52s %5.3f+/-%-7.3f %+.3f [%+.3f, %+.3f] %-6s %d/%d" % (pname, np.mean(cur), np.std(cur), dm, dl, dh, tag, wins, len(cur)))
            rows.append(dict(train_fraction=frac, prior=pname, rmse_vs_law_mean=float(np.mean(cur)),
                             rmse_vs_law_std=float(np.std(cur)), baseline_rmse=float(np.mean(b)),
                             delta=dm, ci_lo=dl, ci_hi=dh, wins=wins, n_seeds=len(cur)))
    print("=" * 112)
    print("Reading rule: for a prior where 'BETTER' survives, the physics gain is realistic up to that kind of prior error.")
    print("'WORSE' means a sufficiently wrong prior makes the physics term harmful -- the break-even limit is reported above.")
    with open(ROBUST_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved: %s" % ROBUST_CSV)
    draw_fig5(rows)
    draw_summary_prior(rows)


# =====================================================================
# [Q2] leave-one-group-out validation on the real Liu specimens
# =====================================================================
def run_real_validation(table_rows, fem, n_seeds, epochs):
    """Leave out each corrosion group (E0,E5,E10,E15,E20) in turn: fit the law on the rest -> build
    the dataset -> train the ANN and the PINN -> predict N of the held-out group's real specimens.
    The held-out specimens are never seen in any step. Only specimens inside the model input
    domain (in-domain) can be predicted by the model.
    Reported per pooled model: RMSE, the within-x2 / within-x3 proportions with Wilson 95% CIs,
    and the per-seed spread (mean +/- s.d. of the pooled RMSE across seeds)."""
    _, lm, lp, tn = cfg_of(FINAL_CONFIG)
    bl = cfg_of(BASELINE)
    group = lambda r: r['specimen'].split('-')[0]
    groups = sorted(set(group(r) for r in table_rows), key=lambda g: float(g.lstrip('ED')))
    out, t0 = [], time.time()
    seed_errs = {'ann': [[] for _ in range(n_seeds)], 'pinn': [[] for _ in range(n_seeds)]}
    for g in groups:
        held = [r for r in table_rows if group(r) == g]
        rest = [r for r in table_rows if group(r) != g]
        if len(rest) < 8:
            continue
        try:
            law_g, _, _ = bd.fit_liu_law([r['rho'] for r in rest], [r['S'] for r in rest], [r['N'] for r in rest])
        except SystemExit:
            print("  fold %s: fit not possible, skipped" % g)
            continue
        rho = np.array([r['rho'] for r in held]); Sa = np.array([r['S'] for r in held])
        logN = np.log10([r['N'] for r in held])
        law_pred = law_g.log10N(rho, Sa)
        Xr = np.column_stack([bd.depth_from_rho_A(rho), bd.peak_stress_from_equivalent_amplitude(Sa)])

        ds0, _, _ = bd.generate_dataset(law_g, fem, seed=getattr(bd, 'NOISE_SEED', 123))
        Xd0 = np.array([[r['corrosion_depth_mm'], r['applied_stress_mpa']] for r in ds0])
        lo, hi = Xd0.min(0), Xd0.max(0)
        mask = np.all((Xr >= lo - 1e-9) & (Xr <= hi + 1e-9), axis=1)
        pinn_p = np.full(len(held), np.nan); ann_p = np.full(len(held), np.nan)
        if mask.any():
            pp, aa = [], []
            for seed in range(n_seeds):
                ds, _, _ = bd.generate_dataset(law_g, fem, seed=getattr(bd, 'NOISE_SEED', 123) + seed)
                Xd = np.array([[r['corrosion_depth_mm'], r['applied_stress_mpa']] for r in ds])
                yd = np.array([r['log10_fatigue_life'] for r in ds])
                net, sc = train_one(Xd, yd, lm, lp, seed, law_g, lo, hi, epochs, tuned=tn)
                pp.append(predict(net, sc, Xr[mask]))
                net, sc = train_one(Xd, yd, bl[1], bl[2], seed, law_g, lo, hi, epochs, tuned=bl[3])
                aa.append(predict(net, sc, Xr[mask]))
            pinn_p[mask] = np.mean(pp, axis=0); ann_p[mask] = np.mean(aa, axis=0)
            for s_i, pv in enumerate(pp):          # per-seed errors for the seed-spread estimate
                seed_errs['pinn'][s_i] += list(np.asarray(pv) - logN[mask])
            for s_i, av in enumerate(aa):
                seed_errs['ann'][s_i] += list(np.asarray(av) - logN[mask])
        for i, r in enumerate(held):
            out.append(dict(fold=g, specimen=r['specimen'], rho_A_pct=r['rho'], S_a_mpa=r['S'], log10N_real=float(logN[i]),
                            in_domain=int(mask[i]), law_pred=float(law_pred[i]), pinn_pred=float(pinn_p[i]), ann_pred=float(ann_p[i])))
        print("  [realval] fold %-4s: %d specimen, %d in-domain (%.0fs)" % (g, len(held), int(mask.sum()), time.time() - t0))

    def stats(pred, real):
        e = np.asarray(pred) - np.asarray(real)
        return math.sqrt(float((e ** 2).mean())), float((np.abs(e) <= math.log10(2)).mean()), float((np.abs(e) <= math.log10(3)).mean())

    print("\n" + "=" * 100)
    print("REAL-DATA VALIDATION (leave-one-corrosion-group-out; error = log10 N predicted - log10 N real)")
    print("%-6s %-5s %-5s | %-22s | %-22s | %s" % ("fold", "n", "n_in", "S-N law only (all n)", "ANN tuned (in-domain)", "PINN (in-domain)"))
    for g in sorted(set(o['fold'] for o in out), key=lambda s: float(s.lstrip('ED'))):
        rs = [o for o in out if o['fold'] == g]
        ind = [o for o in rs if o['in_domain']]
        la = stats([o['law_pred'] for o in rs], [o['log10N_real'] for o in rs])
        if ind:
            an = stats([o['ann_pred'] for o in ind], [o['log10N_real'] for o in ind])
            pi = stats([o['pinn_pred'] for o in ind], [o['log10N_real'] for o in ind])
            at = "RMSE %.2f (x%.2f)" % (an[0], 10 ** an[0]); pt = "RMSE %.2f (x%.2f)" % (pi[0], 10 ** pi[0])
        else:
            at = pt = "(no in-domain specimen)"
        print("%-6s %-5d %-5d | RMSE %.2f (x%.2f)%-6s | %-22s | %s" % (g, len(rs), len(ind), la[0], 10 ** la[0], "", at, pt))
    ind = [o for o in out if o['in_domain']]
    if ind:
        real = [o['log10N_real'] for o in ind]
        print("\nPOOLED in-domain results (n=%d; error = predicted - observed, log10 N):" % len(ind))
        for label, key in (("S-N law only", 'law_pred'), ("ANN (tuned)", 'ann_pred'), ("PINN", 'pinn_pred')):
            r, w2, w3 = stats([o[key] for o in ind], real)
            k2 = int(round(w2 * len(ind))); k3 = int(round(w3 * len(ind)))
            lo2, hi2 = wilson(k2, len(ind)); lo3, hi3 = wilson(k3, len(ind))
            print("  %-14s RMSE=%.3f log10 (x%.2f) | within x2: %.0f%% (%d/%d; Wilson 95%% CI %.1f..%.1f)"
                  " | within x3: %.0f%% (%d/%d; Wilson 95%% CI %.1f..%.1f)"
                  % (label, r, 10 ** r, 100 * w2, k2, len(ind), lo2, hi2, 100 * w3, k3, len(ind), lo3, hi3))
        if any(len(e) for e in seed_errs['ann']) or any(len(e) for e in seed_errs['pinn']):
            def _per_seed(key):
                vals = [math.sqrt(float((np.asarray(e) ** 2).mean())) for e in seed_errs[key] if len(e)]
                return (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))
            ma, sa = _per_seed('ann'); mp, sp = _per_seed('pinn')
            print("  seed spread (pooled in-domain RMSE, mean +/- s.d. over %d seeds):"
                  " ANN %.3f +/- %.3f | PINN %.3f +/- %.3f" % (n_seeds, ma, sa, mp, sp))
            print("  (the predictions above and in the CSV are the per-specimen means over these %d seeds)" % n_seeds)
    print("Limitation: the specimens' rho_A is pit-caused; the model maps depth = rho_A / factor (Q3). Only in-domain specimens are evaluated.")
    print("=" * 100)
    with open(REALVAL_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("Saved: %s" % REALVAL_CSV)
    draw_fig6(out)
    draw_summary_realval(out)


# =====================================================================
# Final model
# =====================================================================
def train_final(X, y, law, lo, hi, epochs, cv_info):
    _, lm, lp, tn = cfg_of(FINAL_CONFIG)
    net, (xm, xs, ym, ys) = train_one(X, y, lm, lp, FINAL_SEED, law, lo, hi, epochs, tuned=tn)
    torch.save(dict(
        state_dict=net.state_dict(), hidden=HIDDEN, features=FEATURES,
        x_mean=[float(v) for v in xm], x_std=[float(v) for v in xs], y_mean=ym, y_std=ys,
        lo=[float(v) for v in lo], hi=[float(v) for v in hi],
        final_config=FINAL_CONFIG, law_sigma=float(getattr(law, 'sigma', 0.0)),
        rho_A_factor=float(getattr(bd, 'RHO_A_FACTOR', 1.8736)),
        thickness_mm=float(getattr(bd, 'ORIGINAL_THICKNESS_MM', 10.0)),
        cv_rmse_vs_law=cv_info.get('rmse_vs_law'), cv_rmse_noisy=cv_info.get('rmse')), BUNDLE)
    print("Saved: %s (config = %s)" % (BUNDLE, FINAL_CONFIG))


def _read_csv_rows(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def figs_only():
    """Draw Figures 4-6 from the CSVs in Results/ without retraining,
    plus the summary figures fig_prior_robustness and fig_real_validation."""
    _ensure_dirs()
    if os.path.exists(RESULTS_CSV):
        rows4 = _read_csv_rows(RESULTS_CSV)
        fl = float(rows4[0].get('noise_floor_rmse')) if rows4[0].get('noise_floor_rmse') not in (None, '', 'nan') else None
        or2 = float(rows4[0].get('oracle_r2')) if rows4[0].get('oracle_r2') not in (None, '', 'nan') else None
        draw_fig4(rows4, floor=fl, oracle_r2=or2)
    else:
        print("( Results/ablation_results.csv missing -- run `python 3_train_models.py` first)")
    if os.path.exists(ROBUST_CSV):
        rows5 = _read_csv_rows(ROBUST_CSV)
        draw_fig5(rows5)
        draw_summary_prior(rows5)
    else:
        print("( Results/prior_robustness.csv missing -- run `python 3_train_models.py --robust` first)")
    if os.path.exists(REALVAL_CSV):
        rows6 = _read_csv_rows(REALVAL_CSV)
        draw_fig6(rows6)
        draw_summary_realval(rows6)
    else:
        print("( Results/real_validation.csv missing -- run `python 3_train_models.py --realval` first)")


def main():
    quick = '--quick' in sys.argv
    allm = '--all' in sys.argv
    do_robust = allm or '--robust' in sys.argv
    do_real = allm or '--realval' in sys.argv
    final_only = '--final-only' in sys.argv
    only_extra = (do_robust or do_real) and not allm and not final_only and '--main' not in sys.argv
    n_seeds = 2 if quick else N_SEEDS
    robust_seeds = 2 if quick else ROBUST_SEEDS
    fractions = (0.8,) if quick else TRAIN_FRACTIONS
    if '--fractions' in sys.argv and not quick:
        fractions = tuple(float(v) for v in sys.argv[sys.argv.index('--fractions') + 1].split(','))
    if '--seeds' in sys.argv and not quick:
        n_seeds = int(sys.argv[sys.argv.index('--seeds') + 1])
        robust_seeds = n_seeds
    epochs = 300 if quick else EPOCHS

    _ensure_dirs()
    law_json = getattr(bd, 'LAW_JSON', 'liu_law.json')
    if not os.path.exists(law_json):                 # the file may live in either place across versions
        alt = os.path.join(RESULTS_DIR, 'liu_law.json')
        if os.path.exists(alt):
            law_json = alt
    if not os.path.exists(law_json):
        raise SystemExit("'%s' missing -- run `python 2_build_dataset.py` first." % law_json)
    law = bd.LiuLaw.load(law_json) if law_json != getattr(bd, 'LAW_JSON', law_json) else bd.LiuLaw.load()
    if getattr(law, 'rho_A_factor', None):
        bd.RHO_A_FACTOR = float(law.rho_A_factor)      # use the factor the dataset was built with
    X, y, yc = load_dataset()
    lo, hi = X.min(0), X.max(0)
    floor = math.sqrt(float(((y - yc) ** 2).mean()))
    oracle_r2 = 1 - float(((y - yc) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    print("Loaded %d rows | features=%s | depth %.2f..%.2f mm, stress %.0f..%.0f MPa | rho_A factor=%.4f"
          % (len(y), FEATURES, lo[0], hi[0], lo[1], hi[1], getattr(bd, 'RHO_A_FACTOR', 1.8736)))
    print("irreducible-error RMSE = %.3f | ORACLE R^2 = %.3f" % (floor, oracle_r2))

    cv_info = {}
    if not final_only and not only_extra:
        res = run_experiments(X, y, yc, law, lo, hi, n_seeds, fractions, epochs)
        summarize(res, fractions, floor, oracle_r2)
        ref = res[(max(fractions), FINAL_CONFIG)]
        cv_info = dict(rmse=float(np.mean([m['rmse'] for m in ref])),
                       rmse_vs_law=float(np.mean([m['rmse_clean'] for m in ref])))
    if do_robust:
        rows, _, _ = bd.load_liu_table()
        run_robustness(X, y, yc, law, rows, lo, hi, robust_seeds, fractions, epochs)
    if do_real:
        rows, _, _ = bd.load_liu_table()
        run_real_validation(rows, bd.read_fem_rows(), 1 if quick else REALVAL_SEEDS, epochs)
    if not only_extra:
        train_final(X, y, law, lo, hi, epochs, cv_info)


if __name__ == '__main__':
    if '--figs-only' in sys.argv:
        figs_only()
    else:
        main()
