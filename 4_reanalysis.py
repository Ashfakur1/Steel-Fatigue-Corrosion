# -*- coding: utf-8 -*-
"""
4_reanalysis.py   (formerly reanalysis.py, v3 -- final; pipeline step 4)
============================
Config-driven, dataset-agnostic reanalysis of corrosion-fatigue S-N data (all real data, not synthetic).

What it does:
  0. Data verification summary + PROVENANCE CHECK (detects duplicated / rescaled data) and the
     duplication evidence table (side-by-side rows for the supplement).
  1. Fits 5 model families for every (dataset, steel, corrosion metric); LOO and
     leave-one-corrosion-group-out (LOGO) CV with percentile bootstrap 95% CIs; slope coefficients
     with t-based 95% CIs; the selection rule is stated in advance. Runouts = censored (Tobit).
  2. Checks the published equations (PRINTED_EQUATIONS) against the data.
  3. Cross-grade: pooled/normalised/transfer across steels sharing the same (stress_def, R); with
     --harmonise, a common Goodman basis allows cross-material pairs (only when the S/f_u ranges overlap).
  3b. STUDY-TO-STUDY TRANSFER: leave-one-study-out on the harmonised basis -- fit on all studies but
     one, predict the held-out study; quantifies the study-to-study transfer gap with bootstrap CIs.
  4. BASE-METAL block: f_u vs f_y normalisation on the corrosion-free (rho=0) steels
     (leave-one-steel-out).
  5. DUPLICATION AUDIT: cleaned vs uncleaned pooled fits for the duplication-flagged pairs -- how much
     do near-duplicate / rescaled copies inflate the apparent precision (Kapoor-Narayanan style check).
  6. SCARCE-DATA LEARNING CURVE on real records: stratified real-record subsampling + refit +
     evaluation on the held-out complement (no synthetic sampling involved).
  7. Sensitivity: note-coded / exclude_reason / ambiguous-N / lower-branch / runout exclusions --
     how much the results change.
  8. REPLICATION MATRIX: whether claims (i)(ii)(v) hold on every dataset, in one table.

Run:
    python 4_reanalysis.py                     # full analysis (CSV -> Results/, figures -> Figures/)
    python 4_reanalysis.py --harmonise         # + cross-material pairs
    python 4_reanalysis.py --selftest          # verify the code itself (Tobit, Goodman, LOO)
    python reanalysis.py --data my.csv --include-all --include-low-branch --drop-runouts --no-figs

Figure outputs:
    Figures/fig1_sn_fits.*, Figures/fig2_logo_pred_vs_obs.*   (pipeline figures, all systems)
    Figures/fig_sn_fits.*, Figures/fig_logo_pred_obs.*        (summary set, 9 systems)
    (fig_fem_verification from 1_fem_pipeline.py; fig_prior_robustness / fig_real_validation
     from 3_train_models.py)

Data file: corrosion_fatigue_data.xlsx (read directly from the 'Data' sheet -- no need to build a CSV;
needs: pip install openpyxl)
or .csv (UTF-8 / Excel 'CSV UTF-8' / comma or semicolon -- all supported).

DATA SCHEMA (a new dataset = new entries; the code does not need changes):
    dataset, steel, group, specimen, n_rep, S, stress_def, R, N, runout, f_u, f_y, f_u_goodman,
    m_<name> ... (corrosion metric, percent), branch, footnote, exclude_reason, source, note
  group: the LOGO unit (corrosion level); n_rep: number of specimens averaged in an entry (weight);
  stress_def: amplitude|range|max;
  runout=1: censored; branch='low': the lower branch of the S-N curve below the authors' knee --
  excluded by default, included with --include-low-branch; entries with a non-empty exclude_reason
  are excluded (included with --include-all).
NOTE limitations: (a) results for group-mean entries refer to "mean values"; (b) the Tobit interval is
approximate; (c) the Goodman conversion is an engineering estimate; (d) a single-line model is wrong for
a bilinear S-N curve -- hence the branch below the knee is separated.
"""
import csv
import math
import os
import sys
from collections import OrderedDict

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # safe piping/redirect on Windows (non-ASCII symbols)
except Exception:
    pass

try:
    from scipy import stats as _st
except ImportError:
    _st = None
try:
    from scipy.optimize import minimize
    from scipy.special import log_ndtr
    HAVE_TOBIT = True
except ImportError:
    HAVE_TOBIT = False

# =====================================================================
# CONFIG
# =====================================================================
DATA_FILES = ['corrosion fatigue data.csv', 'corrosion fatigue data.xlsx',
              'corrosion_fatigue_data.csv', 'corrosion_fatigue_data.xlsx']   # order = priority (deterministic)
OUT_PREFIX = 'reanalysis'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, 'Results')
FIGURES_DIR = os.path.join(_SCRIPT_DIR, 'Figures')


def _ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR):
        if not os.path.isdir(d):
            os.makedirs(d)


# ---------------------------------------------------------------------
# Figure-style and save helpers (previously in a separate figstyle.py -- now inlined here;
# the separate style file was dropped to keep the script self-contained)
# ---------------------------------------------------------------------
_RC_PIPELINE = {   # pipeline figures (fig1/fig2) -- exactly the earlier figstyle.py settings
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
_RC_SUMMARY = {   # summary-figure style (same family as the pipeline figures)
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


def _fig_style(summary=False):
    """Apply the figure style and return plt (idempotent). summary=True gives the
    summary-figure style (same family as the pipeline figures)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update(_RC_SUMMARY if summary else _RC_PIPELINE)
    _ensure_dirs()
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


R_REF = 0.5                 # --harmonise basis: stress RANGE at R=0.5 (following Bai et al. 2021)

MODELS = ['none', 'parallel_lin', 'slope_lin', 'parallel_quad', 'full_quad']
PARSIMONY_TOL = 1.05        # simplest model whose LOGO RMSE <= 1.05 x the best -- rule stated in advance
NORMALISERS = ['raw', 'f_u', 'f_y']
MIN_ROWS = 12
MIN_ROWS_BASE = 6
MIN_GROUPS_LOGO = 3
BASE_N_MAX = 2.0e6          # base-metal block: finite-life part only (N <= this); a single-line model is wrong on the fatigue-limit plateau
OVERLAP_MIN = 0.5           # --harmonise: pairs dropped when the S/f_u range overlap (relative to the smaller) is below this
DUP_N_TOL = 0.05            # provenance: tolerance on the relative difference in N
METRIC_KIND = {'m_rho_me': 'average', 'm_mass_loss': 'average', 'm_rho_A': 'local', 'm_pit_depth_pct': 'local'}

# ---- statistics configuration (stated in advance) ----
BOOT_N = 2000               # percentile bootstrap resamples for the LOGO RMSE / transfer CIs
BOOT_SEED = 20240           # fixed bootstrap seed (reproducibility)
Z95 = 1.959964              # two-sided 95% normal quantile (Wilson intervals)
SCARCE_FRACTIONS = (0.25, 0.5, 0.75)   # real-record learning-curve points (1.0 = full data = the LOGO reference)
SCARCE_SEEDS = 30           # subsample draws per fraction
DUP_MARKERS = ('near-duplicate', 'rescaling')   # exclude_reason markers of duplication-flagged rows
SLOPE_IDX = {'none': 1, 'parallel_lin': 2, 'slope_lin': 2, 'parallel_quad': 3, 'full_quad': 3}

PRINTED_EQUATIONS = [
    dict(dataset='Liu2021', steel='Q550E', metric='m_rho_A', A=(-0.0033, 0.2065, 17.216), m=(0.0015, 0.0975, 5.2662),
         source='Liu et al. 2021, Table 5/Eq.5 as printed (Q550E, quadratic)'),
    dict(dataset='Liu2021', steel='Q460D', metric='m_rho_A', A=(0.0, 0.1261, 22.692), m=(0.0, 0.0638, 7.9959),
         source='Liu et al. 2021, Table 5/Eq.4 as printed (Q460D, linear)'),
]


# =====================================================================
# Data
# =====================================================================
def wilson_ci(k, n, z=Z95):
    """Wilson score interval for a binomial proportion (two-sided, alpha = 0.05); returns fractions."""
    if n <= 0:
        return float('nan'), float('nan')
    p = float(k) / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return ((c - h) / d, (c + h) / d)


def bootstrap_rmse_ci(errs, n_boot=BOOT_N, seed=BOOT_SEED):
    """Percentile bootstrap CI of the (unweighted) RMSE from held-out errors -- the same estimator
    as the LOGO RMSE, so the CI is directly comparable (two-sided, alpha = 0.05)."""
    errs = np.asarray(errs, float)
    errs = errs[errs == errs]
    if len(errs) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errs), (n_boot, len(errs)))
    vals = np.sqrt((errs[idx] ** 2).mean(axis=1))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _is_dup_reason(txt):
    t = (txt or '').strip().lower()
    return any(m in t for m in DUP_MARKERS)


def _f(v):
    try:
        if isinstance(v, str):
            v = v.replace(',', '').replace('\u00a0', '').strip()
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _pos(v):
    return 1.0 if (v != v or v <= 0) else v


_KNOWN = ['dataset', 'steel', 'group', 'specimen', 'n_rep', 'S', 'stress_def', 'R', 'N', 'runout', 'f_u', 'f_y',
          'f_u_goodman', 'branch', 'footnote', 'exclude_reason', 'source', 'note']
_KNOWN_LC = {k.lower(): k for k in _KNOWN}
_TABLE_CACHE = {}


def _canon(h):
    """Clean the header: strip BOM/whitespace; ignore letter case (names like m_rho_A stay intact)"""
    h = str(h).replace('\ufeff', '').strip()
    lc = h.lower()
    if lc in _KNOWN_LC:
        return _KNOWN_LC[lc]
    if lc.startswith('m_'):
        return 'm_' + h[2:]
    return h


def _cell(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if (v == int(v) and abs(v) < 1e15) else repr(v)
    return str(v)


def read_table(path):
    """Read a .xlsx (directly the 'Data' sheet) or a .csv (UTF-8/BOM/cp1252; comma/semicolon/tab); returns (header, rows)"""
    if not os.path.exists(path):
        raise SystemExit("'%s' not found -- put the file in the script folder or pass --data <file>." % path)
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext in ('.xlsx', '.xlsm'):
        try:
            from openpyxl import load_workbook
        except ImportError:
            raise SystemExit("reading xlsx needs openpyxl:  pip install openpyxl   (or save the 'Data' sheet as CSV UTF-8 in Excel)")
        wb = load_workbook(path, read_only=True, data_only=True)
        names = wb.sheetnames
        ws = None
        for n in [x for x in names if x.lower() == 'data'] + names:
            first = next(wb[n].iter_rows(min_row=1, max_row=1, values_only=True), None)
            if first and any(_canon(c) == 'dataset' for c in first if c is not None):
                ws = wb[n]
                break
        if ws is None:
            raise SystemExit("no sheet with a 'dataset' header in the xlsx (sheets: %s). The 'Data' sheet is required." % names)
        it = ws.iter_rows(values_only=True)
        header = [_canon(c) if c is not None else '' for c in next(it)]
        for r in it:
            d = {h: _cell(v) for h, v in zip(header, r) if h}
            if d.get('dataset', '').strip():
                rows.append(d)
        header = [h for h in header if h]
    else:
        raw = open(path, 'rb').read()
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw.decode('cp1252', errors='replace')
        lines = text.splitlines()
        if not lines:
            raise SystemExit("'%s' is empty." % path)
        delim = max([',', ';', '\t'], key=lambda d: lines[0].count(d))
        rd = csv.reader(lines, delimiter=delim)
        header = [_canon(c) for c in next(rd)]
        for rec in rd:
            d = {h: (rec[i].strip() if i < len(rec) else '') for i, h in enumerate(header) if h}
            if d.get('dataset', '').strip():
                rows.append(d)
    _TABLE_CACHE[key] = (header, rows)
    return header, rows


def to_range_at_R(S, stress_def, R, su, R_ref=R_REF):
    """Any (stress_def, R) -> equivalent stress RANGE at R_ref (Goodman, sigma_u = su)."""
    if stress_def == 'amplitude':
        sa = S
    elif stress_def == 'range':
        sa = S / 2.0
    elif stress_def == 'max':
        sa = S * (1.0 - R) / 2.0
    else:
        return float('nan')
    if not (su > 0) or math.isnan(R):
        return float('nan')
    sm = sa * (1.0 + R) / (1.0 - R)
    if sm >= su:
        return float('nan')
    sar = sa / (1.0 - sm / su)
    k = (1.0 + R_ref) / (1.0 - R_ref)
    return 2.0 * sar / (1.0 + k * sar / su)


def load_rows(path, include_all=False, exclude_fn=(), drop_runouts=False, include_low_branch=False,
              restore_dup=False, include_ambiguous=False):
    header, table = read_table(path)
    mcols = [c for c in header if c.startswith('m_')]
    miss = [c for c in ['dataset', 'steel', 'group', 'n_rep', 'S', 'stress_def', 'R', 'N'] if c not in header]
    if miss:
        raise SystemExit("required columns missing from the data file: %s\n  header found: %s\n  (file: %s)" % (miss, header[:12], path))
    if True:
        rows, skipped = [], OrderedDict()
        for raw in table:
            def skip(why):
                skipped[why] = skipped.get(why, 0) + 1
            why_e = (raw.get('exclude_reason') or '').strip()
            # two targeted restorations (used by the dedicated audit sections):
            if why_e and restore_dup and _is_dup_reason(why_e):
                why_e = ''                                  # duplication-flagged rows enter as if clean
            if why_e and include_ambiguous and why_e.startswith('ambiguous'):
                raw = dict(raw, runout='1')                 # ambiguous N: kept as a censored lower bound
                why_e = ''
            if why_e and not include_all:
                skip('exclude_reason'); continue
            if (raw.get('branch') or '').strip() == 'low' and not include_low_branch:
                skip('lower branch (below the knee)'); continue
            is_ro = str(raw.get('runout') or '0').strip().lower() in ('1', 'true', 'yes', 'y')
            if is_ro and (drop_runouts or not HAVE_TOBIT):
                skip('runout (dropped)'); continue
            if (raw.get('footnote') or '').strip() in exclude_fn:
                skip('footnote %s' % raw['footnote'].strip()); continue
            N, S = _f(raw['N']), _f(raw['S'])
            if not (N > 0 and S > 0):
                skip('N or S missing'); continue
            f_u = _f(raw.get('f_u'))
            su = _f(raw.get('f_u_goodman'))
            su = f_u if math.isnan(su) else su
            sd, R = raw['stress_def'], _f(raw['R'])
            rows.append(dict(dataset=raw['dataset'], steel=raw['steel'], group=raw['group'], specimen=raw.get('specimen', ''),
                             n_rep=_pos(_f(raw['n_rep'])), S=S, S_h=to_range_at_R(S, sd, R, su), stress_def=sd, R=R, N=N,
                             cens=is_ro, f_u=f_u, f_y=_f(raw.get('f_y')), footnote=(raw.get('footnote') or '').strip(),
                             branch=(raw.get('branch') or '').strip(), m={c: _f(raw[c]) for c in mcols}))
    return rows, mcols, skipped


def subsets(rows):
    d = OrderedDict()
    for r in rows:
        d.setdefault((r['dataset'], r['steel']), []).append(r)
    return d


def data_summary(rows, mcols, skipped, path=None):
    print("=" * 100)
    print("DATA SUMMARY")
    if not HAVE_TOBIT:
        print("  WARNING scipy missing: runout entries were dropped (censored fit unavailable).")
    for (ds, st), rs in subsets(rows).items():
        phys = int(sum(r['n_rep'] for r in rs))
        print("  %-12s %-11s entries=%-3d (~%d specimens) groups=%d censored=%d R=%s %s"
              % (ds, st, len(rs), phys, len(set(r['group'] for r in rs)), sum(r['cens'] for r in rs),
                 sorted(set(round(r['R'], 2) for r in rs)), sorted(set(r['stress_def'] for r in rs))))
        mets = [m for m in mcols if not all(math.isnan(r['m'][m]) for r in rs)]
        print("               S: %.0f..%.0f | N: %.3g..%.3g | metrics: %s | note-coded: %d"
              % (min(r['S'] for r in rs), max(r['S'] for r in rs), min(r['N'] for r in rs), max(r['N'] for r in rs),
                 mets, sum(1 for r in rs if r['footnote'])))
    if skipped:
        print("  skipped entries:", dict(skipped))
    if path is not None:
        _, raw = read_table(path)
        n_raw = len(raw)
        n_excl = sum(1 for r in raw if (r.get('exclude_reason') or '').strip())
        n_amb = sum(1 for r in raw if (r.get('exclude_reason') or '').strip().startswith('ambiguous'))
        n_dup = sum(1 for r in raw if _is_dup_reason(r.get('exclude_reason')))
        n_phys = int(sum(max(_pos(_f(r.get('n_rep'))), 1.0) for r in rows))
        print("  BOOKKEEPING: registry entries = %d; entering the default fits = %d (~%d specimens);"
              " excluded but preserved in the registry = %d" % (n_raw, len(rows), n_phys, n_excl))
        print("               of the excluded: duplication-flagged = %d (side-by-side evidence in the" % n_dup)
        print("               duplication audit), ambiguous-N = %d (quantified in the sensitivity section as a" % n_amb)
        print("               censored lower bound; the record stays in the registry either way).")
    print("=" * 100)


# =====================================================================
# Fitting (WLS + censored Tobit)
# =====================================================================
def tq(df, p=0.975):
    if df < 1:
        return float('nan')
    if _st is not None:
        return float(_st.t.ppf(p, df))
    z = 1.959964
    return z + (z ** 3 + z) / (4.0 * df)


def design(model, r, x):
    one = np.ones_like(x)
    if model == 'none':
        return np.column_stack([one, -x])
    if model == 'parallel_lin':
        return np.column_stack([one, r, -x])
    if model == 'slope_lin':
        return np.column_stack([one, r, -x, -r * x])
    if model == 'parallel_quad':
        return np.column_stack([one, r, r ** 2, -x])
    if model == 'full_quad':
        return np.column_stack([one, r, r ** 2, -x, -r * x, -(r ** 2) * x])
    raise ValueError(model)


def wls(X, y, w):
    n, p = X.shape
    sw = np.sqrt(w)
    coef, _, rank, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    if rank < p or n <= p:
        return None
    res = y - X @ coef
    s2 = float((w * res ** 2).sum() / (n - p))
    V = np.linalg.pinv((X * w[:, None]).T @ X)
    h = w * np.einsum('ij,jk,ik->i', X, V, X)
    return dict(coef=coef, res=res, s2=s2, V=V, h=h, n=n, p=p, df=n - p, tobit=False)


def fit_tobit(X, y, w, c):
    """Censored Gaussian MLE (on log10 N): a runout entry = 'greater than N'; variance = s^2 / w"""
    n, p = X.shape
    u = ~c
    nu = int(u.sum())
    if nu <= p:
        return None
    f0 = wls(X[u], y[u], w[u] / w[u].mean())
    if f0 is None:
        return None

    def nll(th):
        b, s = th[:p], math.exp(th[p])
        sd = s / np.sqrt(w)
        z = (y - X @ b) / sd
        ll = np.where(c, log_ndtr(-z), -0.5 * z * z - np.log(sd) - 0.9189385332046727)
        return -float(ll.sum())
    r = minimize(nll, np.r_[f0['coef'], 0.5 * math.log(max(f0['s2'], 1e-12))], method='BFGS')
    if not np.isfinite(r.fun):
        return None
    b, s = r.x[:p], math.exp(r.x[p])
    s_corr = s * math.sqrt(nu / max(nu - p, 1))
    V = np.linalg.pinv((X[u] * w[u, None]).T @ X[u])
    return dict(coef=b, res=y - X @ b, s2=s_corr ** 2, V=V, h=None, n=n, p=p, df=nu - p, tobit=True)


def fit_any(X, y, w, c):
    if c is not None and c.any():
        return fit_tobit(X, y, w, c) if HAVE_TOBIT else None
    return wls(X, y, w)


def arrays(rs, metric, norm='raw', basis='native'):
    """metric=None => without a corrosion metric (base metal): m = 0"""
    keyS = 'S' if basis == 'native' else 'S_h'
    if metric is not None:
        rs = [r for r in rs if not math.isnan(r['m'].get(metric, float('nan')))]
    rs = [r for r in rs if not math.isnan(r[keyS])]
    if norm != 'raw':
        rs = [r for r in rs if not math.isnan(r[norm])]
    if not rs:                                   # no data for this metric -- empty arrays (without a warning)
        e = np.array([])
        return [], e, e, e, e, np.array([]), np.array([], bool)
    y = np.log10([r['N'] for r in rs])
    S = np.array([r[keyS] if norm == 'raw' else r[keyS] / r[norm] for r in rs])
    x = np.log10(S)
    m = np.array([r['m'][metric] for r in rs]) / 10.0 if metric is not None else np.zeros(len(rs))
    w = np.array([r['n_rep'] for r in rs], float)
    w = w / w.mean()
    g = np.array([r['group'] for r in rs])
    c = np.array([r['cens'] for r in rs], bool)
    return rs, y, x, m, w, g, c


def loo_generic(X, y, w, c):
    """Leave-one-out RMSE (on uncensored entries). Without censoring: hat-matrix; otherwise refit."""
    if not c.any():
        f = wls(X, y, w)
        if f is None:
            return float('nan')
        e = f['res'] / np.maximum(1.0 - f['h'], 1e-9)
        return math.sqrt(float((w * e ** 2).sum() / w.sum()))
    es, ws = [], []
    for i in np.where(~c)[0]:
        keep = np.arange(len(y)) != i
        f = fit_any(X[keep], y[keep], w[keep] / w[keep].mean(), c[keep])
        if f is None:
            continue
        es.append(float(X[i] @ f['coef'] - y[i])); ws.append(w[i])
    if not es:
        return float('nan')
    es, ws = np.array(es), np.array(ws)
    return math.sqrt(float((ws * es ** 2).sum() / ws.sum()))


def logo(model, y, x, m, w, g, c):
    errs, werrs, hit, hw, ok, tot = [], [], [], [], 0, 0
    for grp in sorted(set(g)):
        te, tr = g == grp, g != grp
        tot += 1
        if tr.sum() < 4:
            continue
        f = fit_any(design(model, m[tr], x[tr]), y[tr], w[tr] / w[tr].mean(), c[tr])
        if f is None or f['df'] < 1:
            continue
        tu = te & ~c
        if not tu.any():
            continue
        ok += 1
        Xte = design(model, m[tu], x[tu])
        e = Xte @ f['coef'] - y[tu]
        w0 = w[tu] / w[tr].mean()
        se = np.sqrt(f['s2'] * (1.0 / w0 + np.einsum('ij,jk,ik->i', Xte, f['V'], Xte)))
        t = tq(f['df'])
        errs += list(e); werrs += list(w0); hit += list(np.abs(e) <= t * se); hw += list(t * se)
    if not errs:
        return dict(rmse=float('nan'), cov=float('nan'), hw=float('nan'), w2=float('nan'), ok=0, tot=tot,
                    err=np.array([]), werr=np.array([]))
    errs, werrs = np.array(errs), np.array(werrs)
    return dict(rmse=math.sqrt(float((errs ** 2).mean())), cov=float(np.mean(hit)), hw=float(np.mean(hw)),
                w2=float(np.mean(np.abs(errs) <= math.log10(2))), ok=ok, tot=tot, err=errs, werr=werrs)


_NOLOGO = dict(rmse=float('nan'), cov=float('nan'), hw=float('nan'), w2=float('nan'), ok=0, tot=0,
               err=np.array([]), werr=np.array([]))


def evaluate(model, y, x, m, w, g, c):
    X = design(model, m, x)
    f = fit_any(X, y, w, c)
    if f is None:
        return None
    out = dict(model=model, p=f['p'], n=f['n'], n_cens=int(c.sum()), sigma=math.sqrt(f['s2']), loo=loo_generic(X, y, w, c), coef=f['coef'])
    lg = logo(model, y, x, m, w, g, c) if len(set(g)) >= MIN_GROUPS_LOGO else _NOLOGO
    out.update(logo_rmse=lg['rmse'], logo_cov=lg['cov'], logo_hw=lg['hw'], logo_w2=lg['w2'], logo_ok=lg['ok'], logo_tot=lg['tot'])
    lo, hi = bootstrap_rmse_ci(lg.get('err', np.array([])))
    out.update(logo_rmse_lo=lo, logo_rmse_hi=hi)
    si = SLOPE_IDX.get(model)
    if si is not None and f.get('V') is not None and f['df'] >= 1:
        se = math.sqrt(max(f['s2'] * float(f['V'][si, si]), 0.0))
        t = tq(f['df'])
        out.update(slope=float(f['coef'][si]), slope_se=se,
                   slope_lo=float(f['coef'][si]) - t * se, slope_hi=float(f['coef'][si]) + t * se)
    else:
        out.update(slope=float('nan'), slope_se=float('nan'), slope_lo=float('nan'), slope_hi=float('nan'))
    return out


def pick(results):
    ok = [r for r in results if not math.isnan(r['logo_rmse'])]
    if not ok:
        return None, None
    best = min(ok, key=lambda r: r['logo_rmse'])
    for r in ok:
        if r['logo_rmse'] <= PARSIMONY_TOL * best['logo_rmse']:
            return best, r
    return best, best


def use_rows(rs, mcols):
    """When one steel carries several metrics, keep only the entries where all metrics are present --
    so the metric comparison uses the same entry set (otherwise a metric supported by a single
    reference entry gains an unfair advantage or disadvantage in LOGO)."""
    act = [mc for mc in mcols if any(not math.isnan(r['m'][mc]) for r in rs)]
    if len(act) > 1:
        return [r for r in rs if all(not math.isnan(r['m'][mc]) for mc in act)]
    return rs


def _is_metric_free(rs, mcols):
    for mc in mcols:
        vals = [r['m'][mc] for r in rs if not math.isnan(r['m'][mc])]
        if vals and (max(vals) - min(vals)) > 0:
            return False
    return True


# =====================================================================
# 0) PROVENANCE CHECK
# =====================================================================
def provenance_check(path):
    rows, _, _ = load_rows(path, include_all=True, include_low_branch=True)
    subs = subsets(rows)
    keys = list(subs.keys())
    print("\n" + "=" * 100)
    print("0) PROVENANCE CHECK  (detect duplicated / rescaled data -- over all entries, exclude_reason included)")
    flagged = False
    evidence = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            A, B = subs[keys[i]], subs[keys[j]]
            a_un = [r for r in A if not r['cens']]
            b_un = [r for r in B if not r['cens']]
            if len(a_un) < 8 or len(b_un) < 8:
                continue
            pair = '%s/%s vs %s/%s' % (keys[i][0], keys[i][1], keys[j][0], keys[j][1])
            # (1) nearly equal N in the same stress cells
            if A[0]['stress_def'] == B[0]['stress_def'] and abs(A[0]['R'] - B[0]['R']) < 0.05:
                small, big = (a_un, b_un) if len(a_un) <= len(b_un) else (b_un, a_un)
                sm_key, bg_key = (keys[i], keys[j]) if len(a_un) <= len(b_un) else (keys[j], keys[i])
                match = 0
                for ra in small:
                    cand = [rb for rb in big if abs(rb['S'] / ra['S'] - 1.0) < 0.01]
                    hit = next((rb for rb in cand if abs(rb['N'] / ra['N'] - 1.0) < DUP_N_TOL), None)
                    if hit is not None:
                        match += 1
                        evidence.append(dict(kind='near-duplicate cell', pair=pair,
                                             dataset_a=sm_key[0], steel_a=sm_key[1], dataset_b=bg_key[0], steel_b=bg_key[1],
                                             S_a=round(ra['S'], 3), S_b=round(hit['S'], 3), N_a=ra['N'], N_b=hit['N'],
                                             N_ratio=round(hit['N'] / ra['N'], 4)))
                if match / len(small) >= 0.6:
                    flagged = True
                    print("  WARNING NEAR-DUPLICATE: %s -- %d/%d entries match in N within +/-%.0f%% at the same stress"
                          % (pair, match, len(small), 100 * DUP_N_TOL))
                    evidence.append(dict(kind='near-duplicate pair summary', pair=pair,
                                         dataset_a=sm_key[0], steel_a=sm_key[1], dataset_b=bg_key[0], steel_b=bg_key[1],
                                         matched_cells=match, n_small=len(small), tolerance_pct=round(100 * DUP_N_TOL, 1)))
            # (2) constant-factor rescaled copy
            n = min(len(a_un), len(b_un))
            if n >= 8 and abs(len(a_un) - len(b_un)) <= 3:
                sa = sorted(a_un, key=lambda r: r['N'])[:n]
                sb = sorted(b_un, key=lambda r: r['N'])[:n]
                ratio = np.array([q['S'] / p['S'] for p, q in zip(sa, sb)])
                dn = np.array([abs(math.log10(q['N'] / p['N'])) for p, q in zip(sa, sb)])
                cv = float(ratio.std() / ratio.mean())
                if cv < 0.02 and np.median(dn) < 0.02 and abs(ratio.mean() - 1.0) > 0.05:
                    flagged = True
                    print("  WARNING RESCALED COPY: %s vs %s -- nearly constant stress ratio (%.3f, CV %.1f%%), median difference in N %.1f%%"
                          % (keys[i][1], keys[j][1], ratio.mean(), 100 * cv, 100 * (10 ** np.median(dn) - 1)))
                    evidence.append(dict(kind='rescaled-copy pair', pair=pair,
                                         dataset_a=keys[i][0], steel_a=keys[i][1], dataset_b=keys[j][0], steel_b=keys[j][1],
                                         stress_ratio=round(float(ratio.mean()), 4), stress_ratio_cv_pct=round(100 * cv, 2),
                                         median_rel_diff_N_pct=round(100 * (10 ** float(np.median(dn)) - 1), 2)))
    if not flagged:
        print("  no duplicated/rescaled pair found.")
    else:
        write_csv('duplication_evidence.csv', evidence)
        print("  (the side-by-side rows behind each warning are in the CSV above -- a numerical-similarity")
        print("   warning only, not a cause finding on its own.)")


# =====================================================================
# 1) model selection
# =====================================================================
def model_selection(subs, mcols, store):
    print("\n" + "=" * 100)
    print("1) MODEL SELECTION  (sigma = residual about the group mean; LOO/LOGO = out-of-sample RMSE, log10 N; cov = 95%-interval coverage)")
    print("   Selection rule (stated in advance): the simplest model whose LOGO RMSE <= %.2f x the best LOGO RMSE" % PARSIMONY_TOL)
    print("   Statistics: LOGO RMSE with percentile bootstrap 95%% CI (two-sided, alpha = 0.05, %d resamples);" % BOOT_N)
    print("   slope m with a t-based 95% CI from the weighted normal equations (Tobit fits: approximate).")
    chosen = {}
    skipped_free = []
    for (ds, st), rs in subs.items():
        if _is_metric_free(rs, mcols):
            skipped_free.append("%s/%s" % (ds, st))
            continue
        seen = {}
        for metric in mcols:
            rsm, y, x, m, w, g, c = arrays(use_rows(rs, mcols), metric)
            if len(rsm) < MIN_ROWS or np.ptp(m) == 0:
                continue
            key = tuple(np.round(m, 9)) + tuple(np.round(y, 9))
            if key in seen:                      # repeated metric with identical values (e.g. m_mass_loss = m_rho_me) -- skip
                print("\n  [%s | %s | %s]  = %s (identical values -- duplicate skipped)" % (ds, st, metric, seen[key]))
                continue
            seen[key] = metric
            res = [e for e in (evaluate(mod, y, x, m, w, g, c) for mod in MODELS) if e is not None]
            best, par = pick(res)
            ng = len(set(g))
            print("\n  [%s | %s | %s]  n=%d entries (%d censored), %d groups" % (ds, st, metric, len(rsm), int(c.sum()), ng))
            if ng < 5:
                print("  WARNING: LOGO has only %d groups -- predictions for unseen groups are extrapolation-dominated; higher-order (quadratic) models can be unstable, read with care" % ng)
            print("  %-14s %2s %7s %7s %15s %6s %7s %8s" % ("model", "p", "sigma", "LOO", "LOGO [95% CI]", "x2%", "cov95", "halfwid"))
            for r in res:
                tag = (" <- best LOGO" if r is best else "") + (" <- PARSIMONIOUS CHOICE" if r is par else "")
                logo_txt = "%9.3f" % r['logo_rmse']
                if r['logo_rmse'] == r['logo_rmse'] and r['logo_rmse_lo'] == r['logo_rmse_lo']:
                    logo_txt = "%6.3f [%5.3f,%5.3f]" % (r['logo_rmse'], r['logo_rmse_lo'], r['logo_rmse_hi'])
                print("  %-14s %2d %7.3f %7.3f %15s %5.0f%% %6.0f%% %8.3f%s" % (
                    r['model'], r['p'], r['sigma'], r['loo'], logo_txt, 100 * r['logo_w2'], 100 * r['logo_cov'], r['logo_hw'], tag))
                lf = round(10 ** r['coef'][1], 3) if r['model'] == 'parallel_lin' else ''
                sm = dict(slope_m='', slope_m_lo='', slope_m_hi='')
                if r['slope'] == r['slope']:
                    sm = dict(slope_m=round(r['slope'], 4), slope_m_lo=round(r['slope_lo'], 4), slope_m_hi=round(r['slope_hi'], 4))
                store.append(dict(dataset=ds, steel=st, metric=metric, model=r['model'], p=r['p'], n=r['n'], n_censored=r['n_cens'],
                                  sigma=round(r['sigma'], 4), loo=round(r['loo'], 4), logo_rmse=round(r['logo_rmse'], 4),
                                  logo_rmse_lo=round(r['logo_rmse_lo'], 4), logo_rmse_hi=round(r['logo_rmse_hi'], 4),
                                  logo_within_x2=round(r['logo_w2'], 3), logo_cov95=round(r['logo_cov'], 3),
                                  logo_halfwidth=round(r['logo_hw'], 4), best_logo=int(r is best), parsimonious=int(r is par),
                                  life_factor_per_plus10pp=lf, n_groups=ng, coef=";".join("%.5g" % v for v in r['coef']), **sm))
            if par is not None:
                chosen[(ds, st, metric)] = par['model']
                cf = par['coef']
                if par['model'] == 'parallel_lin':
                    print("    -> S-N slope m = %.2f [%.2f, %.2f] (fixed); a +10 percentage-point change of the metric"
                          " multiplies life by %.2f (i.e. %.0f%% lower)"
                          % (cf[2], par['slope_lo'], par['slope_hi'], 10 ** cf[1], 100 * (1 - 10 ** cf[1])))
                elif par['model'] == 'parallel_quad':
                    print("    -> S-N slope m = %.2f [%.2f, %.2f] (fixed); from 0 to +10 percentage points of the metric life is multiplied by %.2f"
                          % (cf[3], par['slope_lo'], par['slope_hi'], 10 ** (cf[1] + cf[2])))
                elif par['model'] == 'slope_lin':
                    print("    -> S-N slope at metric = 0: m = %.2f [%.2f, %.2f]; slope change per +10 pp of the metric: %+.3f (log10)"
                          % (cf[2], par['slope_lo'], par['slope_hi'], cf[3]))
    if skipped_free:
        print("\n  (steels without a corrosion metric -- see the base-metal block: %s)" % ", ".join(skipped_free))
    return chosen


# =====================================================================
# 2) printed equations
# =====================================================================
def eq_check(subs, mcols, store):
    print("\n" + "=" * 100)
    print("2) PRINTED-EQUATION CHECK  (error = log10 N predicted - real; uncensored entries only)")
    any_eq = False
    for eq in PRINTED_EQUATIONS:
        rs = subs.get((eq['dataset'], eq['steel']))
        if not rs:
            continue
        any_eq = True
        print("\n  %s  [%s %s]" % (eq['source'], eq['dataset'], eq['steel']))
        variants = [("%s (percent, as printed)" % eq['metric'], eq['metric'], 1.0), ("%s / 100 (fraction)" % eq['metric'], eq['metric'], 0.01)]
        variants += [("%s (percent) [alternative metric]" % mc, mc, 1.0) for mc in mcols if mc != eq['metric']]
        for label, mc, scale in variants:
            use = [r for r in rs if not r['cens'] and not math.isnan(r['m'].get(mc, float('nan')))]
            if not use:
                continue
            e = []
            for r in use:
                v = r['m'][mc] * scale
                A = eq['A'][0] * v * v + eq['A'][1] * v + eq['A'][2]
                mm = eq['m'][0] * v * v + eq['m'][1] * v + eq['m'][2]
                e.append(A - mm * math.log10(r['S']) - math.log10(r['N']))
            e = np.array(e)
            rm = math.sqrt(float((e ** 2).mean()))
            print("    %-42s RMSE=%6.2f  median|e|=%5.2f  worst=%6.2f  bias=%+6.2f" % (label, rm, np.median(np.abs(e)), np.abs(e).max(), e.mean()))
            store.append(dict(source=eq['source'], dataset=eq['dataset'], steel=eq['steel'], variant=label, rmse=round(rm, 3),
                              median_abs=round(float(np.median(np.abs(e))), 3), worst=round(float(np.abs(e).max()), 3), bias=round(float(e.mean()), 3)))
    if any_eq:
        print("  (RMSE <= 0.15 means the equation is consistent with the data; several log10 units means inconsistency.)")
    else:
        print("  (no dataset of PRINTED_EQUATIONS is present in this file.)")


# =====================================================================
# 3) cross-grade / cross-material
# =====================================================================
def _pool(subs, keys, metric, norm, basis):
    ys, xs, ms, ws, cs, ids = [], [], [], [], [], []
    for i, k in enumerate(keys):
        rsm, y, x, m, w, g, c = arrays(subs[k], metric, norm, basis)
        if len(y) == 0:
            return None
        ys.append(y); xs.append(x); ms.append(m); ws.append(w); cs.append(c); ids.append(np.full(len(y), i))
    y, x, m, w, c, ids = map(np.concatenate, (ys, xs, ms, ws, cs, ids))
    return y, x, m, w / w.mean(), c, ids


def _cross_block(title, subs, keys, mcols, basis, store):
    done = False
    for metric in mcols:
        if not all(any(not math.isnan(r['m'][metric]) for r in subs[k]) for k in keys):
            continue
        if any(np.ptp([r['m'][metric] for r in subs[k] if not math.isnan(r['m'][metric])]) == 0 for k in keys):
            continue
        for norm in NORMALISERS:
            if norm != 'raw' and not all(all(not math.isnan(r[norm]) for r in subs[k]) for k in keys):
                continue
            pl = _pool(subs, keys, metric, norm, basis)
            if pl is None:
                continue
            y, x, m, w, c, ids = pl
            for model in ('parallel_lin', 'slope_lin'):
                X = design(model, m, x)
                D = np.column_stack([(ids == i).astype(float) for i in range(1, len(keys))])
                X1 = np.column_stack([X, D])
                f0, f1 = fit_any(X, y, w, c), fit_any(X1, y, w, c)
                if f0 is None or f1 is None:
                    continue
                done = True
                l0, l1 = loo_generic(X, y, w, c), loo_generic(X1, y, w, c)
                print("  [%s | %s | stress/%s | %s]  pooled σ=%.3f LOO=%.3f | + steel-dummy σ=%.3f LOO=%.3f"
                      % (title, metric, norm, model, math.sqrt(f0['s2']), l0, math.sqrt(f1['s2']), l1))
                store.append(dict(kind='pooled', basis=basis, steels=title, metric=metric, normaliser=norm, model=model,
                                  sigma_pooled=round(math.sqrt(f0['s2']), 4), loo_pooled=round(l0, 4),
                                  sigma_dummy=round(math.sqrt(f1['s2']), 4), loo_dummy=round(l1, 4)))
            for a in range(len(keys)):
                for b in range(len(keys)):
                    if a == b:
                        continue
                    _, ya, xa, ma, wa, _, ca = arrays(subs[keys[a]], metric, norm, basis)
                    _, yb, xb, mb, wb, _, cb = arrays(subs[keys[b]], metric, norm, basis)
                    fa = fit_any(design('parallel_lin', ma, xa), ya, wa, ca)
                    if fa is None or (~cb).sum() == 0:
                        continue
                    e = (design('parallel_lin', mb, xb) @ fa['coef'] - yb)[~cb]
                    rm = math.sqrt(float((e ** 2).mean()))
                    print("      transfer %s -> %s (stress/%s, %s): RMSE=%.2f (x%.1f), bias=%+.2f"
                          % (keys[a][1], keys[b][1], norm, metric, rm, 10 ** rm, e.mean()))
                    store.append(dict(kind='transfer', basis=basis, steels="%s->%s" % (keys[a][1], keys[b][1]), metric=metric,
                                      normaliser=norm, model='parallel_lin', rmse=round(rm, 3), bias=round(float(e.mean()), 3)))
    return done


def _norm_range(rs):
    v = [r['S_h'] / r['f_u'] for r in rs if not math.isnan(r['S_h']) and not math.isnan(r['f_u'])]
    return (min(v), max(v)) if v else None


def cross_grade(subs, mcols, store, harmonise):
    print("\n" + "=" * 100)
    print("3) CROSS-GRADE  (native basis: steels with a corrosion metric sharing the same stress_def and R)")
    combos = OrderedDict()
    for key, rs in subs.items():
        if _is_metric_free(rs, mcols):
            continue
        combos.setdefault((rs[0]['stress_def'], round(rs[0]['R'], 1)), []).append(key)
    done = False
    for (sd, R), keys in combos.items():
        if len(keys) >= 2:
            done |= _cross_block(" vs ".join(k[1] for k in keys), subs, keys, mcols, 'native', store)
    if not done:
        print("  (no two corrosion steels share the same (stress_def, R).)")
    if harmonise:
        print("\n  --- HARMONISED basis: stress RANGE at R=%.1f via Goodman; pairs kept only when the S/f_u ranges overlap by at least %.0f%% ---"
              % (R_REF, 100 * OVERLAP_MIN))
        keys = [k for k, rs in subs.items() if not _is_metric_free(rs, mcols) and all(not math.isnan(r['S_h']) for r in rs)]
        used = skipped = 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ra, rb = _norm_range(subs[keys[i]]), _norm_range(subs[keys[j]])
                if ra is None or rb is None:
                    continue
                inter = min(ra[1], rb[1]) - max(ra[0], rb[0])
                ov = inter / min(ra[1] - ra[0], rb[1] - rb[0]) if inter > 0 else 0.0
                if ov < OVERLAP_MIN:
                    skipped += 1
                    print("  (excluded: %s vs %s -- S/f_u range overlap only %.0f%%)" % (keys[i][1], keys[j][1], 100 * ov))
                    continue
                used += 1
                _cross_block("%s vs %s" % (keys[i][1], keys[j][1]), subs, [keys[i], keys[j]], mcols, 'harmonised', store)
        if not used:
            print("  (no suitable pair).")


# =====================================================================
# 4) BASE-METAL block  (steels without corrosion: f_u vs f_y)
# =====================================================================
def base_metal_block(subs, mcols, store):
    free = [k for k, rs in subs.items() if _is_metric_free(rs, mcols) and len(rs) >= MIN_ROWS_BASE]
    if not free:
        return
    print("\n" + "=" * 100)
    print("4) BASE-METAL BLOCK  (steels without corrosion; model: y = a - m x; 'LOSO' = leave-one-steel-out RMSE)")
    print("   finite-life part only: N <= %.1e and runouts excluded (a single-line model is wrong on the fatigue-limit plateau) -- stated in advance" % BASE_N_MAX)
    subs = OrderedDict((k, [r for r in rs if not r['cens'] and r['N'] <= BASE_N_MAX]) for k, rs in subs.items() if k in free)
    free = [k for k in free if len(subs[k]) >= MIN_ROWS_BASE - 2]
    bad_steels = []
    for k in free:
        rsm, y, x, m, w, g, c = arrays(subs[k], None)
        X = design('none', m, x)
        f = fit_any(X, y, w, c)
        if f is None:
            continue
        flags = []
        if len(y) < 6:
            flags.append("n<6")
        if not (1.5 <= f['coef'][1] <= 12.0):
            flags.append("implausible slope m")
        if flags:
            bad_steels.append(k[1])
        print("  %-12s %-11s n=%-3d (censored %d)  m=%.2f  σ=%.3f  LOO=%.3f%s" % (k[0], k[1], len(y), int(c.sum()), f['coef'][1], math.sqrt(f['s2']), loo_generic(X, y, w, c),
                                                                                "   ⚠ " + ", ".join(flags) if flags else ""))
        store.append(dict(kind='single', dataset=k[0], steel=k[1], n=len(y), slope_m=round(float(f['coef'][1]), 3),
                          sigma=round(math.sqrt(f['s2']), 4), loo=round(loo_generic(X, y, w, c), 4)))
    combos = OrderedDict()
    for k in free:
        combos.setdefault((subs[k][0]['stress_def'], round(subs[k][0]['R'], 1)), []).append(k)
    for (sd, R), keys in combos.items():
        if len(keys) < 3:
            continue
        fams = sorted(set(k[1].split('-')[0] for k in keys))
        low_ev = len(fams) < 3 or any(k[1] in bad_steels for k in keys)
        print("\n  pooled across %d steels (stress_def=%s, R=%s): %s" % (len(keys), sd, R, ", ".join(k[1] for k in keys)))
        print("  independent grade families = %d (%s); steels marked weak: %s" % (len(fams), ", ".join(fams), ", ".join(k[1] for k in keys if k[1] in bad_steels) or "none"))
        if low_ev:
            print("  WARNING EVIDENCE LOW: fewer than 3 grade families and/or sparse or implausible data for some steels -- do not claim f_u vs f_y normalisation from the comparison below (exploratory)")
        print("  %-8s %9s %9s %9s %s" % ("stress/", "pooled σ", "LOO", "LOSO", "worst held-out steel"))
        for norm in NORMALISERS:
            if norm != 'raw' and not all(all(not math.isnan(r[norm]) for r in subs[k]) for k in keys):
                continue
            pl = _pool(subs, keys, None, norm, 'native')
            if pl is None:
                continue
            y, x, m, w, c, ids = pl
            X = design('none', m, x)
            f0 = fit_any(X, y, w, c)
            if f0 is None:
                continue
            l0 = loo_generic(X, y, w, c)
            errs_all, worst = [], ('', 0.0)
            for b in range(len(keys)):
                tr, te = ids != b, ids == b
                fb = fit_any(X[tr], y[tr], w[tr] / w[tr].mean(), c[tr])
                tu = te & ~c
                if fb is None or not tu.any():
                    continue
                e = X[tu] @ fb['coef'] - y[tu]
                errs_all += list(e)
                rm = math.sqrt(float((e ** 2).mean()))
                if rm > worst[1]:
                    worst = (keys[b][1], rm)
            loso = math.sqrt(float((np.array(errs_all) ** 2).mean())) if errs_all else float('nan')
            print("  %-8s %9.3f %9.3f %9.3f %s (RMSE %.2f)" % (norm, math.sqrt(f0['s2']), l0, loso, worst[0], worst[1]))
            store.append(dict(kind='pooled_base', evidence='LOW' if low_ev else 'OK', steels=",".join(k[1] for k in keys), normaliser=norm,
                              sigma_pooled=round(math.sqrt(f0['s2']), 4), loo_pooled=round(l0, 4), loso_rmse=round(loso, 4)))
    print("  (a small LOSO means the normalisation predicts an unseen steel well. The S460/S690 pairs of Razmkhah are excluded -- see the duplication audit.)")


# =====================================================================
# 3b) STUDY-TO-STUDY TRANSFER (leave-one-study-out on the harmonised basis)
# =====================================================================
def _pool_arrays(subs, keys, metric, norm, basis, level='study'):
    """Concatenate the arrays() blocks of several systems. Group labels carry the study tag
    ('dataset|steel' or 'dataset|steel|group') so leave-one-out can hold out whole studies
    or corrosion groups without label collisions between studies."""
    ys, xs, ms, ws, cs, gs = [], [], [], [], [], []
    for k in keys:
        rsm, y, x, m, w, g, c = arrays(subs[k], metric, norm, basis)
        if len(y) == 0:
            continue
        tag = '%s|%s' % k
        glab = np.array([tag] * len(y)) if level == 'study' else np.array(['%s|%s' % (tag, gg) for gg in g])
        ys.append(y); xs.append(x); ms.append(m); ws.append(w); cs.append(c); gs.append(glab)
    if not ys:
        return None
    y, x, m, w, c, g = map(np.concatenate, (ys, xs, ms, ws, cs, gs))
    return y, x, m, w / w.mean(), g, c


def study_transfer(subs, mcols, store):
    """Leave-one-STUDY-out transfer: fit the parallel model on all studies but one (same corrosion
    metric, stresses brought to a common Goodman basis), predict the held-out study. The gap
    between the study-level LOGO RMSE and the within-study (group-level) LOGO RMSE quantifies how
    much of the total error is study-to-study variability rather than the corrosion-level effect.
    This is the direct test of 'fit on one study, predict another' (e.g. Lan -> Deng)."""
    did = False
    for metric in mcols:
        keys = []
        for k, rs in subs.items():
            vals = [r['m'][metric] for r in rs if not math.isnan(r['m'].get(metric, float('nan')))]
            if len(vals) < MIN_ROWS or (max(vals) - min(vals)) == 0:
                continue
            if any(math.isnan(r['S_h']) for r in rs if not math.isnan(r['m'].get(metric, float('nan')))):
                continue                          # every entry needs a harmonised stress
            keys.append(k)
        if len(keys) < 2:
            continue
        for norm in ('raw', 'f_u'):
            pl = _pool_arrays(subs, keys, metric, norm, 'harmonised', level='study')
            if pl is None:
                continue
            y, x, m, w, g, c = pl
            if len(set(g)) < 2 or int((~c).sum()) < 8:
                continue
            f = fit_any(design('parallel_lin', m, x), y, w, c)
            if f is None:
                continue
            did = True
            pl_grp = _pool_arrays(subs, keys, metric, norm, 'harmonised', level='group')
            lg_grp = logo('parallel_lin', pl_grp[0], pl_grp[1], pl_grp[2], pl_grp[3], pl_grp[4], pl_grp[5])
            print("\n  [metric %s | stress basis: harmonised range at R=%.1f, normaliser %s | pooled n=%d"
                  " entries (%d censored), %d studies]" % (metric, R_REF, norm, len(y), int(c.sum()), len(set(g))))
            print("  %-26s %5s %8s %8s %9s" % ("held-out study", "n", "RMSE", "bias", "within x2"))
            errs_study = []
            for key in keys:
                tag = '%s|%s' % key
                te, tr = g == tag, g != tag
                ft = fit_any(design('parallel_lin', m[tr], x[tr]), y[tr], w[tr] / w[tr].mean(), c[tr])
                tu = te & ~c
                if ft is None or not tu.any():
                    continue
                e = design('parallel_lin', m[tu], x[tu]) @ ft['coef'] - y[tu]
                errs_study += list(e)
                rm = math.sqrt(float((e ** 2).mean()))
                w2k = float(np.mean(np.abs(e) <= math.log10(2)))
                print("  %-26s %5d %8.3f %+8.3f %8.0f%%" % ('%s %s' % key, int(tu.sum()), rm, float(e.mean()), 100 * w2k))
                store.append(dict(kind='study_transfer', metric=metric, normaliser=norm, held_out='%s %s' % key,
                                  n=int(tu.sum()), rmse=round(rm, 4), bias=round(float(e.mean()), 4),
                                  within_x2=round(w2k, 3)))
            if errs_study:
                es = np.array(errs_study)
                rm = math.sqrt(float((es ** 2).mean()))
                lo, hi = bootstrap_rmse_ci(es)
                grp = lg_grp['rmse']
                infl = (rm / grp) if (grp == grp and grp > 0) else float('nan')
                print("  POOLED study-level LOGO RMSE = %.3f [%.3f, %.3f] (95%% bootstrap CI, %d resamples);"
                      " bias %+.3f" % (rm, lo, hi, BOOT_N, float(es.mean())))
                print("  within-study reference (same model, LOGO by corrosion group) = %.3f"
                      "  ->  study-to-study inflation = x%.2f" % (grp, infl))
                print("  (interpretation: the inflation factor is how much error is added by moving to an unseen"
                      " laboratory/steel; a factor near 1 means the corrosion effect generalises across studies.)")
                store.append(dict(kind='study_transfer_pooled', metric=metric, normaliser=norm, held_out='POOLED',
                                  n=len(es), rmse=round(rm, 4), bias=round(float(es.mean()), 4),
                                  ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                  within_study_logo=round(grp, 4) if grp == grp else '',
                                  inflation_x=round(infl, 2) if infl == infl else ''))
    if not did:
        print("\n  (3b skipped: fewer than two studies share a corrosion metric with harmonisable stresses.)")


# =====================================================================
# 5) DUPLICATION AUDIT (evidence table + cleaned vs uncleaned inflation)
# =====================================================================
def duplication_inflation(path, store):
    """For every duplication-flagged system: restore the flagged rows into the pool ('uncleaned'
    arm) and compare the pooled parallel fit against the default ('cleaned') arm. Reports the
    slope, its t-based CI, sigma and the group-level LOGO RMSE for both arms; the ratio of the
    slope-CI half-widths is the duplication-inflation factor (Kapoor-Narayanan style check:
    duplicated rows must not buy apparent precision)."""
    _, raw = read_table(path)
    flagged = sorted(set((r['dataset'], r['steel']) for r in raw if _is_dup_reason(r.get('exclude_reason'))))
    if not flagged:
        print("  (no duplication-flagged rows -- audit skipped)")
        return
    clean_rows, _, _ = load_rows(path)
    unclean_rows, _, _ = load_rows(path, restore_dup=True)
    clean_subs, unclean_subs = subsets(clean_rows), subsets(unclean_rows)
    print("  flagged systems: %s" % ", ".join('%s %s' % k for k in flagged))
    seen_pools = set()
    for ds_f, st_f in flagged:
        fraw = [r for r in raw if (r['dataset'], r['steel']) == (ds_f, st_f)]
        sd, R = fraw[0]['stress_def'], round(_f(fraw[0]['R']), 2)
        metric = next((mc for mc in ['m_mass_loss', 'm_rho_me', 'm_rho_A', 'm_pit_depth_pct']
                       if any(not math.isnan(_f(r.get(mc))) for r in fraw)), None)
        use_metric = None
        if metric is not None:
            vals = [_f(r.get(metric)) for r in fraw if not math.isnan(_f(r.get(metric)))]
            if vals and (max(vals) - min(vals)) > 0:
                use_metric = metric
        model = 'parallel_lin' if use_metric else 'none'
        si = SLOPE_IDX[model]

        def family(subs_d):
            out = []
            for k, rs in subs_d.items():
                if rs[0]['stress_def'] != sd or round(rs[0]['R'], 2) != R:
                    continue
                if use_metric:
                    if not all(not math.isnan(r['m'].get(use_metric, float('nan'))) for r in rs):
                        continue
                    v = [r['m'][use_metric] for r in rs]
                    if (max(v) - min(v)) <= 0:
                        continue
                elif k[0] != ds_f:
                    continue                      # metric-free audit: keep the family inside the flagged dataset
                out.append(k)
            return out
        keys_c, keys_u = family(clean_subs), family(unclean_subs)
        pool_sig = (model, frozenset(keys_c), frozenset(keys_u))
        if pool_sig in seen_pools:
            continue                       # identical pool already audited (e.g. S460/S690 rescale pair)
        seen_pools.add(pool_sig)
        pl_c = _pool_arrays(clean_subs, keys_c, use_metric, 'raw', 'native', level='group')
        pl_u = _pool_arrays(unclean_subs, keys_u, use_metric, 'raw', 'native', level='group')
        if pl_c is None or pl_u is None:
            continue
        audit_kind = ('corrosion metric %s, model %s' % (use_metric, model)) if use_metric else \
                     ('base S-N audit (no corrosion metric; model %s)' % model)
        print("\n  family of %s %s (%s, R=%.2f; %s): cleaned pool = %s | uncleaned pool = %s"
              % (ds_f, st_f, sd, R, audit_kind,
                 ", ".join(k[1] for k in keys_c) or '(empty)', ", ".join(k[1] for k in keys_u)))
        arm_res = {}
        for label, pl, keys in (('cleaned', pl_c, keys_c), ('uncleaned', pl_u, keys_u)):
            y, x, m, w, g, c = pl
            f = fit_any(design(model, m, x), y, w, c)
            if f is None:
                print("    %-10s fit not possible (n=%d)" % (label, len(y)))
                continue
            se = math.sqrt(max(f['s2'] * float(f['V'][si, si]), 0.0))
            t = tq(f['df'])
            lg = logo(model, y, x, m, w, g, c)
            llo, lhi = bootstrap_rmse_ci(lg.get('err', np.array([])))
            slope, hw = float(f['coef'][si]), t * se
            arm_res[label] = dict(slope=slope, hw=hw, sigma=math.sqrt(f['s2']), logo=lg['rmse'])
            print("    %-10s n=%3d entries (%d censored): slope m = %.2f +/-%.2f | sigma = %.3f |"
                  " LOGO RMSE = %.3f [%.3f, %.3f]"
                  % (label, len(y), int(c.sum()), slope, hw, math.sqrt(f['s2']), lg['rmse'], llo, lhi))
            store.append(dict(kind='dup_inflation', family='%s %s' % ((ds_f, st_f)), arm=label,
                              metric=use_metric or '', model=model, basis='native (%s, R=%.2f)' % (sd, R),
                              steels=";".join('%s %s' % k for k in keys),
                              n=len(y), n_censored=int(c.sum()), slope_m=round(slope, 4),
                              slope_ci_halfwidth=round(hw, 4), sigma=round(math.sqrt(f['s2']), 4),
                              logo_rmse=round(lg['rmse'], 4), logo_ci_lo=round(llo, 4), logo_ci_hi=round(lhi, 4)))
        if 'cleaned' in arm_res and 'uncleaned' in arm_res:
            ac, au = arm_res['cleaned'], arm_res['uncleaned']
            ratio = (au['hw'] / ac['hw']) if ac['hw'] > 0 else float('nan')
            print("    -> duplication-inflation factor (slope-CI half-width, uncleaned/cleaned) = %.2f;"
                  " slope shift %+.3f; LOGO shift %+.3f"
                  % (ratio, au['slope'] - ac['slope'],
                     (au['logo'] - ac['logo']) if (au['logo'] == au['logo'] and ac['logo'] == ac['logo']) else float('nan')))
            print("       (a factor clearly below 1 means the duplicated copies make the slope look more"
                  " precise than the independent evidence supports -- the reason these rows are excluded"
                  " by default and reported separately.)")
            store.append(dict(kind='dup_inflation_factor', family='%s %s' % ((ds_f, st_f)), metric=use_metric or '',
                              model=model, ci_halfwidth_ratio=round(ratio, 3),
                              slope_shift=round(au['slope'] - ac['slope'], 4)))


# =====================================================================
# 6) SCARCE-DATA LEARNING CURVE (real-record subsampling, no synthetic data)
# =====================================================================
def scarce_data(path, chosen, mcols, store):
    """Learning curve on REAL records: for every selected system, subsample the entries stratified
    by corrosion group (group minima preserved), refit the selected model, and evaluate on the
    held-out complement (uncensored entries). Mean +/- s.d. over SCARCE_SEEDS draws per fraction;
    the full-data group-level LOGO RMSE is printed as the reference point."""
    rows, _, _ = load_rows(path)
    subs = subsets(rows)
    print("   (real-record subsampling: stratified by corrosion group; evaluation on the held-out"
          " complement; %d draws per fraction; two-sided 95%% reference intervals are not added because" % SCARCE_SEEDS)
    print("    the quantity of interest is the mean curve, reported with its across-draw s.d.)")
    for (ds, st, metric), model in chosen.items():
        rs = subs.get((ds, st))
        if not rs:
            continue
        rsm, y, x, m, w, g, c = arrays(use_rows(rs, mcols), metric)
        if len(y) == 0 or int((~c).sum()) < 12 or len(set(g)) < 2:
            continue
        X = design(model, m, x)
        lg = logo(model, y, x, m, w, g, c) if len(set(g)) >= MIN_GROUPS_LOGO else _NOLOGO
        print("\n  [%s | %s | %s | %s]  n=%d entries (%d groups), full-data LOGO RMSE = %.3f"
              % (ds, st, metric, model, len(y), len(set(g)), lg['rmse']))
        for frac in SCARCE_FRACTIONS:
            rms, ntr, nte = [], [], []
            for seed in range(SCARCE_SEEDS):
                rng = np.random.default_rng(7919 * int(round(100 * frac)) + seed)
                tr_idx = []
                for grp in sorted(set(g)):
                    gi = np.where(g == grp)[0]
                    k = max(1, int(round(frac * len(gi))))
                    tr_idx += list(rng.choice(gi, size=min(k, len(gi)), replace=False))
                tr_idx = np.array(sorted(tr_idx))
                te_mask = np.ones(len(y), bool)
                te_mask[tr_idx] = False
                tu = te_mask & ~c
                if len(tr_idx) < 6 or not tu.any():
                    continue
                f = fit_any(X[tr_idx], y[tr_idx], w[tr_idx] / w[tr_idx].mean(), c[tr_idx])
                if f is None or f['df'] < 1:
                    continue
                e = X[tu] @ f['coef'] - y[tu]
                rms.append(math.sqrt(float((e ** 2).mean())))
                ntr.append(len(tr_idx)); nte.append(int(tu.sum()))
            if not rms:
                continue
            print("    %-16s n_train ~ %2d  ->  held-out RMSE = %.3f +/- %.3f (n_test ~ %d; %d/%d draws valid)"
                  % ('%d%% of records' % round(100 * frac), float(np.mean(ntr)), float(np.mean(rms)),
                     float(np.std(rms)), float(np.mean(nte)), len(rms), SCARCE_SEEDS))
            store.append(dict(dataset=ds, steel=st, metric=metric, model=model, fraction=frac,
                              n_train_mean=round(float(np.mean(ntr)), 1), n_test_mean=round(float(np.mean(nte)), 1),
                              rmse_mean=round(float(np.mean(rms)), 4), rmse_sd=round(float(np.std(rms)), 4),
                              n_valid_draws=len(rms), full_logo_rmse=round(lg['rmse'], 4) if lg['rmse'] == lg['rmse'] else ''))


# =====================================================================
# 7) sensitivity
# =====================================================================
def sensitivity(path, chosen, mcols):
    print("\n" + "=" * 100)
    print("7) SENSITIVITY  (chosen model; how the results change when the treatment of questionable entries / runouts / the lower branch changes)")
    full, _, _ = load_rows(path, include_all=True, include_low_branch=True)
    codes = sorted(set(r['footnote'] for r in full if r['footnote']))
    has_low = any(r['branch'] == 'low' for r in full)
    has_ro = any(r['cens'] for r in full)
    base_all, _, _ = load_rows(path)
    has_excl = len(full) != len(base_all) + sum(1 for r in full if r['branch'] == 'low')
    _, raw_all = read_table(path)
    has_amb = any((r.get('exclude_reason') or '').strip().startswith('ambiguous') for r in raw_all)
    variants = [("base (runout = censored Tobit)", dict())]
    for cd in codes:
        variants.append(("exclude note '%s'" % cd, dict(exclude_fn=(cd,))))
    if len(codes) > 1:
        variants.append(("exclude all note-coded", dict(exclude_fn=tuple(codes))))
    if has_low:
        variants.append(("INCLUDE lower branch (points below the knee)", dict(include_low_branch=True)))
    variants.append(("include all (ignore exclude_reason)", dict(include_all=True)))
    if has_amb:
        variants.append(("ambiguous record INCLUDED (censored lower bound)", dict(include_ambiguous=True)))
    variants.append(("DROP runouts (naive)", dict(drop_runouts=True)))
    out = []
    for (ds, st, metric), model in chosen.items():
        print("\n  [%s | %s | %s | %s]" % (ds, st, metric, model))
        print("  %-52s %4s %7s %7s %9s %9s" % ("variant", "n", "sigma", "LOO", "LOGO", "slope m"))
        seen = set()
        for label, kw in variants:
            rows, _, _ = load_rows(path, **kw)
            rs = subsets(rows).get((ds, st))
            if not rs:
                continue
            rsm, y, x, m, w, g, c = arrays(use_rows(rs, mcols), metric)
            sig = (len(y), int(c.sum()), round(float(y.sum()), 6))
            if sig in seen:
                continue
            seen.add(sig)
            e = evaluate(model, y, x, m, w, g, c)
            if e is None:
                continue
            mi = {'none': 1, 'parallel_lin': 2, 'slope_lin': 2, 'parallel_quad': 3, 'full_quad': 3}[model]
            print("  %-52s %4d %7.3f %7.3f %9.3f %9.2f" % (label, e['n'], e['sigma'], e['loo'], e['logo_rmse'], e['coef'][mi]))
            out.append(dict(dataset=ds, steel=st, metric=metric, model=model, variant=label, n=e['n'], sigma=round(e['sigma'], 4),
                            loo=round(e['loo'], 4), logo_rmse=round(e['logo_rmse'], 4), slope_m=round(float(e['coef'][mi]), 3)))
    return out


# =====================================================================
# 8) REPLICATION MATRIX
# =====================================================================
def replication_matrix(models_out, base_out):
    print("\n" + "=" * 100)
    print("8) REPLICATION MATRIX  (LOGO RMSE, log10 N; smaller = better)")
    print("   'corrosion helps?' = best corrosion model / none model (<=0.80 YES, <=0.95 WEAK, otherwise NO)")
    print("   the slope direction is reported only when the parallel model is not sufficient in CV -- otherwise 'n.s.' (no evidence;")
    print("   'n.s.' = the parsimony rule finds no benefit beyond the parallel model; it is a model-selection statement, not a null test)")
    by = OrderedDict()
    for r in models_out:
        by.setdefault((r['dataset'], r['steel'], r['metric']), {})[r['model']] = r

    def logo_of(d, mod):
        v = d.get(mod, {}).get('logo_rmse')
        return float('nan') if v is None else v
    rows = []
    print("  %-10s %-11s %-15s %3s %2s | %6s %8s %6s | %-13s | %-9s | %-26s | %5s" % (
        "dataset", "steel", "metric", "n", "G", "none", "parallel", "slope", "corrosion helps?", "parallel", "slope direction", "cov95"))
    for (ds, st, mt), d in by.items():
        ln, lp, ls = logo_of(d, 'none'), logo_of(d, 'parallel_lin'), logo_of(d, 'slope_lin')
        corr = [v for v in (lp, ls, logo_of(d, 'parallel_quad'), logo_of(d, 'full_quad')) if not math.isnan(v)]
        ratio = (min(corr) / ln) if (corr and not math.isnan(ln) and ln > 0) else float('nan')
        helps = 'n/a' if math.isnan(ratio) else ('YES' if ratio <= 0.80 else ('WEAK' if ratio <= 0.95 else 'NO'))
        helps_txt = "%s (x%.2f)" % (helps, ratio) if helps != 'n/a' else 'n/a'
        others = [v for v in (ls, logo_of(d, 'parallel_quad'), logo_of(d, 'full_quad')) if not math.isnan(v)]
        adequate = ''
        if not math.isnan(lp) and others:
            adequate = 'YES' if lp <= PARSIMONY_TOL * min(others + [lp]) else 'NO'
        raw_dm = ''
        if 'slope_lin' in d:
            cf = [float(v) for v in d['slope_lin']['coef'].split(';')]
            raw_dm = "%+.2f" % cf[3]
        if helps == 'NO':
            sd_txt = 'n/a (corrosion not predictive)'
        elif adequate == 'NO' and raw_dm:
            sd_txt = "m %s (%s per 10pp)" % ('rises' if float(raw_dm) > 0 else 'FALLS', raw_dm)
        else:
            sd_txt = 'n.s. (parallel suffices)'
        par = next((v for v in d.values() if v['parsimonious']), None)
        cov = par['logo_cov95'] if par else float('nan')
        n = next(iter(d.values()))['n']
        ng = next(iter(d.values())).get('n_groups', '')
        cov_txt = ("%3.0f%%%s" % (100 * cov, '⚠' if cov < 0.90 else ' ')) if not math.isnan(cov) else '  n/a'
        print("  %-10s %-11s %-15s %3d %2s | %6.3f %8.3f %6.3f | %-16s | %-9s | %-26s | %s" % (
            ds, st, mt, n, ng, ln, lp, ls, helps_txt, adequate, sd_txt, cov_txt))
        row = dict(claim='corrosion/slope', dataset=ds, steel=st, metric=mt, n=n, n_groups=ng, logo_none=ln, logo_parallel=lp, logo_slope=ls,
                   corrosion_helps=helps, ratio_best_over_none=round(ratio, 3) if not math.isnan(ratio) else '',
                   parallel_adequate=adequate, slope_change_raw=raw_dm, slope_direction=sd_txt, cov95=cov)
        if par is not None and par.get('logo_rmse_lo') not in (None, ''):
            row.update(parsimonious_model=par['model'], logo_ci_lo=par['logo_rmse_lo'], logo_ci_hi=par['logo_rmse_hi'])
        if par is not None and par.get('logo_cov95') == par.get('logo_cov95') and par.get('logo_ok'):
            cl, ch = wilson_ci(int(round(float(par['logo_cov95']) * par['logo_ok'])), par['logo_ok'])
            row.update(cov95_wilson_lo=round(cl, 3), cov95_wilson_hi=round(ch, 3), cov95_n_folds=par['logo_ok'])
        rows.append(row)
    print("  LOGO 95% bootstrap CIs for the parsimonious model of each system:")
    for (ds, st, mt), d in by.items():
        par = next((v for v in d.values() if v.get('parsimonious')), None)
        if par is not None and par.get('logo_rmse') == par.get('logo_rmse') and par.get('logo_rmse_lo') == par.get('logo_rmse_lo'):
            print("     %-10s %-11s %-15s %.3f [%.3f, %.3f]" % (ds, st, mt, float(par['logo_rmse']), float(par['logo_rmse_lo']), float(par['logo_rmse_hi'])))
    k_yes = sum(1 for r in rows if r.get('claim') == 'corrosion/slope' and r.get('corrosion_helps') == 'YES')
    n_sys = sum(1 for r in rows if r.get('claim') == 'corrosion/slope')
    if n_sys:
        wlo, whi = wilson_ci(k_yes, n_sys)
        print("  SUMMARY: the corrosion metric improves LOGO (ratio <= 0.80) in %d/%d systems;"
              " Wilson 95%% CI for that proportion: %.0f%%..%.0f%%" % (k_yes, n_sys, 100 * wlo, 100 * whi))
    print("  (G = number of LOGO groups; the flag marks interval coverage below 90%.)")
    print("\n  (ii) average vs local corrosion metric -- best LOGO within the same steel:")
    groups = OrderedDict()
    for (ds, st, mt), d in by.items():
        best = min([v['logo_rmse'] for v in d.values() if not math.isnan(v['logo_rmse'])] or [float('nan')])
        groups.setdefault((ds, st), {})[mt] = best
    found = False
    for (ds, st), dm_ in groups.items():
        avg = {k: v for k, v in dm_.items() if METRIC_KIND.get(k) == 'average'}
        loc = {k: v for k, v in dm_.items() if METRIC_KIND.get(k) == 'local'}
        if avg and loc:
            found = True
            ba, bl = min(avg.items(), key=lambda t: t[1]), min(loc.items(), key=lambda t: t[1])
            ratio = ba[1] / bl[1]
            verdict = 'average metric sufficient (<=1.15x)' if ratio <= 1.15 else 'local metric better'
            print("     %-10s %-11s average %-12s %.3f  vs  local %-16s %.3f   ratio %.2f -> %s" % (ds, st, ba[0], ba[1], bl[0], bl[1], ratio, verdict))
            rows.append(dict(claim='average_vs_local', dataset=ds, steel=st, metric="%s|%s" % (ba[0], bl[0]), logo_none=ba[1], logo_parallel=bl[1],
                             parallel_adequate=verdict, slope_direction='ratio %.2f' % ratio))
    if not found:
        print("     (no steel carries both metric types)")
    base = [r for r in base_out if r.get('kind') == 'pooled_base']
    if base:
        print("\n  (iv) base-metal normalization (leave-one-steel-out RMSE) -- evidence: %s" % base[0].get('evidence', '?'))
        for r in base:
            print("     stress/%-4s  LOSO=%.3f  pooled σ=%.3f   [%s]" % (r['normaliser'], r['loso_rmse'], r['sigma_pooled'], r['steels']))
            rows.append(dict(claim='normaliser_base_metal', dataset='base', steel=r['steels'], metric=r['normaliser'], logo_none=r['loso_rmse'],
                             slope_direction='evidence ' + r.get('evidence', '?')))
        if base[0].get('evidence') == 'LOW':
            print("     WARNING evidence LOW: do not claim f_u/f_y normalisation from this comparison.")
    write_csv('%s_matrix.csv' % OUT_PREFIX, rows)


# =====================================================================
# Figures
# =====================================================================
def figures(subs, mcols, chosen):
    """Pipeline figures: fig1_sn_fits and fig2_logo_pred_vs_obs (all systems, per dataset).
    The summary figures fig_sn_fits / fig_logo_pred_obs are drawn by summary_figures()."""
    try:
        plt = _fig_style()
        from matplotlib.ticker import ScalarFormatter, NullFormatter
    except ImportError:
        print("\n(matplotlib missing -- figures skipped)")
        return
    keys = [k for k, rs in subs.items() if not _is_metric_free(rs, mcols)]
    if not keys:
        return

    # ---------------- Figure 1: S-N fit per dataset ----------------
    n1 = len(keys)
    ncol = min(n1, 4)
    nrow = int(math.ceil(n1 / float(ncol)))
    last = n1 - (nrow - 1) * ncol
    start = (ncol - last) // 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.7 * nrow), squeeze=False)
    targets = [(r, c) for r in range(nrow - 1) for c in range(ncol)]
    targets += [(nrow - 1, c) for c in range(start, start + last)]
    used = set(targets)
    for r in range(nrow):
        for c in range(ncol):
            if (r, c) not in used:
                axs[r][c].axis('off')
    row_left = {}
    for r, c in targets:
        row_left[r] = min(row_left.get(r, c), c)
    sf = ScalarFormatter()
    sf.set_powerlimits((-3, 4))
    for (ri, ci), k in zip(targets, keys):
        ax = axs[ri][ci]
        metric = next((mc for mc in mcols if any(not math.isnan(r['m'][mc]) for r in subs[k])), None)
        rsm, y, x, m, w, g, c = arrays(use_rows(subs[k], mcols), metric)
        if len(y) == 0:
            ax.axis('off')
            continue
        f = fit_any(design('parallel_lin', m, x), y, w, c) if len(y) > 4 else None
        cmap = plt.get_cmap('tab20')
        for gi, grp in enumerate(sorted(set(g))):
            sel = g == grp
            col = cmap(gi % 20)
            ax.scatter(10 ** x[sel & ~c], 10 ** y[sel & ~c], s=13, color=col, label=grp, zorder=3)
            if (sel & c).any():
                ax.scatter(10 ** x[sel & c], 10 ** y[sel & c], s=26, marker='>', facecolors='none',
                           edgecolors='k', linewidths=0.8, zorder=3)
            if f is not None and sel.any():
                xs = np.linspace(x.min(), x.max(), 20)
                ax.plot(10 ** xs, 10 ** (f['coef'][0] + f['coef'][1] * m[sel].mean() - f['coef'][2] * xs),
                        lw=1.1, color=col, zorder=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(sf)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=9)
        ax.set_xlabel('Stress (MPa, native)', fontsize=10)
        if ci == row_left[ri]:
            ax.set_ylabel('N (cycles)', fontsize=10)
        ax.set_title('%s %s\nmetric: %s%s' % (k[0], k[1], METRIC_LABEL.get(metric, metric),
                    '   (> marks runouts)' if c.any() else ''), fontsize=10)
        n_grp = len(set(g))
        if 2 <= n_grp <= 7:
            ax.legend(fontsize=7.5, loc='best', handlelength=1.2, handletextpad=0.4,
                      borderaxespad=0.2, labelspacing=0.25)
    paths = _fig_save(fig, 'fig1_sn_fits')
    print("Figure 1 saved: %s" % ", ".join(os.path.basename(p) for p in paths))

    # ---------------- Figure 2: LOGO predicted vs observed ----------------
    n = max(len(chosen), 1)
    ncol = min(n, 5)
    nrow = int(math.ceil(n / float(ncol)))
    last = n - (nrow - 1) * ncol
    start = (ncol - last) // 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.7 * nrow), squeeze=False)
    targets = [(r, c) for r in range(nrow - 1) for c in range(ncol)]
    targets += [(nrow - 1, c) for c in range(start, start + last)]
    used = set(targets)
    for r in range(nrow):
        for c in range(ncol):
            if (r, c) not in used:
                axs[r][c].axis('off')
    row_left = {}
    for r, c in targets:
        row_left[r] = min(row_left.get(r, c), c)
    for (ri, ci), ((ds, st, metric), model) in zip(targets, chosen.items()):
        ax = axs[ri][ci]
        rsm, y, x, m, w, g, c = arrays(use_rows(subs[(ds, st)], mcols), metric)
        if len(y) == 0:
            ax.axis('off')
            continue
        pr, ob = [], []
        for grp in sorted(set(g)):
            te, tr = g == grp, g != grp
            f = fit_any(design(model, m[tr], x[tr]), y[tr], w[tr] / w[tr].mean(), c[tr])
            tu = te & ~c
            if f is None or not tu.any():
                continue
            pr += list(design(model, m[tu], x[tu]) @ f['coef'])
            ob += list(y[tu])
        if not ob:
            ax.axis('off')
            continue
        err = np.array(pr) - np.array(ob)
        rmse = math.sqrt(float((err ** 2).mean()))
        lo = min(min(ob), min(pr)) - 0.2
        hi = max(max(ob), max(pr)) + 0.2
        band = [lo, hi]
        ax.fill_between(band, [l - math.log10(2) for l in band], [l + math.log10(2) for l in band],
                        color='0.87', zorder=0)
        ax.plot(band, band, 'k-', lw=0.9, zorder=1)
        ax.plot(band, [l + math.log10(2) for l in band], 'k:', lw=0.7, zorder=1)
        ax.plot(band, [l - math.log10(2) for l in band], 'k:', lw=0.7, zorder=1)
        ax.scatter(ob, pr, s=13, c='#3b6db4', alpha=0.85, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=9)
        ax.text(0.04, 0.94, 'RMSE = %.2f' % rmse, transform=ax.transAxes, fontsize=9.5,
                va='top', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', alpha=0.9))
        ax.set_xlabel('Observed log$_{10}$N', fontsize=10)
        if ci == row_left[ri]:
            ax.set_ylabel('LOGO predicted\nlog$_{10}$N', fontsize=10)
        ax.set_title('%s %s\n%s (%s)' % (ds, st, METRIC_LABEL.get(metric, metric),
                    MODEL_LABEL.get(model, model)), fontsize=10)
    paths = _fig_save(fig, 'fig2_logo_pred_vs_obs')
    print("Figure 2 saved: %s" % ", ".join(os.path.basename(p) for p in paths))


# =====================================================================
# Summary figures (fig_sn_fits, fig_logo_pred_obs) -- same style as the pipeline
# figures. 9 reanalysis systems, in the display order, built with this script's own
# fit/LOGO machinery.
# =====================================================================
SUMMARY_SYSTEMS = [
    ('Liu2021',   'Q460D',      'm_rho_A',         'Liu et al. (2021) Q460D plate'),
    ('Liu2021',   'Q550E',      'm_rho_A',         'Liu et al. (2021) Q550E plate'),
    ('Jiang2018', 'HSW-Jiang',  'm_mass_loss',     'Jiang et al. (2018) wire'),
    ('Lan2018',   'HSW-Lan',    'm_mass_loss',     'Lan et al. (2018) wire'),
    ('Liu2017',   'HSW-III',    'm_mass_loss',     'Liu et al. (2017) wire'),
    ('Guo2021',   'Q690D-weld', 'm_mass_loss',     'Guo et al. (2021) Q690D weld'),
    ('Zhou2024',  'Q420',       'm_mass_loss',     'Zhou et al. (2024) Q420 plate'),
    ('Deng2022',  'HSW-Deng',   'm_mass_loss',     'Deng et al. (2022) suspender wire'),
    ('Li2021',    'HSW-Li5',    'm_pit_depth_pct', 'Li et al. (2021) bridge wire'),
]

MODEL_LABEL = {
    'none':          'baseline (no corrosion)',
    'parallel_lin':  'parallel fit',
    'slope_lin':     'slope fit',
    'parallel_quad': 'parallel quadratic fit',
    'full_quad':     'full quadratic fit',
}
METRIC_LABEL = {
    'm_rho_A':         'max. section loss $\\rho_A$ (%)',
    'm_rho_me':        'mean section loss $\\rho_{me}$ (%)',
    'm_mass_loss':     'mass loss (%)',
    'm_pit_depth_pct': 'pit depth (%)',
}
XLBL = {'amplitude': 'Stress amplitude (MPa)', 'range': 'Stress range (MPa)', 'max': 'Maximum stress (MPa)'}


def _summary_fit(subs, mcols, ds, st, metric):
    """Fit the 5 model families for one system + parsimony selection stated in advance (the model_selection rule)."""
    rs = subs.get((ds, st))
    if not rs:
        return None
    rsm, y, x, m, w, g, c = arrays(use_rows(rs, mcols), metric)
    if len(rsm) < MIN_ROWS or np.ptp(m) == 0:
        return None
    res = [e for e in (evaluate(mod, y, x, m, w, g, c) for mod in MODELS) if e is not None]
    if not res:
        return None
    best, par = pick(res)
    if par is None:
        return None
    return dict(rs=rsm, y=y, x=x, m=m, w=w, g=g, c=c, model=par['model'], coef=par['coef'],
                rmse=par['logo_rmse'], stress_def=rsm[0]['stress_def'])


def summary_figures(subs, mcols):
    """Summary figures fig_sn_fits and fig_logo_pred_obs -- the 9 reanalysis systems.
    Same fit/LOGO machinery as this script, so the figure RMSE / model selection
    match reanalysis_models.csv exactly."""
    fits = []
    for ds, st, metric, label in SUMMARY_SYSTEMS:
        f = _summary_fit(subs, mcols, ds, st, metric)
        if f is None:
            print("  [warning] %s %s (%s) could not be fitted -- panel dropped from the summary figures" % (ds, st, metric))
        else:
            f['label'] = label
            f['metric'] = metric
            fits.append(f)
    if not fits:
        print("  (summary figures skipped -- no system could be fitted)")
        return
    try:
        plt = _fig_style(summary=True)
        from matplotlib.ticker import ScalarFormatter, NullFormatter
        import matplotlib.colors as mcolors
        import matplotlib.cm as cmx
    except ImportError as e:
        print("(matplotlib missing -- summary figures skipped: %s)" % e)
        return
    n = len(fits)
    ncol = 3
    nrow = int(math.ceil(n / float(ncol)))
    sf = ScalarFormatter()
    sf.set_powerlimits((-1, 7))

    # ---------------- fig_sn_fits: S-N fit for the 9 systems ----------------
    fig, axs = plt.subplots(nrow, ncol, figsize=(10.4, 3.1 * nrow), squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis('off')
    for pi, (ax, f) in enumerate(zip(axs.ravel(), fits)):
        y, x, m, w, g, c = f['y'], f['x'], f['m'], f['w'], f['g'], f['c']
        gmean = {grp: float(m[g == grp].mean()) * 10.0 for grp in sorted(set(g))}
        vmin, vmax = min(gmean.values()), max(gmean.values())
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('viridis')
        model = f['model']
        if model != 'none':
            ffull = fit_any(design(model, m, x), y, w, c)
        for grp in sorted(set(g)):
            sel = g == grp
            col = cmap(norm(gmean[grp]))
            ax.scatter(10 ** x[sel & ~c], 10 ** y[sel & ~c], s=16, color=col, zorder=3, linewidths=0)
            if (sel & c).any():
                ax.scatter(10 ** x[sel & c], 10 ** y[sel & c], s=26, marker='^', color=col,
                           edgecolors='k', linewidths=0.5, zorder=4)
            if model != 'none' and sel.any():
                xs = np.linspace(x.min(), x.max(), 30)
                Xd = design(model, np.full_like(xs, m[sel].mean()), xs)
                ax.plot(10 ** xs, 10 ** (Xd @ ffull['coef']), lw=1.0, color=col, zorder=2)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(sf)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(XLBL.get(f['stress_def'], 'Stress (MPa, native)'))
        if pi % ncol == 0:
            ax.set_ylabel('Life $N$ (cycles)')
        ax.set_title(f['label'], pad=5)
        ax.text(0.04, 0.06, MODEL_LABEL[model], transform=ax.transAxes, fontsize=9,
                style='italic', va='bottom')
        sm = cmx.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cl = plt.colorbar(sm, ax=ax, fraction=0.055, pad=0.03)
        cl.set_label(METRIC_LABEL.get(f.get('metric', ''), ''), fontsize=8.5)
        cl.ax.tick_params(labelsize=8)
    paths = _fig_save(fig, 'fig_sn_fits')
    print("Figure saved: %s" % ", ".join(os.path.basename(p) for p in paths))

    # ---------------- fig_logo_pred_obs: LOGO predicted vs observed ----------------
    fig, axs = plt.subplots(nrow, ncol, figsize=(11.0, 3.2 * nrow), squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis('off')
    for pi, (ax, f) in enumerate(zip(axs.ravel(), fits)):
        y, x, m, w, g, c = f['y'], f['x'], f['m'], f['w'], f['g'], f['c']
        model = f['model']
        pr, ob, gr = [], [], []
        for grp in sorted(set(g)):
            te, tr = g == grp, g != grp
            ff = fit_any(design(model, m[tr], x[tr]), y[tr], w[tr] / w[tr].mean(), c[tr])
            tu = te & ~c
            if ff is None or not tu.any():
                continue
            pr += list(design(model, m[tu], x[tu]) @ ff['coef'])
            ob += list(y[tu])
            gr += [grp] * int(tu.sum())
        if not ob:
            ax.axis('off')
            continue
        pr, ob = np.array(pr), np.array(ob)
        err = pr - ob
        rmse = math.sqrt(float((err ** 2).mean()))
        lo = min(min(ob), min(pr)) - 0.25
        hi = max(max(ob), max(pr)) + 0.25
        band = [lo, hi]
        ax.fill_between(band, [l - math.log10(2) for l in band], [l + math.log10(2) for l in band],
                        color='#cfe0f2', zorder=0)
        ax.plot(band, band, '--', color='0.35', lw=1.1, zorder=1)
        pct = {grp: float(m[g == grp].mean()) * 10.0 for grp in set(gr)}
        vmin, vmax = min(pct.values()), max(pct.values())
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('viridis')
        cols = cmap(norm(np.array([pct[grp] for grp in gr])))
        ax.scatter(ob, pr, s=22, c=cols, zorder=3, linewidths=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.text(0.05, 0.93, 'RMSE %.3f' % rmse, transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.5', alpha=0.95))
        ax.set_xlabel('Observed log$_{10}N$')
        if pi % ncol == 0:
            ax.set_ylabel('LOGO predicted\nlog$_{10}N$')
        ax.set_title(f['label'], pad=5)
        sm = cmx.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cl = plt.colorbar(sm, ax=ax, fraction=0.055, pad=0.03)
        cl.set_label('corrosion level (%)', fontsize=8.5)
        cl.ax.tick_params(labelsize=8)
        print('    LOGO %-34s %-12s RMSE=%.3f (%d pts)' % (f['label'], model, rmse, len(ob)))
    paths = _fig_save(fig, 'fig_logo_pred_obs')
    print("Figure saved: %s" % ", ".join(os.path.basename(p) for p in paths))


def write_csv(path, rows):
    if not rows:
        return
    _ensure_dirs()
    path = os.path.join(RESULTS_DIR, path)
    keys = list(OrderedDict((k, 0) for r in rows for k in r).keys())
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print("Saved: %s" % path)


# =====================================================================
# SELF-TEST
# =====================================================================
def selftest():
    print("SELF-TEST")
    ok_all = True

    def chk(name, cond, detail=''):
        nonlocal ok_all
        ok_all &= bool(cond)
        print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))
    # 1) Goodman: printed value of Bai et al. (2021) (520 MPa range, R=0.4 -> R=0.5, sigma_u=1770 => 473.62)
    v = to_range_at_R(520.0, 'range', 0.4, 1770.0, 0.5)
    chk("Goodman conversion matches Bai's printed value", abs(v - 473.62) < 0.1, "(%.2f vs 473.62)" % v)
    # 2) Tobit: recovers the true coefficients on 30% censored synthetic data and beats naive dropping
    if HAVE_TOBIT:
        rng = np.random.default_rng(3)
        n = 400
        r = rng.uniform(0, 2, n); S = rng.uniform(250, 550, n); x = np.log10(S)
        true = np.array([17.0, -0.4, 5.0])
        ylat = true[0] + true[1] * r - true[2] * x + rng.normal(0, 0.15, n)
        cut = np.quantile(ylat, 0.70)
        cens = ylat > cut
        y = np.where(cens, cut, ylat)
        X = design('parallel_lin', r, x)
        ft = fit_any(X, y, np.ones(n), cens)
        fn = wls(X[~cens], y[~cens], np.ones((~cens).sum()))
        et, en = abs(ft['coef'][2] - true[2]), abs(fn['coef'][2] - true[2])
        chk("Tobit recovers the slope (censored %.0f%%)" % (100 * cens.mean()), et < 0.25 and et < en,
            "(Tobit err %.3f, naive-drop err %.3f; true slope 5.0)" % (et, en))
        chk("Tobit sigma approximately correct (true 0.15)", abs(math.sqrt(ft['s2']) - 0.15) < 0.04, "(%.3f)" % math.sqrt(ft['s2']))
    else:
        print("  (scipy missing -- Tobit test skipped)")
    # 3) LOO: hat-matrix vs brute force
    rng = np.random.default_rng(5)
    n = 40
    xx = rng.uniform(2.1, 2.6, n); rr = rng.uniform(0, 2, n); yy = 15 - 0.3 * rr - 4.5 * xx + rng.normal(0, 0.1, n)
    X = design('parallel_lin', rr, xx); w = np.ones(n)
    fast = loo_generic(X, yy, w, np.zeros(n, bool))
    es = []
    for i in range(n):
        keep = np.arange(n) != i
        f = wls(X[keep], yy[keep], w[keep]); es.append(float(X[i] @ f['coef'] - yy[i]))
    brute = math.sqrt(float(np.mean(np.square(es))))
    chk("LOO (hat-matrix) = brute-force LOO", abs(fast - brute) < 1e-6, "(%.5f vs %.5f)" % (fast, brute))
    print("SELF-TEST %s" % ("PASSED" if ok_all else "FAILED"))
    return ok_all


def main():
    argv = sys.argv[1:]
    if '--selftest' in argv:
        sys.exit(0 if selftest() else 1)
    if '--data' in argv:
        path = argv[argv.index('--data') + 1]
    else:
        cands = [p for p in DATA_FILES if os.path.exists(p)]
        if not cands:
            raise SystemExit("corrosion fatigue data.xlsx/.csv (or corrosion_fatigue_data.xlsx/.csv) not found -- put the file in this folder (or use --data <file>).")
        path = cands[0]   # the order of DATA_FILES is the priority (deterministic)
    print("Data file: %s" % path)
    include_all = '--include-all' in argv
    drop_ro = '--drop-runouts' in argv
    inc_low = '--include-low-branch' in argv
    exclude_fn = tuple(argv[argv.index('--exclude-fn') + 1].split(',')) if '--exclude-fn' in argv else ()
    rows, mcols, skipped = load_rows(path, include_all, exclude_fn, drop_ro, inc_low)
    subs = subsets(rows)
    data_summary(rows, mcols, skipped, path)
    provenance_check(path)
    models_out, eq_out, cg_out, base_out, transfer_out, dup_out, scarce_out = [], [], [], [], [], [], []
    chosen = model_selection(subs, mcols, models_out)
    eq_check(subs, mcols, eq_out)
    cross_grade(subs, mcols, cg_out, '--harmonise' in argv)
    print("\n" + "=" * 100)
    print("3b) STUDY-TO-STUDY TRANSFER  (leave-one-study-out on the harmonised stress basis)")
    print("    fit on all studies but one -> predict the held-out study (e.g. Lan -> Deng); compared")
    print("    with the within-study LOGO reference to separate study-to-study variability from the")
    print("    corrosion-level effect.")
    study_transfer(subs, mcols, transfer_out)
    base_metal_block(subs, mcols, base_out)
    print("\n" + "=" * 100)
    print("5) DUPLICATION AUDIT  (cleaned vs uncleaned pooled fits for the duplication-flagged rows)")
    print("    quantifies how much apparent precision duplicated rows would add if treated as independent data.")
    duplication_inflation(path, dup_out)
    print("\n" + "=" * 100)
    print("6) SCARCE-DATA LEARNING CURVE  (real-record subsampling; refit on the draw, evaluate on the complement)")
    scarce_data(path, chosen, mcols, scarce_out)
    sens_out = sensitivity(path, chosen, mcols) if not (include_all or exclude_fn or drop_ro or inc_low) else []
    replication_matrix(models_out, base_out)
    print()
    write_csv('%s_models.csv' % OUT_PREFIX, models_out)
    write_csv('%s_eq_check.csv' % OUT_PREFIX, eq_out)
    write_csv('%s_cross_grade.csv' % OUT_PREFIX, cg_out + base_out)
    write_csv('%s_transfer.csv' % OUT_PREFIX, transfer_out)
    write_csv('%s_duplication_inflation.csv' % OUT_PREFIX, dup_out)
    write_csv('%s_scarce_data.csv' % OUT_PREFIX, scarce_out)
    write_csv('%s_sensitivity.csv' % OUT_PREFIX, sens_out)
    if '--no-figs' not in argv:
        figures(subs, mcols, chosen)
        summary_figures(subs, mcols)


if __name__ == '__main__':
    main()
