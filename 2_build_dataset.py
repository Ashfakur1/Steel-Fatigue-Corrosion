# -*- coding: utf-8 -*-
"""
2_build_dataset.py   (STEP 2 of 5)
==================================
Liu S-N law fit + verification gate + synthetic training-grid generation.

INPUTS :  liu_table4.csv                    (Liu et al. 2021, Table 4 -- this folder)
          Results/fem_results_raw.csv       (STEP 1 output; bundled copy ships with the pipeline)
OUTPUTS :  Results/training_dataset.csv     synthetic grid for STEP 4
           Results/liu_law.json             verified S-N law (prior)
           Results/build_dataset_report.txt full fit/gate/verification report

Run:
    python 2_build_dataset.py                # build the dataset
    python 2_build_dataset.py --verify       # only the S-N fit report
    python 2_build_dataset.py --sensitivity  # how much life changes when the rho_A factor changes
    python 2_build_dataset.py --factor 1.0   # build the dataset with an alternative factor
                                             #   (1.0 = strictly uniform; default 1.8736 =
                                             #   the pit-equivalent of Liu Table 4)

train_models.py imports this module (filename-based load; prefix-safe).
"""
import csv
import json
import math
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')

# =====================================================================
# CONFIG (all constants in one place)
# =====================================================================
LIU_TABLE_CSV = os.path.join(BASE_DIR, 'liu_table4.csv')
FEM_CSV = os.path.join(RESULTS_DIR, 'fem_results_raw.csv')
MESH_CSV = os.path.join(RESULTS_DIR, 'fem_mesh_study.csv')
OUT_CSV = os.path.join(RESULTS_DIR, 'training_dataset.csv')
LAW_JSON = os.path.join(RESULTS_DIR, 'liu_law.json')

ORIGINAL_THICKNESS_MM = 10.0
SIGMA_UTS_MPA = 696.8          # Liu et al. (2021), Table 3, Q550E
R_TEST = 0.1

# Uniform thickness loss (rho_me) -> the rho_A of Liu (pit-caused maximum section loss).
# 1.8736 = the zero-intercept regression of rho_A/rho_me over the E-series of Liu Table 4.
# NOTE modelling assumption: strictly uniform would be 1.0. Measure the effect with --sensitivity.
RHO_A_FACTOR = 1.8736

HCF_MIN_CYCLES = 100
GRID_MAX_DEPTH_MM = 2.0        # default max depth of the FEM grid (taken from the FEM file when present)

MIN_TABLE_ROWS = 15
MAX_RESIDUAL_SIGMA = 0.5
NOISE_MODE = 'fit_residual'
NOISE_SEED = 123
GAUGE_TOL = 0.01
RHO_SCALE = 10.0
GAP_WARN_PP = 5.0              # warn when Table 4 has a larger gap inside the grid range

# NOTE UNVERIFIED -- previously extracted coefficients of the published Eq.5 (per-% units);
# for comparison / stress tests only
PUBLISHED_COEFFS_UNVERIFIED = {'a': (17.216, 0.2065, -0.0033),
                               'b': (5.2662, 0.0975, 0.0015)}


# =====================================================================
# Small utility: write to both the console and the report file
# =====================================================================
class _Tee(object):
    def __init__(self, *targets):
        self.targets = targets

    def write(self, s):
        for t in self.targets:
            try:
                t.write(s)
            except Exception:
                pass

    def flush(self):
        for t in self.targets:
            try:
                t.flush()
            except Exception:
                pass


def start_report(filename='build_dataset_report.txt'):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    fh = open(path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, fh)
    return fh


def stop_report(fh):
    try:
        sys.stdout = sys.__stdout__
        fh.close()
    except Exception:
        pass


# =====================================================================
# Physics helpers
# =====================================================================
def _factor(f=None):
    return RHO_A_FACTOR if f is None else f


def rho_A_from_depth(depth_mm, factor=None):
    """corrosion depth (mm) -> rho_A (%)"""
    return _factor(factor) * np.asarray(depth_mm, dtype=float) / ORIGINAL_THICKNESS_MM * 100.0


def depth_from_rho_A(rho_A_pct, factor=None):
    """rho_A (%) -> equivalent uniform corrosion depth (mm) -- the inverse of rho_A_from_depth"""
    return np.asarray(rho_A_pct, dtype=float) / _factor(factor) / 100.0 * ORIGINAL_THICKNESS_MM


def equivalent_amplitude_R01(sigma_max_mpa, sigma_uts=SIGMA_UTS_MPA, r_ref=R_TEST):
    """FEA peak stress (assuming R=0) -> equivalent amplitude on Liu's R=0.1 baseline (Goodman, one step)"""
    s = np.asarray(sigma_max_mpa, dtype=float)
    sa = s / 2.0
    sm = s / 2.0
    if np.any(sm >= sigma_uts):
        raise ValueError("mean stress equal to / above UTS -- not physical")
    sa_rev = sa / (1.0 - sm / sigma_uts)
    ratio = (1.0 + r_ref) / (1.0 - r_ref)
    return sa_rev / (1.0 + ratio * sa_rev / sigma_uts)


def peak_stress_from_equivalent_amplitude(s_eq, sigma_uts=SIGMA_UTS_MPA, r_ref=R_TEST):
    """Exact closed-form inverse of equivalent_amplitude_R01: Liu's S_a (R=0.1) -> R=0 peak stress.
    Needed to map a real specimen onto the model input (real-data validation)."""
    s = np.asarray(s_eq, dtype=float)
    ratio = (1.0 + r_ref) / (1.0 - r_ref)
    sa_rev = s / (1.0 - ratio * s / sigma_uts)
    sa = sa_rev / (1.0 + sa_rev / sigma_uts)
    return 2.0 * sa


class LiuLaw(object):
    """log10 N = A(rho_A) - m(rho_A) * log10(S_eq); r = rho_A/rho_scale;
    A = a0 + a1 r + a2 r^2 ,  m = b0 + b1 r + b2 r^2"""

    def __init__(self, a, b, rho_scale=RHO_SCALE, sigma=0.0, rho_min=0.0,
                 rho_max=0.0, n=0, r2=float('nan'), rho_A_factor=None):
        self.a = [float(v) for v in a]
        self.b = [float(v) for v in b]
        self.rho_scale = float(rho_scale)
        self.sigma = float(sigma)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.n = int(n)
        self.r2 = float(r2)
        self.rho_A_factor = rho_A_factor

    @classmethod
    def from_percent_coeffs(cls, a, b, rho_scale=RHO_SCALE):
        s = rho_scale
        return cls((a[0], a[1] * s, a[2] * s * s), (b[0], b[1] * s, b[2] * s * s), rho_scale)

    def A_m(self, rho_A):
        r = np.asarray(rho_A, dtype=float) / self.rho_scale
        A = self.a[0] + self.a[1] * r + self.a[2] * r ** 2
        m = self.b[0] + self.b[1] * r + self.b[2] * r ** 2
        return A, m

    def log10N(self, rho_A, S_eq):
        A, m = self.A_m(rho_A)
        return A - m * np.log10(np.asarray(S_eq, dtype=float))

    def to_dict(self):
        return dict(a=self.a, b=self.b, rho_scale=self.rho_scale, sigma=self.sigma,
                    rho_min=self.rho_min, rho_max=self.rho_max, n=self.n, r2=self.r2,
                    rho_A_factor=self.rho_A_factor)

    def save(self, path=LAW_JSON):
        d = self.to_dict()
        d['rho_A_factor'] = _factor(self.rho_A_factor)
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path=LAW_JSON):
        with open(path, 'r') as f:
            d = json.load(f)
        return cls(**d)


# =====================================================================
# Liu Table 4 loading + fit
# =====================================================================
def load_liu_table(path=LIU_TABLE_CSV):
    """CSV columns: specimen, rho_A_pct, stress_amp_mpa, N_cycles, [runout], [rho_me_pct], [exclude_reason]
    '#'-prefixed comment lines. Specimens with runout=1 or an exclude_reason are left out of
    the fit (reported with the reason)."""
    if not os.path.exists(path):
        raise SystemExit("'%s' not found (provide the specimen data of Liu Table 4)." % path)
    with open(path, 'r') as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith('#')]
    rows, n_runout, excluded = [], 0, []
    for row in csv.DictReader(lines):
        try:
            rho = float(row['rho_A_pct'])
            S = float(row['stress_amp_mpa'])
            N = float(row['N_cycles'])
        except (KeyError, TypeError, ValueError):
            spec = (row.get('specimen') or '?').strip()
            print("NOTE: specimen %-8s is outside the fit -- N unclear/incomplete "
                  "(e.g. the '750,97' censored bound of E0-1; n=29 per the source table)." % spec)
            continue
        why = (row.get('exclude_reason') or '').strip()
        if why:
            excluded.append(((row.get('specimen') or '?').strip(), why))
            continue
        if str(row.get('runout') or '0').strip().lower() in ('1', 'true', 'yes', 'y'):
            n_runout += 1
            continue
        rme = None
        try:
            rme = float(row.get('rho_me_pct'))
        except (TypeError, ValueError):
            pass
        rows.append(dict(specimen=(row.get('specimen') or '?').strip(), rho=rho, S=S, N=N, rme=rme))
    return rows, n_runout, excluded


def fit_liu_law(rho, S, N, quadratic=True):
    """Least squares. quadratic=False => A, m linear in rho only (for the structural-misspecification
    test). Returns: law, residuals, LOO-sigma"""
    rho = np.asarray(rho, float)
    r = rho / RHO_SCALE
    x = np.log10(np.asarray(S, float))
    y = np.log10(np.asarray(N, float))
    if quadratic:
        X = np.column_stack([np.ones_like(r), r, r ** 2, -x, -r * x, -(r ** 2) * x])
    else:
        X = np.column_stack([np.ones_like(r), r, -x, -r * x])
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    n, p = X.shape
    if rank < p:
        raise SystemExit("S-N fit not possible (rank %d < %d): more specimens / different rho_A levels needed." % (rank, p))
    resid = y - X @ coef
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    sigma = math.sqrt(sse / (n - p)) if n > p else float('nan')
    h = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
    loo = resid / np.maximum(1.0 - h, 1e-9)
    sigma_loo = math.sqrt(float((loo ** 2).mean()))
    if quadratic:
        a, b = coef[0:3], coef[3:6]
    else:
        a, b = (coef[0], coef[1], 0.0), (coef[2], coef[3], 0.0)
    law = LiuLaw(a, b, sigma=sigma, rho_min=rho.min(), rho_max=rho.max(), n=n,
                 r2=1.0 - sse / sst if sst > 0 else float('nan'), rho_A_factor=_factor())
    return law, resid, sigma_loo


def coverage_gaps(rho_values, grid_max_rho):
    """Where Table 4 has no specimen inside the grid range [0, grid_max] -- returns (gap_pp, from, to)"""
    pts = sorted(set([0.0] + [float(v) for v in rho_values if 0.0 <= v <= grid_max_rho] + [float(grid_max_rho)]))
    best = (0.0, 0.0, 0.0)
    for lo, hi in zip(pts[:-1], pts[1:]):
        if hi - lo > best[0]:
            best = (hi - lo, lo, hi)
    return best


def check_law(law, grid_max_rho, rho_values=None):
    problems, warns = [], []
    if law.n < MIN_TABLE_ROWS:
        problems.append("only %d specimens present (at least %d needed)." % (law.n, MIN_TABLE_ROWS))
    if not (law.sigma <= MAX_RESIDUAL_SIGMA):
        problems.append("residual scatter of the fit sigma=%.2f > %.2f (log10) -- data/units may be wrong."
                        % (law.sigma, MAX_RESIDUAL_SIGMA))
    rr = np.linspace(0.0, grid_max_rho, 60)
    _, m = law.A_m(rr)
    if np.any(m <= 0):
        problems.append("m(rho) <= 0 somewhere in the rho_A=0..%.1f%% range -- the S-N law is unusable." % grid_max_rho)
    if grid_max_rho > law.rho_max * 1.0001:
        warns.append("max rho_A of the dataset=%.1f%% > max %.1f%% of Table 4 -- extrapolation." % (grid_max_rho, law.rho_max))
    life = law.log10N(rr, 170.0)
    if np.any(np.diff(life) > 1e-9):
        warns.append("life increases somewhere when rho_A grows at S_eq=170 MPa -- the fit shape is suspicious.")
    if rho_values is not None:
        gap, g0, g1 = coverage_gaps(rho_values, grid_max_rho)
        if gap > GAP_WARN_PP:
            warns.append("DATA GAP: Table 4 has no specimen between rho_A=%.1f%% and %.1f%% (%.1f pp); "
                         "that part of the grid (depth ~ %.2f..%.2f mm) is interpolated, relying on the "
                         "quadratic shape of the law."
                         % (g0, g1, gap, float(depth_from_rho_A(g0)), float(depth_from_rho_A(g1))))
    return problems, warns


def report_law(law, resid, sigma_loo, rows):
    print("\n" + "=" * 72)
    print("Liu S-N fit (from our own Table 4 data):  n=%d  R^2=%.3f  sigma=%.3f  LOO-sigma=%.3f (log10)"
          % (law.n, law.r2, law.sigma, sigma_loo))
    print("%-8s %7s %8s %10s %12s %9s" % ("spec", "rho_A%", "S(MPa)", "exp N", "fit N", "log10 err"))
    for rw, e in zip(rows, resid):
        print("%-8s %7.2f %8.2f %10d %12.0f %+9.2f" % (rw['specimen'], rw['rho'], rw['S'], rw['N'],
                                                       rw['N'] * 10 ** (-e), -e))
    pub = LiuLaw.from_percent_coeffs(PUBLISHED_COEFFS_UNVERIFIED['a'], PUBLISHED_COEFFS_UNVERIFIED['b'])
    errs = [float(pub.log10N(rw['rho'], rw['S']) - math.log10(rw['N'])) for rw in rows]
    print("Comparison: using the published Eq.5 coefficients directly gives median|log10 err| = %.2f, worst = %.2f"
          % (float(np.median(np.abs(errs))), float(np.max(np.abs(errs)))))
    print("=" * 72)


def grid_max_depth(path=FEM_CSV):
    """Maximum corrosion depth present in the FEM results file (default when the file is missing)."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            ds = [float(r['corrosion_depth_mm']) for r in csv.DictReader(f)]
        if ds:
            return max(ds)
    return GRID_MAX_DEPTH_MM


def calibrate(grid_max_depth_mm=None, verbose=True, save=True):
    if grid_max_depth_mm is None:
        grid_max_depth_mm = grid_max_depth()
    rows, n_runout, excluded = load_liu_table()
    if verbose:
        if n_runout:
            print("%d runout specimens left out of the fit (censored)." % n_runout)
        if excluded:
            print("Excluded specimens (%d, with reasons):" % len(excluded))
            for name, why in excluded[:8]:
                print("   %-8s %s" % (name, why))
            if len(excluded) > 8:
                print("   ... and %d more" % (len(excluded) - 8))
        pairs = [(r['rho'], r['rme']) for r in rows if r['rme']]
        if len(pairs) >= 10:
            a = np.array([p[0] for p in pairs]); m = np.array([p[1] for p in pairs])
            f = float((a * m).sum() / (m * m).sum())
            print("rho_A/rho_me (zero-intercept, n=%d) = %.4f ; ratio range %.2f..%.2f ; config RHO_A_FACTOR = %.4f%s"
                  % (len(pairs), f, float((a / m).min()), float((a / m).max()), RHO_A_FACTOR,
                     "" if abs(f / RHO_A_FACTOR - 1) < 0.05 else "   <-- mismatch (see --factor or the config)"))
    if len(rows) < 7:
        raise SystemExit("liu_table4.csv has only %d specimens -- at least %d needed." % (len(rows), MIN_TABLE_ROWS))
    law, resid, sigma_loo = fit_liu_law([r['rho'] for r in rows], [r['S'] for r in rows], [r['N'] for r in rows])
    if verbose:
        report_law(law, resid, sigma_loo, rows)
    grid_max_rho = float(rho_A_from_depth(grid_max_depth_mm))
    if verbose:
        print("Maximum depth of the FEM grid = %.2f mm  (rho_A ~ %.1f%%)" % (grid_max_depth_mm, grid_max_rho))
    problems, warns = check_law(law, grid_max_rho, [r['rho'] for r in rows])
    if verbose:
        for w in warns:
            print("WARNING: " + w)
    if problems:
        print("\nGATE FAILED -- dataset generation stopped:")
        for p in problems:
            print("  - " + p)
        raise SystemExit(1)
    if verbose:
        print("GATE PASSED (sigma=%.2f <= %.2f).  rho_A factor = %.4f" % (law.sigma, MAX_RESIDUAL_SIGMA, RHO_A_FACTOR))
    if save:
        law.save()
    return law


# =====================================================================
# Dataset generation (3_train_models.py also calls this -- in the real-data validation)
# =====================================================================
def read_fem_rows(path=FEM_CSV):
    if not os.path.exists(path):
        raise SystemExit("'%s' missing -- run `python 1_fem_pipeline.py` first "
                         "(or place the bundled fem_results_raw.csv in the Results/ folder)." % path)
    by_job, dup, stray = {}, 0, []
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            if row['job_name'].startswith('mesh_'):      # mesh-study jobs never enter the training data
                stray.append(row['job_name'])
                continue
            if row['job_name'] in by_job:
                dup += 1
            by_job[row['job_name']] = row
    if stray:
        print("NOTE: %s contained %d mesh-study jobs (%s ...) -- left out of the training set."
              % (path, len(stray), stray[0]))
    if dup:
        print("WARNING: %d duplicate job_name entries -- keeping the last entry." % dup)
    cells = {}
    for r in by_job.values():
        key = (round(float(r['corrosion_depth_mm']), 4), round(float(r['applied_stress_mpa']), 4))
        cells[key] = cells.get(key, 0) + 1
    n_dup_cells = sum(1 for c in cells.values() if c > 1)
    if n_dup_cells:
        print("WARNING: %d (depth, stress) grid cells contain more than one job -- risk of train/test"
              % n_dup_cells)
        print("         leakage; clean the file.")
    return list(by_job.values())


def generate_dataset(law, fem_rows, seed=NOISE_SEED, noise_sigma=None):
    """FEA entries -> (entries, excluded_count, sigma_noise). noise_sigma=None => law.sigma (per NOISE_MODE)"""
    rng = np.random.default_rng(seed)
    if noise_sigma is None:
        noise_sigma = law.sigma if NOISE_MODE == 'fit_residual' else 0.0
    out, excluded = [], 0
    for row in fem_rows:
        depth = float(row['corrosion_depth_mm'])
        applied = float(row['applied_stress_mpa'])
        gauge = float(row['mean_principal_stress_mpa'])
        if applied > 0 and abs(gauge / applied - 1.0) > GAUGE_TOL:
            raise SystemExit("job %s: FEA gauge stress (%.1f) differs from applied (%.1f) by more than "
                             "1%% -- the 2-feature model is no longer valid." % (row['job_name'], gauge, applied))
        rho_A = float(rho_A_from_depth(depth))
        s_eq = float(equivalent_amplitude_R01(gauge))
        log_clean = float(law.log10N(rho_A, s_eq))
        if log_clean < math.log10(HCF_MIN_CYCLES):
            excluded += 1
            continue
        log_noisy = log_clean + (rng.normal(0.0, noise_sigma) if noise_sigma > 0 else 0.0)
        if log_noisy < math.log10(HCF_MIN_CYCLES):
            excluded += 1
            continue
        out.append(dict(corrosion_depth_mm=depth, applied_stress_mpa=applied, gauge_stress_mpa=gauge,
                        rho_A_pct=round(rho_A, 4), S_eq_mpa=round(s_eq, 4), log10N_clean=round(log_clean, 5),
                        fatigue_life_cycles=10 ** log_noisy, log10_fatigue_life=round(log_noisy, 5)))
    return out, excluded, noise_sigma


# =====================================================================
# FEA verification -- quantitative report of what the FEA actually provides
# =====================================================================
def fea_verification(fem_rows, mesh_csv=MESH_CSV):
    rm, rx = [], []
    for r in fem_rows:
        a = float(r['applied_stress_mpa'])
        if a > 0:
            rm.append(float(r['mean_principal_stress_mpa']) / a)
            rx.append(float(r['max_principal_stress_mpa']) / a)
    print("\n" + "=" * 72)
    print("FEA VERIFICATION (n=%d jobs)" % len(rm))
    print("  gauge-mean / applied stress : %.4f .. %.4f   (analytical sigma=F/A = 1.0000)" % (min(rm), max(rm)))
    print("  gauge-max  / applied stress : %.4f .. %.4f   (max deviation %.2f%% -- grip/Poisson edge effect)"
          % (min(rx), max(rx), 100 * (max(rx) - 1.0)))
    print("  => verifies the FEA stress state (uniform uniaxial). The stress does not depend on depth;")
    print("     the corrosion effect enters only through rho_A in the S-N law. Present the FEA as a")
    print("     verification, not as a source of new information.")
    if os.path.exists(mesh_csv):
        groups = {}
        with open(mesh_csv, 'r') as f:
            for r in csv.DictReader(f):
                groups.setdefault((r['corrosion_depth_mm'], r['applied_stress_mpa']), []).append(
                    (float(r['mesh_size_mm']), float(r['max_principal_stress_mpa']), float(r['mean_principal_stress_mpa'])))
        print("  MESH STUDY (%s):" % os.path.basename(mesh_csv))
        for key, vals in sorted(groups.items()):
            vals.sort(reverse=True)    # coarse to fine
            s = "  depth=%s S=%s : " % key + " | ".join("h=%.2f max=%.3f mean=%.3f" % v for v in vals)
            if len(vals) >= 2:
                s += "   d(mean, finest two)=%.4f%%" % (100 * abs(vals[-1][2] - vals[-2][2]) / vals[-1][2])
            print(s)
    else:
        print("  (no mesh study: run `python 1_fem_pipeline.py --mesh-study` -> %s)" % os.path.basename(mesh_csv))
    print("=" * 72)


# =====================================================================
# rho_A factor sensitivity
# =====================================================================
def sensitivity():
    global RHO_A_FACTOR
    saved = RHO_A_FACTOR
    rows, _, _ = load_liu_table()
    ratios = [r['rho'] / r['rme'] for r in rows if r['rme']]
    lo_f, hi_f = (min(ratios), max(ratios)) if ratios else (1.5, 2.5)
    factors = [('uniform (1.0)', 1.0), ('Liu min obs', lo_f), ('Liu regression', saved), ('Liu max obs', hi_f)]
    law = calibrate(verbose=False, save=False)
    gmax = grid_max_depth()
    cases = [(0.0, 350.0)] + [(d, st) for d in (round(gmax / 2, 2), gmax) for st in (290.0, 430.0)]
    print("\nrho_A FACTOR SENSITIVITY  (law fitted from our own Table 4; only the depth -> rho_A mapping changes)")
    print("%-10s %-8s | " % ("depth(mm)", "S(MPa)") + " | ".join("%-22s" % ("%s=%.2f" % (n, f)) for n, f in factors))
    ref_idx = 2
    try:
        for d, s in cases:
            vals = []
            for _, f in factors:
                RHO_A_FACTOR = f
                vals.append(float(law.log10N(rho_A_from_depth(d), equivalent_amplitude_R01(s))))
            cells = ["N=%9.0f (x%.2f)" % (10 ** v, 10 ** (v - vals[ref_idx])) for v in vals]
            print("%-10.2f %-8.0f | " % (d, s) + " | ".join("%-22s" % c for c in cells))
    finally:
        RHO_A_FACTOR = saved
    print("(x..) = ratio relative to the Liu-regression factor. Strictly uniform gives 1.0 (max rho_A ="
          " %.1f%%) -- the data gap of Table 4 then grows." % float(rho_A_from_depth(grid_max_depth(), 1.0)))


# =====================================================================
# main
# =====================================================================
def build():
    fh = start_report()
    try:
        law = calibrate()
        fem = read_fem_rows()
        out, excluded, sigma_noise = generate_dataset(law, fem)
        fields = ['corrosion_depth_mm', 'applied_stress_mpa', 'gauge_stress_mpa', 'rho_A_pct', 'S_eq_mpa',
                  'log10N_clean', 'fatigue_life_cycles', 'log10_fatigue_life']
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
        print("Created: %s (%d entries, %d excluded at the HCF floor), noise sigma=%.3f (log10)"
              % (os.path.relpath(OUT_CSV, BASE_DIR), len(out), excluded, sigma_noise))
        if len(out) < 20:
            print("WARNING: only %d entries -- the ML results will be unstable." % len(out))
        fea_verification(fem)
        print("Next step: python 3_train_models.py  (training) or python 4_reanalysis.py  (independent)")
    finally:
        stop_report(fh)
    print("Report saved: Results/build_dataset_report.txt")


if __name__ == '__main__':
    if '--factor' in sys.argv:
        RHO_A_FACTOR = float(sys.argv[sys.argv.index('--factor') + 1])
        print("RHO_A_FACTOR override = %.4f" % RHO_A_FACTOR)
    if '--sensitivity' in sys.argv:
        fh = start_report()
        try:
            sensitivity()
        finally:
            stop_report(fh)
        print("Report saved: Results/build_dataset_report.txt")
    elif '--verify' in sys.argv:
        fh = start_report()
        try:
            calibrate()
        finally:
            stop_report(fh)
        print("Report saved: Results/build_dataset_report.txt")
    else:
        build()
