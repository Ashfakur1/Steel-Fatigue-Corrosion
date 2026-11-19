# -*- coding: utf-8 -*-
"""
1_fem_pipeline.py  (pipeline step 1; faithful copy of fem_pipeline.py + output-folder scheme)
=============================================================================================
One file, two modes -- it detects where it is running:

  1) Plain Python (terminal):           python 1_fem_pipeline.py
                                         python 1_fem_pipeline.py --mesh-study   # [Q4] mesh convergence
                                         python 1_fem_pipeline.py --figs-only    # without running FEA, only
                                            # redraw the FEM verification figure
       -> BATCH DRIVER: for every (corrosion, stress) pair it writes run_params.txt and runs
          `abaqus cae noGUI=1_fem_pipeline.py`. Resume support: jobs already present in
          Results/fem_results_raw.csv are skipped. After the batch / mesh study it draws
          Figures/fig_fem_verification.png/.pdf from the two CSVs in Results/.

  2) Inside Abaqus (the command above calls it itself): the "ABAQUS MODE" section below runs --
       the previously working FEA + extraction code unchanged; only parameter reading and the
       CSV path changed.

NOTE: why the earlier 1_fem_pipeline.py broke (lesson kept for the record):
   `abaqus cae noGUI=script.py` executes the script with exec(), and in that exec context the
   Python `__file__` variable does not exist -> NameError. Therefore the ABAQUS MODE section of
   this script never uses `__file__`; the driver (plain Python, where `__file__` exists) passes
   all paths to Abaqus mode through run_params.txt. A script-folder path containing spaces is
   not a problem either: the noGUI argument is only the file name (no spaces), and Abaqus always
   writes files in the CWD -- the earlier fem_pipeline.py ran exactly this way in this folder.

Output scheme (the only modification relative to fem_pipeline.py):
   - fem_results_raw.csv / fem_mesh_study.csv        ->  <script folder>/Results/
   - per-job .odb/.dat/.msg/.log/.lck/... and abaqus.rpy ->  <script folder>/Abaqus/
   - CSV and job files left in the root from the fem_pipeline.py era are moved/merged
     automatically on the first run -- nothing is deleted; completed runs of earlier eras
     are skipped on resume.

NOTE: no f-strings or type hints are used, for Python 2.7/Abaqus compatibility.
"""
from __future__ import print_function
try:
    from abaqus import *
    from abaqusConstants import *
    INSIDE_ABAQUS = True
except ImportError:
    INSIDE_ABAQUS = False

import sys
import os
import csv
import glob
import shutil
import itertools
import subprocess

# ---------------------------------------------------------------
# Shared config (visible to both modes; grid identical to fem_pipeline.py -- 21x6=126 runs)
# ---------------------------------------------------------------
# NOTE: the grid 0-2.0 mm (rho_A ~ 0-37.5%) stays inside the Liu Table 4 coverage (rho_A <= 41%).
# The earlier 1.0 mm limit existed because of the broken published equation; the law fitted from
# our own data does not need it, and at 0-1.0 mm the corrosion effect is very mild (~x0.75).
# Completed runs from earlier eras are skipped on resume.
CORROSION_DEPTHS_MM = [round(0.1 * i, 1) for i in range(0, 21)]
STRESS_AMPLITUDES_MPA = [290, 320, 350, 380, 410, 430]
PARAMS_FILE = 'run_params.txt'
RESULTS_BASENAME = 'fem_results_raw.csv'     # goes into the Results/ folder
MESH_BASENAME = 'fem_mesh_study.csv'         # goes into the Results/ folder
DEFAULT_MESH_MM = 1.0
MESH_CASES = [(0.0, 350), (1.0, 350), (2.0, 350)]    # (depth mm, stress MPa)
MESH_SIZES_MM = [2.0, 1.0, 0.5]


# ================================================================
# DRIVER MODE helpers (run only in plain Python)
# ================================================================
def ensure_dir(path):
    """Python 2.7-safe makedirs-if-missing"""
    if not os.path.isdir(path):
        os.makedirs(path)


ARTIFACT_EXTRA_PATTERNS = ('abaqus.rpy*', 'abaqus.rec*')


def harvest_artifacts(abaqus_dir, job_name=None):
    """Move Abaqus artifacts left in the script folder (launch CWD) into the Abaqus/ folder.
    With job_name: that job's files + abaqus.rpy; without: all job_* files + abaqus.rpy.
    Deletes nothing -- only moves (on the same name the older file is replaced, the new one kept)."""
    ensure_dir(abaqus_dir)
    if job_name:
        patterns = [job_name + '.*'] + list(ARTIFACT_EXTRA_PATTERNS)
    else:
        patterns = ['job_*.*'] + list(ARTIFACT_EXTRA_PATTERNS)
    moved = 0
    for pat in patterns:
        for fname in glob.glob(pat):
            if not os.path.isfile(fname):
                continue
            dest = os.path.join(abaqus_dir, fname)
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except OSError:
                    pass
            try:
                shutil.move(fname, abaqus_dir)
                moved += 1
            except (OSError, IOError) as e_mv:
                print("  (harvest move failed: %s -- %s)" % (fname, e_mv))
    return moved


def get_completed_job_names(results_file):
    if not os.path.exists(results_file):
        return set()
    done = set()
    with open(results_file, 'r') as f:
        for row in csv.DictReader(f):
            if row.get('job_name'):
                done.add(row['job_name'])
    return done


def merge_legacy_csv(src_path, dst_path):
    """Append the entries of an old root-level CSV into the Results/ CSV (dedupe by job_name).
    Returns: the number of entries added."""
    with open(src_path, 'r') as f:
        src_lines = [l for l in f.readlines() if l.strip()]
    with open(dst_path, 'r') as f:
        dst_lines = [l for l in f.readlines() if l.strip()]
    if not src_lines:
        return 0
    dst_jobs = set(l.split(',')[0].strip() for l in dst_lines[1:])
    added = 0
    with open(dst_path, 'a') as f:
        for line in src_lines[1:]:
            jname = line.split(',')[0].strip()
            if jname and jname not in dst_jobs:
                f.write(line if line.endswith('\n') else line + '\n')
                added += 1
    return added


def migrate_legacy_outputs(script_dir, results_dir, abaqus_dir):
    """Tidy up files from the fem_pipeline.py era (nothing is deleted):
    - root-level fem_results_raw.csv / fem_mesh_study.csv -> Results/ (with entry merge)
    - job_*.* and abaqus.rpy* scattered in the root -> Abaqus/"""
    ensure_dir(results_dir)
    ensure_dir(abaqus_dir)
    for base in (RESULTS_BASENAME, MESH_BASENAME):
        src = os.path.join(script_dir, base)
        dst = os.path.join(results_dir, base)
        if not os.path.isfile(src):
            continue
        if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
            if os.path.isfile(dst):
                os.remove(dst)
            shutil.move(src, dst)
            print("LEGACY: moved old %s -> Results/ (earlier runs will be skipped on resume)" % base)
        else:
            n = merge_legacy_csv(src, dst)
            archive = os.path.join(abaqus_dir, 'legacy_' + base)
            if os.path.exists(archive):
                try:
                    os.remove(archive)
                except OSError:
                    pass
            shutil.move(src, archive)
            if n:
                print("LEGACY: merged %d entries of old %s into Results/ "
                      "(original file archived as Abaqus/legacy_%s)" % (base, n, base))
            else:
                print("LEGACY: old %s contained nothing new (archived as Abaqus/legacy_%s)" % (base, base))
    n = harvest_artifacts(abaqus_dir)   # old job files scattered in the root also go to Abaqus/
    if n:
        print("LEGACY: moved %d old Abaqus files from the root into the Abaqus/ folder" % n)


def fea_sanity_summary(results_file):
    """Quick check of whether the FEA adds any real information: gauge mean stress / applied stress"""
    if not os.path.exists(results_file):
        return
    ratios = []
    with open(results_file, 'r') as f:
        for row in csv.DictReader(f):
            try:
                ratios.append(float(row['mean_principal_stress_mpa']) / float(row['applied_stress_mpa']))
            except (KeyError, ValueError, ZeroDivisionError):
                pass
    if ratios:
        print("FEA CHECK: gauge-mean / applied stress = %.4f .. %.4f (n=%d)" % (min(ratios), max(ratios), len(ratios)))
        print("  (a value ~1.000 means the pressure BC produces uniform stress -- depth has no")
        print("   effect on stress; the corrosion effect enters only through rho_A in the S-N law.")


def make_fem_figure():
    """Draw the fig_fem_verification figure directly from the two CSVs in Results/
    (fem_results_raw.csv + fem_mesh_study.csv) -- no separate script or style file needed.
    (a) stress-mapping grid: peak/applied principal stress vs corrosion depth (all batch jobs)
    (b) mesh refinement study: peak stress above nominal vs element size
    If matplotlib is missing, skip with a warning -- the FEA results stay untouched."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p_raw = os.path.join(script_dir, 'Results', RESULTS_BASENAME)
    p_mesh = os.path.join(script_dir, 'Results', MESH_BASENAME)
    if not (os.path.exists(p_raw) and os.path.exists(p_mesh)):
        print("Figure skipped: Results/%s or Results/%s missing -- the figure is drawn after the FEA runs"
              % (RESULTS_BASENAME, MESH_BASENAME))
        return
    raw, mesh = [], []
    with open(p_raw, 'r') as fh:
        for r in csv.DictReader(fh):
            try:
                raw.append(dict(depth=float(r['corrosion_depth_mm']), S=float(r['applied_stress_mpa']),
                                pk=float(r['max_principal_stress_mpa'])))
            except (KeyError, ValueError):
                pass
    with open(p_mesh, 'r') as fh:
        for r in csv.DictReader(fh):
            try:
                mesh.append(dict(depth=float(r['corrosion_depth_mm']), h=float(r['mesh_size_mm']),
                                 pk=float(r['max_principal_stress_mpa']), S=float(r['applied_stress_mpa'])))
            except (KeyError, ValueError):
                pass
    if not raw or not mesh:
        print("Figure skipped: FEA CSV empty")
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e_imp:
        print("matplotlib missing -- fig_fem_verification skipped (%s)" % e_imp)
        return
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
        'axes.linewidth': 0.8, 'lines.linewidth': 1.4, 'savefig.dpi': 300,
        'figure.constrained_layout.use': True, 'axes.unicode_minus': False,
    })
    fig_dir = os.path.join(script_dir, 'Figures')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
    try:
        fig.get_layout_engine().set(wspace=0.30)
    except Exception:
        pass

    # (a) stress-mapping grid: peak/applied vs uniform corrosion depth (all batch jobs)
    # band = max deviation over all FEA verification jobs (grid + mesh refinement)
    ratios = [r['pk'] / r['S'] for r in raw if r['S']]
    ratios_m = [r['pk'] / r['S'] for r in mesh if r['S']]
    dev = max(abs(v - 1.0) for v in ratios + ratios_m)
    ax = axs[0]
    for S, col in ((430.0, '#e68a2e'), (290.0, '#1f3a93')):   # 290 drawn last -- stays visible on top
        pts = sorted((r['depth'], r['pk'] / r['S']) for r in raw if abs(r['S'] - S) < 1e-6)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.6, color=col,
                    solid_capstyle='round', label='%d MPa' % int(S))
    ax.axhline(1.0, ls='--', color='0.5', lw=0.9)
    ax.text(0.5, 0.94, 'band: \u00b1%.2f%%' % (100 * dev), transform=ax.transAxes,
            ha='center', va='top', fontsize=9)
    ax.set_ylim(1.0 - 2.2 * dev, 1.0 + 3.8 * dev)
    ax.set_xlabel('Uniform corrosion depth (mm)')
    ax.set_ylabel('Peak / applied principal stress')
    ax.set_title('(a) Stress-mapping grid (126 jobs)', fontweight='bold', pad=5)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9, borderaxespad=0.3)

    # (b) mesh refinement: peak stress above nominal vs element size
    ax = axs[1]
    cols = ('#1f77b4', '#ff7f0e', '#2ca02c')
    ys_b = []
    for i, dep in enumerate(sorted(set(r['depth'] for r in mesh))):
        pts = sorted(((r['h'], r['pk'] - r['S']) for r in mesh if r['depth'] == dep), reverse=True)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', ms=4.5, lw=1.4,
                color=cols[i % 3], label='depth %g mm' % dep)
        ys_b.extend(p[1] for p in pts)
    ax.axhline(0.0, ls='--', color='0.5', lw=0.9)
    if ys_b:
        span = max(ys_b) - 0.0
        ax.set_ylim(-0.10 * span, 1.16 * max(ys_b))
    ax.invert_xaxis()
    ax.set_xlabel('Element size (mm)')
    ax.set_ylabel('Peak stress above nominal (MPa)')
    ax.set_title('(b) Mesh refinement study', fontweight='bold', pad=5)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9, borderaxespad=0.3, handletextpad=0.4)

    out = []
    for ext in ('.png', '.pdf'):
        p = os.path.join(fig_dir, 'fig_fem_verification' + ext)
        fig.savefig(p)
        out.append(os.path.basename(p))
    plt.close(fig)
    print("Figure saved: %s" % ", ".join(out))


def run_batch():
    # --- paths: everything relative to the script folder (no matter where python is launched) ---
    # __file__ is safe only here (plain Python) -- ABAQUS MODE has no __file__.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    results_dir = os.path.join(script_dir, 'Results')
    abaqus_dir = os.path.join(script_dir, 'Abaqus')
    results_file = os.path.join(results_dir, RESULTS_BASENAME)

    print("Folder scheme: CSV -> %s" % results_file)
    print("               Abaqus artifacts -> %s" % abaqus_dir)
    migrate_legacy_outputs(script_dir, results_dir, abaqus_dir)

    completed = get_completed_job_names(results_file)
    if completed:
        print("Found %d already-completed jobs in Results/%s -- these will be skipped." % (len(completed), RESULTS_BASENAME))
    total = len(CORROSION_DEPTHS_MM) * len(STRESS_AMPLITUDES_MPA)
    print("Total planned runs: %d" % total)
    skipped, failed, count = 0, [], 0
    for depth, stress in itertools.product(CORROSION_DEPTHS_MM, STRESS_AMPLITUDES_MPA):
        count += 1
        job_name = "job_c%s_s%s" % (str(depth).replace('.', 'p'), str(stress))
        if job_name in completed:
            skipped += 1
            print("[%d/%d] SKIP (already done): %s" % (count, total, job_name))
            continue
        print("[%d/%d] RUNNING: corrosion=%.2fmm, stress=%dMPa (%s)" % (count, total, depth, stress, job_name))
        # run_params.txt (6 lines): depth, stress, job_name, mesh_size, Results folder, mesh-study flag
        # Abaqus mode cannot use __file__, so the absolute Results-folder path is passed from here.
        with open(PARAMS_FILE, 'w') as f:
            f.write("%s\n%s\n%s\n%s\n%s\n0\n" % (depth, stress, job_name, DEFAULT_MESH_MM, results_dir))
        try:
            subprocess.check_call("abaqus cae noGUI=%s" % os.path.basename(__file__), shell=True)
        except subprocess.CalledProcessError as e:
            print("  !! Abaqus error (returncode %s): %s" % (getattr(e, 'returncode', '?'), job_name))
        except OSError as e_cmd:
            print("  !! could not run the 'abaqus' command: %s" % e_cmd)
            print("     run from the Abaqus command window, or check that abaqus.bat is on PATH.")
            return
        # all files of this job (.odb/.dat/.msg/.log/.lck/...) + abaqus.rpy -> Abaqus/ folder
        harvest_artifacts(abaqus_dir, job_name)
        if job_name not in get_completed_job_names(results_file):
            failed.append(job_name)
            print("  !! FAILED: %s -- result entry not found in Results/%s." % (job_name, RESULTS_BASENAME))
            print("     To find the cause: Abaqus/%s.msg, Abaqus/%s.log (solver error)," % (job_name, job_name))
            print("     or the last lines of Abaqus/abaqus.rpy (Python/script error).")
    print("\n===== BATCH COMPLETE =====")
    print("Total planned : %d | Skipped (resume): %d | Failed: %d" % (total, skipped, len(failed)))
    if failed:
        print("Failed job names: %r" % (failed,))
        print("(running `python 1_fem_pipeline.py` again will re-run only these)")
    fea_sanity_summary(results_file)
    print("Next step: python 2_build_dataset.py   (the CSV is now in Results/%s)" % RESULTS_BASENAME)


def run_mesh_study():
    """[Q4] mesh convergence: a few cases at three mesh sizes. Results go to Results/fem_mesh_study.csv (training data untouched)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    results_dir = os.path.join(script_dir, 'Results')
    abaqus_dir = os.path.join(script_dir, 'Abaqus')
    mesh_file = os.path.join(results_dir, MESH_BASENAME)

    print("Folder scheme: CSV -> %s | Abaqus artifacts -> %s" % (mesh_file, abaqus_dir))
    migrate_legacy_outputs(script_dir, results_dir, abaqus_dir)
    done = get_completed_job_names(mesh_file)
    for (depth, stress), msize in itertools.product(MESH_CASES, MESH_SIZES_MM):
        job_name = "mesh_c%s_s%s_m%s" % (str(depth).replace('.', 'p'), str(stress), str(msize).replace('.', 'p'))
        if job_name in done:
            print("SKIP (already done): %s" % job_name)
            continue
        print("MESH RUN: depth=%.1f stress=%d mesh=%.2f (%s)" % (depth, stress, msize, job_name))
        with open(PARAMS_FILE, 'w') as f:
            f.write("%s\n%s\n%s\n%s\n%s\n1\n" % (depth, stress, job_name, msize, results_dir))
        try:
            subprocess.check_call("abaqus cae noGUI=%s" % os.path.basename(__file__), shell=True)
        except subprocess.CalledProcessError as e:
            print("  !! Abaqus error (returncode %s): %s" % (getattr(e, 'returncode', '?'), job_name))
        except OSError as e_cmd:
            print("  !! could not run the 'abaqus' command: %s" % e_cmd)
            return
        harvest_artifacts(abaqus_dir, job_name)
        if job_name not in get_completed_job_names(mesh_file):
            print("  !! FAILED: %s -- see Abaqus/%s.msg / Abaqus/abaqus.rpy" % (job_name, job_name))
    print("Mesh study results: Results/%s  (next: python 2_build_dataset.py -- shown in its report)" % MESH_BASENAME)


if not INSIDE_ABAQUS:
    if '--figs-only' in sys.argv:
        make_fem_figure()
    elif '--mesh-study' in sys.argv:
        run_mesh_study()
        make_fem_figure()
    else:
        run_batch()
        make_fem_figure()
    sys.exit(0)

# =====================================================================
# ABAQUS MODE -- all code below this point runs only inside Abaqus
#
# NOTE: `__file__` must not be used in this section: `abaqus cae noGUI=...` executes the
#    script with exec(), and in that context `__file__` does not exist (the earlier version
#    hit exactly this NameError). All paths come from run_params.txt (6 lines), which the
#    driver writes before each job.
# =====================================================================
import regionToolset

# =============== configuration ===============
USE_PLASTICITY = False   # set True to add the bilinear hardening of the Q550 steel
# ===========================================

# ---------------------------------------------------------------
# 1. Read the parameter file (6-line format -- the driver writes it before each job)
#    line 1: corrosion depth (mm)   line 2: stress amplitude (MPa)
#    line 3: job name               line 4: mesh size (mm)
#    line 5: absolute path of the Results folder (the CSV must be written there)
#    line 6: mesh-study flag ('1' = mesh study, '0' = normal batch)
# ---------------------------------------------------------------
PARAMS_FILE = 'run_params.txt'

if not os.path.exists(PARAMS_FILE):
    raise RuntimeError(
        "\n\n"
        "=====================================================\n"
        "Could not find '%s' in this folder: %s\n\n"
        "Do not run this script directly -- run it through the driver:\n"
        "  python 1_fem_pipeline.py\n"
        "(the driver writes run_params.txt before every job)\n"
        "=====================================================\n" % (PARAMS_FILE, os.getcwd())
    )

with open(PARAMS_FILE, 'r') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

if len(lines) < 6:
    raise RuntimeError(
        "'%s' needs 6 lines (depth, stress, job_name, mesh_size, results_dir, "
        "mesh_flag), found %d lines: %r\n"
        "Most likely a run_params.txt in the old format is sitting here -- running "
        "`python 1_fem_pipeline.py` again makes the driver write a fresh file." % (PARAMS_FILE, len(lines), lines)
    )

try:
    corrosion_depth_mm = float(lines[0])
    stress_amplitude_mpa = float(lines[1])
    job_name = lines[2]
    mesh_size = float(lines[3])
    results_dir_param = lines[4]
    mesh_study = (lines[5] == '1')
except ValueError:
    raise RuntimeError(
        "lines 1, 2 and 4 must be numbers; found: %r" % (lines[:5],)
    )

# CSV destination: the Results/ folder (absolute path sent by the driver)
if mesh_study:
    results_file = os.path.join(results_dir_param, MESH_BASENAME)
else:
    results_file = os.path.join(results_dir_param, RESULTS_BASENAME)
if not os.path.isdir(results_dir_param):
    os.makedirs(results_dir_param)

print("PARSED PARAMS: corrosion_depth=%.3f, stress=%.3f, job_name=%s, mesh=%.2f, mesh_study=%s" % (
    corrosion_depth_mm, stress_amplitude_mpa, job_name, mesh_size, mesh_study))
print("RESULT FILE: %s" % results_file)

# ---------------------------------------------------------------
# 2. Geometry
# ---------------------------------------------------------------
ORIGINAL_THICKNESS = 10.0
LENGTH = 100.0
WIDTH = 20.0

effective_thickness = ORIGINAL_THICKNESS - corrosion_depth_mm
if effective_thickness <= 0:
    raise ValueError("corrosion_depth > plate thickness")

model_name = 'CorrosionFatigueModel'
mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)
myModel = mdb.models[model_name]

# --- sketch + part ---
s = myModel.ConstrainedSketch(name='plate_sketch', sheetSize=200.0)
s.rectangle(point1=(0.0, 0.0), point2=(LENGTH, WIDTH))
part = myModel.Part(name='CorrodedPlate', dimensionality=THREE_D, type=DEFORMABLE_BODY)
part.BaseSolidExtrude(sketch=s, depth=effective_thickness)

# ---------------------------------------------------------------
# 3. Material (with / without plasticity)
#
# NOTE correction: the material data of Wang et al. (2025) was used before, but the
# fatigue-life model (compute_fatigue_life.py) uses the Q550E steel of Liu et al. (2021)
# -- two different papers/grades were being mixed. Now everything is taken consistently
# from the same source (Liu et al. 2021).
#
# Source: Liu, X., Xiao, L., Cai, C.S., Zhang, J., Wang, L. (2021). "Fatigue
# Properties Investigation of Corroded High-Performance Steel Specimens."
# Journal of Materials in Civil Engineering, 33(1), 04020410. Table 3
# ("Mechanical properties"), Q550E uncorroded reference plate:
#   Yield (f_y) = 639.0 MPa (engineering), UTS (f_u) = 696.8 MPa (engineering)
#   DOI: 10.1061/(ASCE)MT.1943-5533.0003508
#
# @article{liu2021fatigue,
#   author = {Liu, Xinhua and Xiao, Linfa and Cai, C. S. and Zhang, Jianren and Wang, Lei},
#   title = {Fatigue Properties Investigation of Corroded High-Performance Steel Specimens},
#   journal = {Journal of Materials in Civil Engineering},
#   volume = {33}, number = {1}, pages = {04020410}, year = {2021},
#   doi = {10.1061/(ASCE)MT.1943-5533.0003508}
# }
#
# NOTE UNVERIFIED PLACEHOLDER: the elongation (A_gt) data of Liu et al. (2021) is not
# available in the information extracted so far -- only yield/UTS were obtained. The
# plastic-strain point of the hardening curve is therefore still a placeholder
# (elongation=15% assumed, a typical range for HSLA steels, not Liu's own value).
# Before setting USE_PLASTICITY=True, replace this value with the actual elongation
# extracted from the Liu et al. paper.
#
# Engineering -> TRUE stress/plastic-strain conversion:
#   true_stress = eng_stress * (1 + eng_strain)
#   true_plastic_strain = ln(1+eng_strain) - true_stress/E
#   At the yield point the strain is negligible (~0.3%), so true ~= engineering (639.0->640.9, a negligible difference)
#   At the UTS point (placeholder elong=15%): true_stress=801.32 MPa, true_plastic_strain=0.1359
#
# E = 210000 MPa, nu = 0.3 -- verified by convention (universal constants for structural
#   steels, 200-210 GPa range, valid across grades)
#
# Limitation (documentation): this script runs a static
# single-load-step FEA -- it does not simulate cyclic fatigue; it extracts the static
# peak/mean stress and feeds an empirical S-N (Basquin) equation outside (stress-life approach).
#
# Limitation (documentation): no convergence study at mesh size=1.0 mm was carried out --
# a separate mesh-refinement sensitivity run is recommended before use.
# ---------------------------------------------------------------
mat = myModel.Material(name='Q550_Steel')
mat.Elastic(table=((210000.0, 0.3), ))
if USE_PLASTICITY:
    mat.Plastic(table=((640.94, 0.0), (801.32, 0.1359)))  # true stress/plastic strain; UTS-strain point is placeholder, see note above

myModel.HomogeneousSolidSection(name='PlateSection', material='Q550_Steel', thickness=None)
part.SectionAssignment(region=(part.cells,), sectionName='PlateSection')

# ---------------------------------------------------------------
# 4. Assembly, mesh, and gauge section set
# ---------------------------------------------------------------
a = myModel.rootAssembly
inst = a.Instance(name='PlateInstance', part=part, dependent=ON)

part.seedPart(size=mesh_size, deviationFactor=0.1)
part.generateMesh()

a.regenerate()

all_elements = inst.elements
print("DIAGNOSTIC: total elements in instance = %d" % len(all_elements))

all_node_x = [n.coordinates[0] for n in inst.nodes]
if all_node_x:
    print("DIAGNOSTIC: instance node X range = %.3f to %.3f" % (min(all_node_x), max(all_node_x)))
else:
    print("DIAGNOSTIC: inst.nodes is EMPTY -- mesh/instance problem")

GAUGE_X_MIN = LENGTH * 0.25
GAUGE_X_MAX = LENGTH * 0.75

gauge_elements = inst.elements.getByBoundingBox(
    xMin=GAUGE_X_MIN, xMax=GAUGE_X_MAX,
    yMin=-1, yMax=WIDTH + 1,
    zMin=-1, zMax=effective_thickness + 1
)
print("DIAGNOSTIC: gauge_elements found via inst.elements = %d" % len(gauge_elements))

if len(gauge_elements) == 0:
    print("DIAGNOSTIC: retrying with part.elements instead of inst.elements ...")
    gauge_elements = part.elements.getByBoundingBox(
        xMin=GAUGE_X_MIN, xMax=GAUGE_X_MAX,
        yMin=-1, yMax=WIDTH + 1,
        zMin=-1, zMax=effective_thickness + 1
    )
    print("DIAGNOSTIC: gauge_elements found via part.elements = %d" % len(gauge_elements))

if len(gauge_elements) == 0:
    raise RuntimeError(
        "No element found in the gauge region (X range: %.2f to %.2f), "
        "for either the instance or the part. See the DIAGNOSTIC lines above."
        % (GAUGE_X_MIN, GAUGE_X_MAX)
    )

a.Set(elements=gauge_elements, name='GaugeSet')
print("Gauge set created with %d elements (X: %.1f to %.1f)" % (
    len(gauge_elements), GAUGE_X_MIN, GAUGE_X_MAX))

# ---------------------------------------------------------------
# 5. Step, boundary conditions, load
# ---------------------------------------------------------------
myModel.StaticStep(name='LoadStep', previous='Initial',
                   initialInc=0.1, minInc=1e-08, maxInc=0.1)

fixed_faces = inst.faces.getByBoundingBox(xMin=-0.01, xMax=0.01,
                                           yMin=-1, yMax=WIDTH + 1,
                                           zMin=-1, zMax=effective_thickness + 1)
loaded_faces = inst.faces.getByBoundingBox(xMin=LENGTH - 0.01, xMax=LENGTH + 0.01,
                                            yMin=-1, yMax=WIDTH + 1,
                                            zMin=-1, zMax=effective_thickness + 1)

# --- NEW: Create a node set from fixed_faces for diagnostic reaction-force check ---
a.Set(faces=fixed_faces, name='FixedFaceSet')

myModel.EncastreBC(name='FixedEnd', createStepName='Initial',
                    region=regionToolset.Region(faces=fixed_faces))

a.Surface(name='LoadSurf', side1Faces=loaded_faces)
myModel.Pressure(name='AxialLoad', createStepName='LoadStep',
                  region=a.surfaces['LoadSurf'],
                  magnitude=-stress_amplitude_mpa)

# ---------------------------------------------------------------
# 6. Job creation, submission and monitoring
# ---------------------------------------------------------------
myModel.rootAssembly.regenerate()

job = mdb.Job(name=job_name, model=model_name, description='')
job.submit()
job.waitForCompletion()

# ---------------------------------------------------------------
# 7. Extract results (from the gauge section, retrieving the Set from the ODB)
# ---------------------------------------------------------------
odb_path = job_name + '.odb'
if not os.path.exists(odb_path):
    # NOTE correction: previously glob('*.odb')[0] opened any ODB -- results of a wrong job
    # could silently slip in. Now only this job's own ODB is accepted.
    msg_file = job_name + '.msg'
    if os.path.exists(msg_file):
        with open(msg_file, 'r') as f:
            msg_content = f.read()
        if "THE ANALYSIS HAS BEEN COMPLETED" in msg_content:
            raise RuntimeError("Analysis completed but ODB '%s' not found. "
                               "Check field output settings." % odb_path)
        else:
            raise RuntimeError("Job failed. Last lines from .msg:\n%s" % msg_content[-2000:])
    else:
        raise RuntimeError("ODB file '%s' not found and no .msg file exists." % odb_path)

from odbAccess import openOdb
odb = openOdb(path=odb_path)
step = odb.steps['LoadStep']

if not step.frames:
    raise RuntimeError("No frames in step 'LoadStep'.")

last_frame = step.frames[-1]
stress_field = last_frame.fieldOutputs['S']

try:
    gauge_set = odb.rootAssembly.elementSets['GAUGESET']
except KeyError:
    raise RuntimeError(
        "'GaugeSet' not found in the ODB. Available element sets: %r" %
        (list(odb.rootAssembly.elementSets.keys()),)
    )

gauge_stress = stress_field.getSubset(region=gauge_set)
stress_values = [v.maxPrincipal for v in gauge_stress.values]

if not stress_values:
    raise RuntimeError("No stress values found in the gauge set.")

max_stress = max(stress_values)
mean_stress = sum(stress_values) / len(stress_values)

print("GAUGE-SECTION STRESS: max=%.2f MPa, mean=%.2f MPa (n=%d values)" % (
    max_stress, mean_stress, len(stress_values)))

# --- NEW DIAGNOSTIC BLOCK: reaction force check (before odb.close) ---
try:
    rf_field = last_frame.fieldOutputs['RF']
    fixed_node_set = odb.rootAssembly.nodeSets['FIXEDFACESET']
    rf_subset = rf_field.getSubset(region=fixed_node_set)
    total_rf1 = sum(v.data[0] for v in rf_subset.values)  # axial (X) direction
    loaded_area_actual = WIDTH * effective_thickness
    implied_stress_from_force = abs(total_rf1) / loaded_area_actual

    print("DIAGNOSTIC: total axial reaction force = %.3f N" % total_rf1)
    print("DIAGNOSTIC: applied_stress = %.3f | stress-from-RF/actual-area = %.3f" % (
        stress_amplitude_mpa, implied_stress_from_force))
except Exception as e:
    print("DIAGNOSTIC: reaction force check failed: %s" % e)

odb.close()

# ---------------------------------------------------------------
# 8. Write the results to the output file (Results/ folder -- path sent by the driver)
# ---------------------------------------------------------------
if mesh_study:
    # mesh-study results go to a separate file (the training data stays clean)
    header = ("job_name,corrosion_depth_mm,applied_stress_mpa,mesh_size_mm,n_elements,"
              "max_principal_stress_mpa,mean_principal_stress_mpa\n")
    result_line = "%s,%.4f,%.4f,%.4f,%d,%.4f,%.4f\n" % (
        job_name, corrosion_depth_mm, stress_amplitude_mpa, mesh_size, len(all_elements), max_stress, mean_stress)
else:
    header = ("job_name,corrosion_depth_mm,applied_stress_mpa,effective_thickness_mm,"
              "max_principal_stress_mpa,mean_principal_stress_mpa\n")
    result_line = "%s,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (
        job_name, corrosion_depth_mm, stress_amplitude_mpa, effective_thickness, max_stress, mean_stress)

write_header = not os.path.exists(results_file)
with open(results_file, 'a') as f:
    if write_header:
        f.write(header)
    f.write(result_line)

print("DONE: %s -> max=%.2f MPa, mean=%.2f MPa (written to %s)" % (
    job_name, max_stress, mean_stress, results_file))
