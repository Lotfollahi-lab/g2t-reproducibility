#!/usr/bin/env bash
# Set up an isolated uv environment for the COME baseline, and clone the authors'
# code next to the other benchmarking assets.
#
# COME ("COntrastive MApping lEarning"; Wei, Chen, Wang, Shen, Liu, Wu, Wong,
# Bioinformatics 41(3):btaf083, 2025, doi:10.1093/bioinformatics/btaf083) is the
# second reviewer-requested contrastive baseline, alongside CellContrast. We run
# THEIR published code rather than a reimplementation, so this script installs
# dependencies and changes nothing about the method.
#
#   Upstream: https://github.com/cindyway/COME
#   Licence : NONE — the repo ships no LICENSE file and GitHub reports no licence
#             (checked 2026-08-14). We only import it; we redistribute nothing.
#             Flag this to the lead before any code from it is vendored.
#
# Why a dedicated venv (and NOT the cellcontrast one):
#  * Upstream ships NO requirements.txt, NO environment.yml and NO setup.py. Its
#    README asks for "Python 3.7+ / PyTorch 1.6+ / Other dependencies" and then
#    `conda create python=3.7 -n COME`. That is not a specification, so every pin
#    below is OURS and is justified where it is set.
#  * The CellContrast env is pinned to python 3.9 + scanpy==1.9.3 + numpy<2
#    BECAUSE THAT UPSTREAM DEMANDS IT (its environment.yml pins scanpy 1.9.3,
#    which uses the removed np.float_). COME demands none of that: it calls only
#    sc.pp.normalize_total / log1p / filter_genes / filter_cells / sc.read_h5ad,
#    all stable for years, so we are free to use a supported scanpy. Do not
#    "unify" the two envs — the pins differ for real reasons and one shared env
#    would have to satisfy the older, stricter set.
#
# Dependencies were ENUMERATED FROM THE SOURCE at the pinned commit, not guessed:
#   grep -hE '^[[:space:]]*(import|from) ' *.py
# third-party top-level packages: torch, scanpy, numpy, pandas, scipy, sklearn,
# matplotlib, seaborn.  (anndata is added explicitly: our wrapper constructs
# AnnData objects directly, and scanpy only brings it in transitively.)
#
#   ** seaborn is NOT optional. ** train_eval.py:16 does
#   `from GenesMetrics import count`, and GenesMetrics.py:9-10 imports seaborn
#   and matplotlib.pyplot AT MODULE LEVEL. run_come.py imports train_eval (for
#   its pretrain_ae/train loops), so a missing seaborn breaks the baseline at
#   import — inside the LSF job, after the queue wait. scanpy>=1.10 happens to
#   depend on seaborn too, but we install it by name so the env does not depend
#   on that staying true.
#
# Defaults (override via env vars):
#   VENV_DIR        = /nfs/team361/sb75/.venvs/come        uv-managed venv
#   REPO_DIR        = <this dir>/COME                      authors' code
#   REPO_COMMIT     = 2005812810f3...  (upstream main, pinned — see below)
#   PYTHON_VERSION  = 3.10    (see the note where it is set)
#   TORCH_MAX       = <2.6    (cu121; a CEILING not an exact pin -- see below)
#   SCANPY_VERSION  = 1.10.4
#   NUMPY_SPEC      = numpy>=1.26,<2
#   LOCKFILE        = <VENV_DIR>/requirements.lock         resolved pins
#
# The script ends by (a) running COME itself for two epochs on a 12x9 toy problem
# inside the new venv and (b) running the wrapper's numpy-only test suite, and
# FAILS if either fails — an env that merely imports is not evidence the baseline
# computes anything.
#
# Usage:
#   bash scgg-reproducibility/analysis/benchmarking/setup_come_env.sh
#
# Rebuild from scratch:
#   rm -rf /nfs/team361/sb75/.venvs/come
#   bash scgg-reproducibility/analysis/benchmarking/setup_come_env.sh
#
# Env only (skip the clone):
#   SKIP_CLONE=1 bash .../setup_come_env.sh
#
# Skip the post-install checks (not recommended):
#   SKIP_FUNCTIONAL=1 SKIP_TESTS=1 bash .../setup_come_env.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/come}"
REPO_DIR="${REPO_DIR:-$HERE/COME}"
REPO_URL="${REPO_URL:-https://github.com/cindyway/COME.git}"
# Pin the upstream revision. Upstream has NO tags and NO releases, so a bare
# clone tracks main and "the COME baseline" would quietly mean different code on
# every re-run.
#
# This is the tip of main as of 2026-08-14 (resolved via
#   curl -s https://api.github.com/repos/cindyway/COME/commits/main
# -> 2005812810f3ffa4407c7d549ba39464c0348316, "Delete data/test.txt", authored
# 2024-07-12). main has not moved since 2024-07-12, i.e. this is also the state
# of the code as published with the 2025 Bioinformatics paper. The blob shas of
# model.py / utils.py / train_eval.py / configure.py / datasets.py /
# GenesMetrics.py at this commit were verified byte-identical to the read-only
# reference copy every claim in run_come.py's docstring was derived from.
REPO_COMMIT="${REPO_COMMIT:-2005812810f3ffa4407c7d549ba39464c0348316}"
# Python: upstream says "3.7+", but 3.7 has been EOL since 2023-06 and no current
# scanpy/torch wheel exists for it, so its floor carries no information. 3.10 is
# chosen because (a) it is what setup_scgg_env.sh / setup_novosparc_env.sh use,
# so the cluster's toolchain is known to provide it; (b) cu121 torch wheels exist
# for cp310 (verified against the PyTorch index, 2026-08-14); (c) COME's source
# uses no syntax past 3.6 (no walrus, no match, no PEP-604 unions — grepped), so
# nothing is lost; and (d) it usefully FREEZES the resolve: today's scanpy 1.12 /
# numpy 2.5 / scipy 1.18 all declare requires-python>=3.12, so they are excluded
# by construction rather than by a pin we would have to keep updating.
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# scanpy: any 1.10/1.11 works for COME's five calls; 1.10.4 is the last of the
# 1.10 line, is the release contemporaneous with the torch build below, and
# supports python 3.10 (1.12+ requires 3.12). Pinned exactly so the preprocessing
# COME applies (normalize_total/log1p on sttype=sequence) cannot change under us.
SCANPY_VERSION="${SCANPY_VERSION:-1.10.4}"
# numpy. NOTE: unlike the CellContrast env, this <2 bound is NOT a compatibility
# fix for the upstream code. COME's source is numpy-2 clean: it uses none of the
# aliases numpy 2.0 removed (checked with
#   grep -nE 'np\.(float_|int_|bool8|object_|str_|NaN|Inf|alltrue|product|in1d|trapz)' *.py
# -> no hits), so COME itself would run on numpy 2.
# The bound is there because nothing else in this env pins numpy: torch's wheels
# declare no numpy dependency at all, and the only real ceiling comes from numba
# (a hard scanpy dep), which caps numpy per release — so an unbounded resolve is
# decided by whichever numba uv happens to pick, i.e. it can resolve today and
# fail next month. numpy 1.26.4 is the last 1.x, is accepted by every numba that
# supports py3.10 and by scanpy 1.10.x, and sidesteps the numpy-2 ABI break
# entirely for a stack that is almost all compiled extensions (torch, numba,
# scipy, h5py) and that we cannot rebuild-and-test on demand.
# To move to numpy 2 later: NUMPY_SPEC='numpy>=2,<3' and re-run — the
# verification below will tell you whether the binary deps agree.
NUMPY_SPEC="${NUMPY_SPEC:-numpy>=1.26,<2}"
# torch. Upstream pins nothing beyond "1.6+" and its README tells you to pick a
# CUDA build from pytorch.org, so the choice is ours.
#
# ON THE torch>=2.6 weights_only FLIP — checked in COME's source, not assumed:
#   train_eval.py:167  torch.save(model.state_dict(), model_path)   # pretrain
#   train_eval.py:197  torch.save(model.state_dict(), model_path)   # train
#   train_eval.py:109  model.ae.load_state_dict(torch.load(pretrain_path))
# Everything COME writes and reads back is a plain state_dict — an OrderedDict of
# tensors, which torch.load accepts under weights_only=True. run_come.py does the
# same. So COME is NOT exposed to the 2.6 default change; that ceiling was
# CellContrast's problem (it pickles a pandas Index into its checkpoint and dies
# at inference), and copying its `<2.6` rationale here would be wrong.
#
# WHY A CEILING AND NOT AN EXACT PIN — this script previously pinned
# `torch==2.5.1` and was UNSATISFIABLE. torch 2.5.1+cu121 declares
# `nvidia-cudnn-cu12==9.1.0.70`, and that exact cuDNN build is NOT on the cu121
# index (verified 2026-08-14: it carries 9.0.0.312 and 9.1.1.17, skipping
# 9.1.0.70, which exists only on PyPI). Because `--index-url` REPLACES PyPI
# rather than adding to it, the dependency cannot be found and uv reports
# "no solution found ... torch==2.5.1+cu121 cannot be used".
# A CEILING fixes it the way setup_cellcontrast_env.sh already does: the resolver
# is free to back off to a build whose whole dependency closure IS on that index
# (the cu121 index carries 2.1.0 ... 2.5.1, so 2.5.0 / 2.4.1 are available).
# We keep `<2.6` purely to match the CellContrast env and to stay on the cu121
# line the rest of this project is validated against -- NOT for weights_only,
# which as established above does not apply to COME. The functional check below
# exercises the actual save/load round trip on whatever torch is resolved, so a
# future bump is verified rather than argued about, and the lockfile records the
# build that was actually used.
TORCH_MAX="${TORCH_MAX:-<2.6}"
TORCH_SPEC="${TORCH_SPEC:-torch$TORCH_MAX --index-url https://download.pytorch.org/whl/cu121}"
SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_FUNCTIONAL="${SKIP_FUNCTIONAL:-0}"

# The uv cache and /nfs are on different filesystems here, so hardlinking is
# unavailable; say so up front instead of emitting a warning per install.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

log() { printf '[setup_come_env] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Pre-flight: uv must be on PATH (conda is NOT used on this cluster)
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
ERROR: uv is not installed.
Install it once via:
    curl -LsSf https://astral.sh/uv/install.sh | sh
Then re-run this script.
EOF
    exit 1
fi
log "uv $(uv --version)"

# ---------------------------------------------------------------------------
# 1. Clone / refresh the authors' code
# ---------------------------------------------------------------------------
if [[ "$SKIP_CLONE" != "1" ]]; then
    if [[ -d "$REPO_DIR/.git" ]]; then
        log "repo exists: $REPO_DIR (fetching)"
        git -C "$REPO_DIR" fetch --all --tags --quiet
        # Refuse to check out over local edits: the pinned sha is the whole
        # reproducibility claim, and a dirty tree makes .checked_out_commit a lie
        # about the code that actually ran. Our own droppings in the clone
        # (.checked_out_commit, and the __pycache__/*.pyc CPython writes next to
        # the modules — section 4 below imports them with cwd=$REPO_DIR, and so
        # does every real run through run_come.py) are expected; nothing else is.
        # Without that exemption the SECOND run of this script would abort on its
        # own bytecode and tell the user a human had edited the baseline.
        # COME's own code also WRITES INTO ITS CHECKOUT at run time —
        # utils.py:95 savemat's into data/, train_eval.py torch.save's into
        # pretrain/ and result/ — but run_come.py redirects all of that into a
        # private per-fit work dir (fit_dir), so anything else reported here
        # means a human edited the baseline.
        # (Two steps so that a failing `git status` aborts under set -e instead
        # of being read as "clean" by the filter below.)
        STATUS="$(git -C "$REPO_DIR" status --porcelain)"
        DIRTY=""
        if [[ -n "$STATUS" ]]; then
            DIRTY="$(printf '%s\n' "$STATUS" \
                     | grep -Ev '^\?\? (\.checked_out_commit|([^ ]*/)?__pycache__/|[^ ]*\.pyc)$' \
                     || true)"
        fi
        if [[ -n "$DIRTY" ]]; then
            log "ERROR: working tree at $REPO_DIR has local changes:"
            printf '%s\n' "$DIRTY" >&2
            cat >&2 <<EOF
Not checking out $REPO_COMMIT on top of them. Either discard the changes
    git -C $REPO_DIR checkout -- . && git -C $REPO_DIR clean -fd
and re-run, or keep them on purpose and re-run with
    SKIP_CLONE=1 bash $0
which leaves the tree untouched and skips the pinned checkout (the env is then
NOT reproducible from the recorded commit — say so in the methods).
EOF
            exit 1
        fi
    else
        log "cloning $REPO_URL -> $REPO_DIR"
        mkdir -p "$(dirname "$REPO_DIR")"
        git clone --quiet "$REPO_URL" "$REPO_DIR"
    fi
    # Fetching and RECORDING HEAD is not enough: without a checkout the code on
    # disk is whatever main pointed at on the day of the run.
    if [[ ! "$REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
        log "WARNING: REPO_COMMIT='$REPO_COMMIT' is not a full 40-char sha."
        log "         Branches and tags move; this run may not be reproducible."
    fi
    log "checking out pinned revision $REPO_COMMIT (detached HEAD) ..."
    if ! git -C "$REPO_DIR" checkout --detach --quiet "$REPO_COMMIT"; then
        log "ERROR: cannot check out '$REPO_COMMIT' in $REPO_DIR."
        log "  If upstream rewrote history, pick a revision from"
        log "  https://github.com/cindyway/COME/commits and re-run with"
        log "  REPO_COMMIT=<sha> bash $0   (then update the default in this file)"
        exit 1
    fi
    COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
    log "commit: $COMMIT"
    # A baseline we did not author must be reproducible to a revision.
    printf '%s\n' "$COMMIT" > "$REPO_DIR/.checked_out_commit"
else
    COMMIT="$(cat "$REPO_DIR/.checked_out_commit" 2>/dev/null || echo unknown)"
    log "SKIP_CLONE=1 (using existing $REPO_DIR, commit $COMMIT)"
fi

# Every module run_come.py imports, plus the two it pulls in transitively.
# datasets.py and GenesMetrics.py are NOT dead weight: train_eval.py:9,16 import
# them at module level, so a missing file breaks the wrapper's import, not just
# upstream's own main().
for f in model.py configure.py utils.py train_eval.py datasets.py \
         GenesMetrics.py README.md; do
    if [[ ! -e "$REPO_DIR/$f" ]]; then
        log "ERROR: expected '$f' under $REPO_DIR"; exit 1
    fi
done
log "sanity: all six upstream modules + README present"
if [[ ! -e "$REPO_DIR/LICENSE" && ! -e "$REPO_DIR/LICENSE.md" ]]; then
    log "NOTE: upstream ships no LICENSE (matches the GitHub API, which reports"
    log "      licence: none). We execute the code in place and redistribute"
    log "      nothing; do not vendor it into this repo without asking."
fi
# The shipped data/ dir is decoration: data/dro_rna.h5ad is a 2-byte placeholder,
# so upstream's datasets.load_data cannot run as distributed. run_come.py never
# calls load_data (it builds AnnData in memory from our silver), which is why
# this is a note and not an error.
if [[ -e "$REPO_DIR/data/dro_rna.h5ad" ]]; then
    _sz="$(wc -c < "$REPO_DIR/data/dro_rna.h5ad" | tr -d ' ')"
    if [[ "$_sz" -lt 1024 ]]; then
        log "NOTE: upstream data/dro_rna.h5ad is ${_sz} bytes (a placeholder, not"
        log "      data). Nothing here depends on the shipped data/ dir."
    fi
fi

# ---------------------------------------------------------------------------
# 2. Create / refresh the uv venv
# ---------------------------------------------------------------------------
if [[ -d "$VENV_DIR" ]]; then
    log "venv already exists: $VENV_DIR"
    log "  (to rebuild:  rm -rf $VENV_DIR  &&  bash $0  )"
else
    log "creating venv at $VENV_DIR (Python $PYTHON_VERSION)..."
    mkdir -p "$(dirname "$VENV_DIR")"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

UV_PIP=(uv pip install --python "$VENV_DIR/bin/python")

# ---------------------------------------------------------------------------
# 3. Dependencies
#
#    ORDER MATTERS. torch goes first, from the CUDA index, so the CUDA build is
#    what lands; then the scientific stack from PyPI carrying the numpy
#    constraint. Installing the scanpy group LAST lets its resolve fix numpy for
#    the env — the other order leaves torch's (nonexistent) numpy requirement
#    free and numpy is then whatever the last install pulls in.
#
#    scanpy brings anndata/pandas/scipy/scikit-learn/matplotlib/seaborn/h5py
#    transitively, but every package COME imports by name is ALSO named here:
#    the env must not break because a future scanpy drops a dependency (seaborn
#    is the live example — see the header).
# ---------------------------------------------------------------------------
# Apply the ceiling to whatever TORCH_SPEC holds, so the documented CPU escape
# hatch (TORCH_SPEC="torch") gets it too and not just the CUDA default: a bare
# `torch` token becomes `torch$TORCH_MAX`; a constraint the caller wrote
# themselves is left alone, and so are the index-url flags. Identical to
# setup_cellcontrast_env.sh, deliberately.
TORCH_ARGS=()
# shellcheck disable=SC2086  # TORCH_SPEC is a command line; word-splitting is the point
for _tok in $TORCH_SPEC; do
    [[ "$_tok" == "torch" ]] && _tok="torch$TORCH_MAX"
    TORCH_ARGS+=("$_tok")
done

log "installing torch (${TORCH_ARGS[*]}) ..."
"${UV_PIP[@]}" "${TORCH_ARGS[@]}"

log "installing scanpy==$SCANPY_VERSION with '$NUMPY_SPEC' plus anndata, pandas, scipy, scikit-learn, matplotlib, seaborn ..."
"${UV_PIP[@]}" "scanpy==$SCANPY_VERSION" "$NUMPY_SPEC" \
    anndata pandas scipy scikit-learn matplotlib seaborn

# ---------------------------------------------------------------------------
# 4. Verify — every import the upstream package actually makes
# ---------------------------------------------------------------------------
log "verifying ..."
COME_TORCH_MAX="$TORCH_MAX" COME_NUMPY_SPEC="$NUMPY_SPEC" \
"$VENV_DIR/bin/python" - <<'PY'
import importlib, os, re, sys
# Enumerated from upstream itself, at the pinned commit:
#   grep -hE '^[[:space:]]*(import|from) ' *.py
# Third-party top-level packages (the rest are stdlib: os, argparse, random,
# datetime). anndata is ours (run_come.py builds AnnData objects directly).
required = ("torch", "scanpy", "anndata", "numpy", "pandas",
            "scipy", "sklearn", "matplotlib", "seaborn")
missing = []
for m in required:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        missing.append(m)
        print(f"  MISS {m:12s} {type(e).__name__}: {e}")

# numpy: honour the declared bound rather than trusting the resolve. See the
# NUMPY_SPEC comment — this is drift control, not a COME requirement.
# EVERY comparator in the spec is checked, not just the ceiling: a substring
# test for '<2' would report "satisfies 'numpy>=1.26,<2'" for numpy 1.22 (the
# floor is part of the claim) and would misread an override like 'numpy<2.5'
# as the numpy-1 pin, failing a perfectly valid env.
def _ver(s):
    out = []
    for part in str(s).split(".")[:3]:
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        out.append(int(digits or 0))
    return tuple(out + [0] * (3 - len(out)))


spec = os.environ.get("COME_NUMPY_SPEC", "")
try:
    import numpy as _np
    got = _ver(_np.__version__)
    bounds = re.findall(r"(>=|<=|==|!=|>|<)\s*([0-9][0-9.]*)", spec)
    broken = []
    for op, ver in bounds:
        want = _ver(ver)
        ok = {">=": got >= want, "<=": got <= want, ">": got > want,
              "<": got < want, "==": got == want, "!=": got != want}[op]
        if not ok:
            broken.append(f"{op}{ver}")
    if broken:
        missing.append(spec)
        print(f"  FAIL numpy {_np.__version__} violates {', '.join(broken)} of "
              f"NUMPY_SPEC '{spec}'. The resolve ignored the bound; rebuild the "
              f"venv.")
    elif bounds:
        print(f"  OK   numpy satisfies '{spec}' ({_np.__version__})")
    else:
        print(f"  OK   numpy {_np.__version__} (NUMPY_SPEC declares no bound)")
except Exception as e:
    print(f"  NOTE numpy bound not checked: {type(e).__name__}: {e}")

# torch: report, do not veto. COME's checkpoints are plain state_dicts
# (train_eval.py:167,197 save; :109 loads straight into load_state_dict), so the
# torch>=2.6 weights_only default does NOT break it — unlike CellContrast, whose
# checkpoint carries a pandas Index. What matters here is only that we got the
# build we asked for; the functional check below proves the round trip.
# We declare a CEILING (TORCH_MAX, default "<2.6"), not an exact build -- an
# exact pin on this index is unsatisfiable, see the TORCH_MAX comment near the
# top -- so verify the ceiling holds rather than an equality.
want = os.environ.get("COME_TORCH_MAX", "").strip()
try:
    import torch as _t
    got = str(_t.__version__)

    def _rel(v):
        # "2.5.0+cu121" -> (2, 5, 0); ignore the local/CUDA suffix.
        base = v.split("+")[0].split("rc")[0]
        out = []
        for part in base.split(".")[:3]:
            digits = "".join(c for c in part if c.isdigit())
            out.append(int(digits) if digits else 0)
        while len(out) < 3:
            out.append(0)
        return tuple(out)

    ok, bound = True, ""
    for op in ("<=", ">=", "==", "!=", "<", ">"):
        if want.startswith(op):
            bound = want[len(op):].strip()
            a, b = _rel(got), _rel(bound)
            ok = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b,
                  "==": a == b, "!=": a != b}[op]
            break
    if want and not bound:
        print(f"  NOTE torch {got}: TORCH_MAX '{want}' has no recognised "
              f"comparator; not checked")
    elif not want:
        print(f"  OK   torch {got} (no ceiling declared)")
    elif ok:
        print(f"  OK   torch {got} satisfies TORCH_MAX '{want}'")
    else:
        print(f"  WARN torch {got} VIOLATES TORCH_MAX '{want}'. Fine if you "
              f"overrode TORCH_SPEC on purpose; otherwise the resolve drifted "
              f"and the recorded env is not the one in the methods.")
    mj, mn = (int(p) for p in got.split(".")[:2])
    if (mj, mn) >= (2, 6):
        print("  NOTE torch>=2.6 defaults torch.load to weights_only=True. "
              "COME only ever loads a state_dict, so this is safe — but the "
              "functional check below is what actually proves it.")
except Exception as e:
    missing.append("torch"); print(f"  MISS torch version check: {e}")

# Submodules/symbols upstream actually reaches for, with the line that uses each.
# A top-level package can import while these do not.
# matplotlib.pyplot AND seaborn are imported at MODULE level by GenesMetrics.py,
# which train_eval.py:16 imports — so both must be importable on a display-less
# node. matplotlib picks Agg automatically when there is no DISPLAY; force it
# here so the check behaves the same on a head node and in the job.
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass


def resolve(path):
    """Resolve a dotted path that may mix submodules and attributes.

    Import the LONGEST importable prefix, then walk the rest with getattr. A
    plain import_module(path.rsplit('.', 1)[0]) is not enough: `scanpy.pp` is an
    ALIAS ATTRIBUTE (scanpy/__init__.py does `from . import preprocessing as
    pp`), not a submodule, so import_module('scanpy.pp') raises
    ModuleNotFoundError even in a perfectly healthy env — which would have made
    this check fail every run. Verified against a minimal alias package.
    """
    parts = path.split(".")
    first_err = None
    for i in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:i]))
        except Exception as e:      # keep the deepest failure for the message
            first_err = first_err or e
            continue
        obj = mod
        for p in parts[i:]:
            obj = getattr(obj, p)   # AttributeError here is a real failure
        return obj
    raise first_err or ImportError(f"cannot import any prefix of {path}")


symbols = (
    ("sklearn.preprocessing.MinMaxScaler",  "utils.py:5,47,50 (normalize_type, sttype=image)"),
    ("sklearn.model_selection.KFold",       "train_eval.py:15,50"),
    ("scipy.io.savemat",                    "utils.py:8,95 (writes <name>_type_mask.mat)"),
    ("scipy.io.loadmat",                    "train_eval.py:5,62 (reads the cached mask)"),
    ("scipy.stats.pearsonr",                "GenesMetrics.py:8,231"),
    ("scipy.stats.spearmanr",               "GenesMetrics.py:8,247"),
    ("scipy.stats.entropy",                 "GenesMetrics.py:8,268 (JSD term)"),
    ("scipy.stats.zscore",                  "GenesMetrics.py:7,119"),
    ("torch.nn.functional.mse_loss",        "model.py:2-3,43 (AutoEncoder.loss_ae)"),
    ("torch.utils.data.DataLoader",         "train_eval.py:7,142"),
    ("torch.utils.data.RandomSampler",      "datasets.py:3"),
    ("torch.utils.data.SequentialSampler",  "datasets.py:3"),
    ("torch.optim.Adam",                    "train_eval.py:103-104"),
    ("scanpy.pp.normalize_total",           "utils.py:55,59 (sttype=sequence)"),
    ("scanpy.pp.log1p",                     "utils.py:56,60"),
    ("scanpy.pp.filter_cells",              "datasets.py:38 (the min_genes=1 drop run_come.py guards)"),
    ("scanpy.pp.filter_genes",              "datasets.py:37"),
    ("scanpy.read_h5ad",                    "datasets.py:8ff (load_data)"),
    ("anndata.AnnData",                     "run_come.py:_make_adata"),
    ("anndata.read_h5ad",                   "run_come.py:build_reference/load_query"),
    ("pandas.DataFrame",                    "train_eval.py:12,134; utils.py:9"),
    ("seaborn.boxplot",                     "GenesMetrics.py:9,390 (MODULE-LEVEL import via train_eval)"),
    ("matplotlib.pyplot.savefig",           "GenesMetrics.py:10 (MODULE-LEVEL import via train_eval)"),
)
for path, where in symbols:
    try:
        resolve(path)
        print(f"  OK   {path}  <- {where}")
    except Exception as e:
        missing.append(path)
        print(f"  MISS {path} ({where}): {type(e).__name__}: {e}")

try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}"
          + (f" -> {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    if not torch.cuda.is_available():
        print("  NOTE: no CUDA here. Head nodes usually have no GPU — this is "
              "expected. For COME specifically, CPU is not just a fallback: "
              "upstream builds full_mask on the CPU while cross_mask lives on "
              "the model's device (model.py loss_fn), so run_come.py's "
              "--device cpu is the safe path anyway.")
except Exception:
    pass

sys.exit(1 if missing else 0)
PY

# Import the AUTHORS' own modules — the real proof the env can run their code.
# train_eval.py runs argparse AT MODULE LEVEL (line 29) and its train loops read
# the resulting global (args.patience), so sys.argv is stubbed to a bare vector
# exactly as run_come.py:_import_come does; otherwise it would try to parse
# whatever flags happen to be around and exit. This import is also what proves
# the GenesMetrics -> seaborn/matplotlib chain resolves.
log "importing upstream modules from $REPO_DIR ..."
( cd "$REPO_DIR" && "$VENV_DIR/bin/python" - <<'PY'
import sys, traceback
sys.argv = ["train_eval.py"]
bad = []
for m in ("configure", "utils", "model", "datasets", "GenesMetrics", "train_eval"):
    try:
        __import__(m)
        print(f"  OK   {m}")
    except Exception as e:
        bad.append(m)
        print(f"  FAIL {m}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
sys.exit(1 if bad else 0)
PY
)

# ---------------------------------------------------------------------------
# 4b. Functional check — actually RUN COME for two epochs
#
#     Importing proves nothing about whether these pins can fit the model. This
#     runs the authors' real code path end to end on a 12-spot x 9-cell toy
#     problem: their normalize_type (the MinMaxScaler path), their
#     cell_type_2_martix (which savemat's into data/), their pretrain_ae
#     (torch.save) + torch.load round trip, their train loop (the full five-term
#     loss, including the (n1+n2)^2 full_mask and ContrastiveLoss), and the
#     Coefficient read-out run_come.py takes the argmax of. Seconds on CPU.
#
#     It runs in a scratch cwd, never in the checkout, because utils.py:95 and
#     train_eval.py:167,197 write relative to cwd.
# ---------------------------------------------------------------------------
if [[ "$SKIP_FUNCTIONAL" != "1" ]]; then
    log "functional check: fitting COME for 2 epochs on a toy problem ..."
    FUNC_DIR="$(mktemp -d "${TMPDIR:-/tmp}/come_envcheck.XXXXXX")"
    # shellcheck disable=SC2064  # expand FUNC_DIR now, on purpose
    trap "rm -rf '$FUNC_DIR'" EXIT
    if ! ( cd "$FUNC_DIR" && COME_REPO="$REPO_DIR" MPLBACKEND=Agg \
           "$VENV_DIR/bin/python" - <<'PY'
import os, sys
import numpy as np

sys.path.insert(0, os.environ["COME_REPO"])
sys.argv = ["train_eval.py"]          # train_eval parses argv at import
import anndata as ad
import pandas as pd
import torch
import configure                       # noqa: E402
import model as come_model             # noqa: E402
import utils as come_utils             # noqa: E402
import train_eval                      # noqa: E402

for d in ("data", "pretrain", "result"):
    os.makedirs(d, exist_ok=True)
train_eval.device = torch.device("cpu")   # the path run_come.py --device cpu uses

rng = np.random.default_rng(0)
n1, n2, g = 12, 9, 6                      # spots, cells, genes
genes = [f"g{i}" for i in range(g)]


def _ad(n, tag):
    a = ad.AnnData(X=rng.random((n, g)).astype(np.float32),
                   obs=pd.DataFrame({"cell_type": [f"t{i % 3}" for i in range(n)]}))
    a.var_names = genes
    a.obs_names = [f"{tag}{i}" for i in range(n)]
    return a


spot, rna = _ad(n1, "s"), _ad(n2, "c")
spot_n, rna_n = come_utils.normalize_type(spot, rna, type="image")   # THEIR prep
# np.array (a COPY), not np.asarray: normalize_type ends in
# filter_with_overlap_gene, which returns anndata VIEWS, and train_eval.train
# hands the arrays to torch.from_numpy — which wants a plain writable ndarray,
# not an ArrayView. Upstream copies at the same point (train_eval.py:80-81
# `.astype(np.float32)`).
x1 = np.array(spot_n.X, dtype=np.float32)
x2 = np.array(rna_n.X, dtype=np.float32)
assert x1.shape == (n1, g) and x2.shape == (n2, g), (x1.shape, x2.shape)

type_mask = come_utils.cell_type_2_martix(rna_n, data_name="envcheck")  # savemat
assert type_mask.shape == (n2, n2)
assert os.path.exists("data/envcheck_type_mask.mat"), "savemat wrote nothing"

config = configure.get_default_config("MERFISH")
config["num_sample1"], config["num_sample2"] = n1, n2
config["dims"] = list(config["dims"])
config["dims"][0] = g
config["pretrain_epochs"], config["epochs"], config["batch_size"] = 2, 2, 8

m = come_model.Model(config).to(train_eval.device)
opt_pre = torch.optim.Adam(m.ae.parameters(), lr=config["pre_lr"])
opt = torch.optim.Adam(m.parameters(), lr=config["lr"])

pre_path = "pretrain/envcheck.pkl"
train_eval.pretrain_ae(m.ae, opt_pre, np.concatenate((x1, x2), axis=0),
                       config, pre_path)
# The torch.load the header argues about, exercised on the RESOLVED torch: what
# COME saves is a state_dict, so this must work with or without weights_only.
m.ae.load_state_dict(torch.load(pre_path))
m.train()
train_eval.train(m, opt, x1, x2, type_mask, config, "result/envcheck_model.pkl")
m.eval()

C = m.map.Coefficient.detach().cpu().numpy()
assert C.shape == (n1, n2), f"Coefficient is {C.shape}, expected {(n1, n2)}"
assert np.isfinite(C).all(), "Coefficient has NaN/Inf after 2 epochs"
idx = np.argmax(C, axis=0)                # run_come.coords_from_coefficient rule
assert idx.shape == (n2,) and idx.max() < n1
print(f"  OK   fitted: Coefficient {C.shape}, finite, argmax over spots -> "
      f"{len(set(idx.tolist()))} distinct spot(s) for {n2} cells")
print("  OK   state_dict save/load round trip works on this torch")
PY
    ); then
        log "ERROR: the env imports but COME does not RUN in it."
        log "  Re-run the block above by hand for the full traceback, or rebuild:"
        log "    rm -rf $VENV_DIR && bash $0"
        log "  (To get past this deliberately: SKIP_FUNCTIONAL=1 — but then the"
        log "   env has never fitted the model and jobs may die in the queue.)"
        exit 1
    fi
    rm -rf "$FUNC_DIR"
    trap - EXIT
else
    log "SKIP_FUNCTIONAL=1 (COME was NOT actually fitted in this env)"
fi

# ---------------------------------------------------------------------------
# 5. Lockfile — the dependency set we actually resolved
#
#    `uv pip install` re-resolves on every run, so two builds months apart can
#    differ in every transitive pin from identical arguments here. Freeze what we
#    got, next to the venv it describes, so the env is recoverable (and, when a
#    result stops reproducing, diffable).
# ---------------------------------------------------------------------------
LOCKFILE="${LOCKFILE:-$VENV_DIR/requirements.lock}"
# The freeze records the CUDA build as a LOCAL version (torch==2.5.1+cu121),
# which exists only on the PyTorch index — so a recreate line without that index
# cannot resolve, and a "copy-pasteable" comment would be a lie. It goes in as
# --extra-index-url rather than the --index-url we install torch with:
# --index-url REPLACES PyPI, and scanpy/anndata/... are not on the PyTorch index.
RECREATE_EXTRA=""
_want_url=0
for _tok in "${TORCH_ARGS[@]}"; do
    if (( _want_url )); then
        RECREATE_EXTRA="$RECREATE_EXTRA --extra-index-url $_tok"; _want_url=0
    elif [[ "$_tok" == --index-url || "$_tok" == --extra-index-url ]]; then
        _want_url=1
    elif [[ "$_tok" == --index-url=* || "$_tok" == --extra-index-url=* ]]; then
        RECREATE_EXTRA="$RECREATE_EXTRA --extra-index-url ${_tok#*=}"
    fi
done
{
    printf '# COME baseline env, resolved %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '# upstream commit : %s\n' "$COMMIT"
    printf '# upstream        : https://github.com/cindyway/COME (no licence file)\n'
    printf '# python          : %s\n' "$PYTHON_VERSION"
    printf '# torch requested : %s\n' "${TORCH_ARGS[*]}"
    printf '# scanpy / numpy  : scanpy==%s , %s\n' "$SCANPY_VERSION" "$NUMPY_SPEC"
    printf '# recreate        : uv venv --python %s <venv> && uv pip install --python <venv>/bin/python -r %s%s\n' \
        "$PYTHON_VERSION" "$LOCKFILE" "$RECREATE_EXTRA"
} > "$LOCKFILE"
uv pip freeze --python "$VENV_DIR/bin/python" >> "$LOCKFILE"
log "lockfile: $LOCKFILE ($(grep -cv '^#' "$LOCKFILE" || true) pinned packages)"

# ---------------------------------------------------------------------------
# 6. Post-install self-check — the wrapper's own test suite
#
#    numpy-only: no GPU, no /nfs, seconds to run. test_come_wrapper.py covers the
#    read-out axis (argmax over SPOTS — the wrong axis silently produces
#    plausible-but-wrong coordinates), the degenerate/NaN refusals, the
#    feasibility arithmetic that decides whether a job is launched at all, and
#    the empty-cell policy protecting row alignment against upstream's
#    filter_cells(min_genes=1). test_harness_adapter.py is included because
#    run_come.py imports harness_adapter for split discovery, the coordinate
#    scaler and the artifact schema the scorer reads.
#    An env that imports cleanly is not evidence of any of that.
# ---------------------------------------------------------------------------
if [[ "$SKIP_TESTS" != "1" ]]; then
    CB_DIR="$HERE/contrastive_baselines"
    log "running the wrapper test suite ($VENV_DIR/bin/python) ..."
    if [[ ! -f "$CB_DIR/test_come_wrapper.py" ]]; then
        log "ERROR: expected test file $CB_DIR/test_come_wrapper.py"; exit 1
    fi
    test_fails=()
    for t in test_come_wrapper.py test_harness_adapter.py; do
        if [[ ! -f "$CB_DIR/$t" ]]; then
            log "  SKIP $t (not present)"; continue
        fi
        # cwd must be CB_DIR: each test imports run_come / harness_adapter from
        # alongside itself.
        if out="$( cd "$CB_DIR" && "$VENV_DIR/bin/python" "$t" 2>&1 )"; then
            log "  PASS $t"
        else
            log "  FAIL $t"
            printf '%s\n' "$out" >&2
            test_fails+=("$t")
        fi
    done
    if (( ${#test_fails[@]} > 0 )); then
        log "ERROR: ${#test_fails[@]} test file(s) failed: ${test_fails[*]}"
        log "  The env built but the baseline's own checks do not pass. Do not"
        log "  submit jobs until they do; re-run one for full output with"
        log "    cd $CB_DIR && $VENV_DIR/bin/python ${test_fails[0]}"
        exit 1
    fi
else
    log "SKIP_TESTS=1 (wrapper test suite NOT run)"
fi

# ---------------------------------------------------------------------------
log "done."
log "  venv : $VENV_DIR"
log "  repo : $REPO_DIR (commit $COMMIT)"
log "  lock : $LOCKFILE"
cat <<EOF

NEXT — smoke-test before any real run (proves the install end-to-end, minutes):

  bash $HERE/contrastive_baselines/submit_come.sh \\
      --dataset mmc_luna --smoke_test

Then the real runs (mmc_luna ONLY — see below):

  bash $HERE/contrastive_baselines/submit_come.sh \\
      --dataset mmc_luna --seeds "0 1 2 3 4"

(The submitter defaults to this venv and repo, so no extra flags are needed.)

SCALE — why mmc_luna only. COME is transductive: MapNet.Coefficient is a dense
(n_spots x n_cells) nn.Parameter, refitted per (reference, test slice) pair, and
the loss rebuilds several dense (n_ref+n_query)^2 masks every epoch, so peak
memory is O((n_ref+n_query)^2). The protocol is the FULL test slice (<=5,235
cells) against a reference capped by --max_ref_cells (default 20,000, ~15 GB), so
the evaluated cell population stays identical to LUNA/G2T/CeLEry/CellContrast and
the only deviation is reference size. cns_luna is OUT OF SCOPE: its 63,343-cell
query needs ~96 GB with an EMPTY reference, so a smaller reference cannot rescue
it; the submitter refuses it with that arithmetic.

If submit_come.sh is not in place yet, the equivalent direct call is:

  cd $HERE/contrastive_baselines && $VENV_DIR/bin/python run_come.py \\
      --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
      --come_repo $REPO_DIR \\
      --dataset mmc_luna --max_ref_cells 20000 --device cpu --seed 0 --smoke_test
EOF
