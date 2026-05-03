#!/usr/bin/env bash
# Run one of the four CMB notebooks under nice 10 with BLAS pinned to 1
# thread per worker. Output is logged to logs/<short>_<timestamp>.log and the
# notebook is executed in place so plots/print outputs are persisted.
#
# Usage:
#     ./run_nb.sh cal           # calibration uncertainty
#     ./run_nb.sh oe            # odd-even split
#     ./run_nb.sh noise         # noise miscalibration
#     ./run_nb.sh bt            # beam telescope
#
# Recommended invocation from a fresh shell on a remote box:
#     tmux new -d -s nb_<short> /data/vault/jv447/repos/Part-III-Research-Project/run_nb.sh <short>
#
# Then `tmux attach -t nb_<short>` to watch progress, Ctrl-B D to detach.

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 {cal|oe|noise|bt}" >&2
    exit 2
fi

REPO=/data/vault/jv447/repos/Part-III-Research-Project
PYTHON=/data/vault/jv447/venvs/research-project/bin/python

case "$1" in
    cal)   NB="Cmb_Test_signal_perturbation_output.ipynb"        ; SHORT=cal   ;;
    oe)    NB="CMB_Test_Beam_Miscalibration_Odd_Even_Split.ipynb"; SHORT=oe    ;;
    noise) NB="CMB_Test_Noise_Detector_newest.ipynb"             ; SHORT=noise ;;
    bt)    NB="CMB_Test_Beam_Telescope_newest.ipynb"             ; SHORT=bt    ;;
    *)
        echo "unknown notebook key: $1 (use cal, oe, noise, bt)" >&2
        exit 2
        ;;
esac

cd "$REPO"
mkdir -p logs
LOG="logs/${SHORT}_$(date +%Y%m%d_%H%M).log"

# Kernel to use. The `research-project` kernel.json hardcodes the full path
# to the venv's python, so it cannot be hijacked by a wrong $PATH. The bare
# `python3` kernel uses just "python" and resolves via $PATH at launch time
# - which is exactly how runs end up in the wrong env on shared boxes.
KERNEL=research-project

# BLAS thread cap: each joblib worker is 1 thread, so total threads ~ N_WORKERS
# rather than N_WORKERS * 48. Without these the box gets oversubscribed and
# every other user notices.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

{
    echo "=== run_nb.sh start  notebook=$NB  host=$(hostname)  pid=$$  $(date) ==="
    echo "    log=$LOG"
    echo "    nice 10, OMP/OPENBLAS/MKL_NUM_THREADS=1"
    echo "    nbconvert python = $PYTHON"
    "$PYTHON" -c "import sys; print('    nbconvert python version =', sys.version.split()[0])"
    echo "    forced kernel    = $KERNEL"
    echo "    kernel argv      = $($PYTHON -m jupyter kernelspec list --json 2>/dev/null \
        | $PYTHON -c "import sys, json; ks = json.load(sys.stdin)['kernelspecs'].get('$KERNEL', {}); \
        print(ks.get('spec', {}).get('argv', ['<not found>']))")"
    echo "    -> if 'kernel argv' starts with /data/vault/jv447/venvs/research-project/bin/python"
    echo "       you are running in the right env. Anything else means rebuild the kernel."
    echo

    nice -n 10 "$PYTHON" -m jupyter nbconvert \
        --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name="$KERNEL" \
        "$NB"
    rc=$?

    echo
    echo "=== run_nb.sh end  rc=$rc  $(date) ==="
    exit $rc
} 2>&1 | tee "$LOG"
