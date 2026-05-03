"""Smoke test: confirm the venv has everything the four CMB notebooks need.

Usage on the remote box:
    cd /data/vault/jv447/repos/Part-III-Research-Project
    /data/vault/jv447/venvs/research-project/bin/python check_env.py

Exits 0 on success, non-zero if any import or sanity call fails.
"""
import sys
import time
import importlib

REQUIRED = [
    "numpy", "matplotlib", "pandas", "torch", "joblib",
    "scipy", "camb", "healpy",
    "sbi", "sbi.utils", "sbi.inference", "sbi.analysis",
    "torch.distributions", "nbformat", "nbconvert", "ipykernel",
    "getdist",
]

OPTIONAL = ["chainconsumer", "corner"]


def check_imports():
    print(f"python: {sys.version.split()[0]}  ({sys.executable})")
    failed = []
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  ok    {mod:<24} {ver}")
        except Exception as e:
            print(f"  FAIL  {mod:<24} {e!r}")
            failed.append(mod)
    for mod in OPTIONAL:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  ok    {mod:<24} {ver}  (optional)")
        except Exception:
            print(f"  --    {mod:<24} not installed (optional)")
    return failed


def check_camb():
    """One CAMB call - this is the most likely thing to be misconfigured
    on a fresh box (CAMB needs a Fortran build at install time)."""
    print("\nCAMB sanity check (one TT spectrum at lmax=3000)...")
    import camb
    t0 = time.time()
    pars = camb.set_params(
        H0=67.5, ombh2=0.022, omch2=0.122,
        mnu=0.06, omk=0, tau=0.06,
        As=2e-9, ns=0.965,
        halofit_version="mead", lmax=3000,
    )
    res = camb.get_results(pars)
    cl_tt = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["total"][:, 0]
    dt = time.time() - t0
    print(f"  ok    cl_tt[2:5] = {cl_tt[2:5]}  ({dt:.2f}s)")
    return dt


def check_healpy():
    print("\nhealpy sanity check (synalm at lmax=3000)...")
    import healpy as hp
    import numpy as np
    cl = np.ones(3001) * 1e-3
    t0 = time.time()
    alm = hp.synalm(cl, lmax=3000)
    dt = time.time() - t0
    print(f"  ok    len(alm) = {len(alm)}  ({dt:.2f}s)")


def check_torch():
    print("\nTorch sanity check...")
    import torch
    print(f"  ok    torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"        gpus: {torch.cuda.device_count()}, name: {torch.cuda.get_device_name(0)}")


def check_sbi_quick():
    """Train a 30-second NPE on toy data to confirm sbi+nflows works end-to-end."""
    print("\nsbi end-to-end smoke (toy NPE, 1000 sims, ~30s)...")
    import torch, time
    from sbi.utils import BoxUniform
    from sbi.inference import NPE
    prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
    theta = prior.sample((1000,))
    x = theta + 0.1 * torch.randn_like(theta)
    t0 = time.time()
    inf = NPE(prior=prior, density_estimator="nsf")
    inf.append_simulations(theta, x).train(max_num_epochs=30, stop_after_epochs=10)
    post = inf.build_posterior()
    samples = post.sample((100,), x=torch.tensor([0.5, 0.5]))
    dt = time.time() - t0
    print(f"  ok    samples shape {tuple(samples.shape)}  ({dt:.1f}s)")


def check_jupyter():
    """Make sure nbconvert can actually run a kernel."""
    print("\nnbconvert + ipykernel smoke...")
    import subprocess
    out = subprocess.run(
        [sys.executable, "-m", "jupyter", "kernelspec", "list"],
        capture_output=True, text=True, timeout=20,
    )
    if out.returncode != 0:
        print(f"  FAIL  {out.stderr.strip()}")
        return False
    print("  ok    available kernels:")
    for line in out.stdout.strip().splitlines():
        print(f"        {line}")
    return True


def main():
    failed = check_imports()
    if failed:
        print(f"\n*** {len(failed)} required imports failed: {failed}")
        sys.exit(1)

    try:
        check_torch()
        check_camb()
        check_healpy()
        check_sbi_quick()
        check_jupyter()
    except Exception as e:
        print(f"\n*** sanity call raised: {e!r}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    print("\n*** all checks passed - venv is ready to run the four notebooks.")


if __name__ == "__main__":
    main()
