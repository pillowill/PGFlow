# PG-Flow toy experiments

This repository starts with the hypotheses in `PG_Flow_Research_Proposal.md`. `toy/toy1.py` tests the discrete two-dimensional metric path only; it does not train a flow model.

Install NumPy and SciPy, then run `python toy/toy1.py`. Results, including path nodes and optimizer status, are written to `results/toy1.json`. The straight path is included as a candidate, and multiple initializations reduce sensitivity to symmetry. Midpoint discretization and local optimization do not certify a global optimum. Future experiments should use separate scripts/configurations and record seeds and outputs under `results/`.

Run python toy/toy4_control.py for the preliminary analytic-path FM control. This is not a physical geodesic experiment; see the proposal revision log for interpretation.
`nRun `python toy/toy4_metric.py` for the paired metric-path FM experiment (three seeds; SciPy required). Results include preprocessing time and MMD at several Euler step counts.
`nRun `python toy/toy4_strength_scan.py` for the paired strength scan and `python toy/toy2_periodic.py` for the two-particle periodic stiffness check.
`nThe exploratory MP-20 script `experiments/mp20_path_pilot.py` takes `--data` pointing to val.csv. Its target-dependent source and finite periodic-image approximation limit it to path diagnostics; see the proposal for details.
`nThe corrected periodic diagnostic is `experiments/mp20_path_scan.py`; use `--data` to supply the MP-20 validation CSV. It remains a target-dependent path experiment, not a generation evaluation.

## CrystalFlow integration (experimental)

`crystalflow-upstream/` contains the locally supplied CrystalFlow source, with a PG-Flow path switch. Its git metadata identifies official upstream HEAD `9c25ff0245d787efd87c9d1d797a8968443608cc`, but checkout of upstream blobs failed and the local source has not been verified against that revision; the git index/status therefore does not represent a clean upstream checkout. Do not treat this directory as a reproducible upstream revision until a clean checkout and diff have been completed.

For `model=flow_polar`, `model.physical_path_enabled=false` is the default. Set `model.physical_path_enabled=true` to use the experimental CPU SciPy path solver for coordinate supervision; options are `model.physical_path_segments`, `model.physical_path_alpha`, `model.physical_path_cutoff`, `model.physical_path_sigma`, and `model.physical_path_maxiter`. It keeps CrystalFlow's source, pairing and lattice branch, and uses the target lattice solely during training to define the coordinate-path proxy. It is not a joint lattice-coordinate geodesic and rejects symmetry-constrained modes. CPU per-pair numerical optimization is expensive and is for small studies, not production training.

Run `python experiments/check_crystalflow_path.py` from the project root for endpoint, derivative and zero-alpha checks. Full training needs CrystalFlow's dependencies (including torch_scatter and torchdyn) and dataset configuration. Set Hydra `data.root_path` to a writable local copy of the MP-20 split files, since CrystalFlow creates preprocessing caches alongside the CSVs. Use identical seeds and settings for switch-off/on runs, and report path preprocessing costs separately. No full training or CSP benchmark has been run here.

### Current working copy and environment

The current implementation and full local data are in `repos/CrystalFlow-project/`, copied from the user-provided local CrystalFlow tree; it is not a verified upstream checkout. The previous `crystalflow-upstream/` section is retained as historical provenance, not the active working copy. `experiments/check_crystalflow_path.py` now targets `repos/CrystalFlow-project/`. The AI4MatGen environment had lightning 2.6.6 installed during setup, but executing the check exposed an OpenMP runtime collision (`libomp.dll` and `libiomp5md.dll`); the integration has not yet passed checks in that environment. Full model forward and training remain unverified.

### Environment repair update

AI4MatGen now uses the PyPI Windows wheels for NumPy 2.4.4 and SciPy 1.17.1 (same versions as before). Without `KMP_DUPLICATE_LIB_OK`, the path checks and CSPFlow import pass. Full training and other projects using the shared environment have not been regression tested. The previous collision warning above describes the earlier state. Old directories remain because the execution policy rejected recursive deletion.
`nRun the real-CIF integration check with `AI4MatGen` Python and `python experiments/crystalflow_forward_smoke.py` from the project root. It disables pandas Arrow string inference locally to avoid an observed Windows crash; this is not yet a verified fix for the full training entry point.

### AI4MatGen training environment (2026-10-30)

The pandas Arrow crash was avoided by installing pandas 2.3.3 and pytz; NumPy 2.4.4, SciPy 1.17.1 and PyArrow 24.0.0 use PyPI Windows wheels. SwanLab and torchdyn were installed; protobuf changed to 6.33.6. `pip check` passes. A small CPU `fast_dev_run` completed one train and validation batch in each path mode (physical mode used maxiter=2 and small crystals). This does not validate GPU or long runs, nor unrelated projects sharing this environment.

Run from `repos/CrystalFlow-project`; `diffcsp/run.py` now places this repository first on the Python import path. The current PyTorch version defaults to safe weights-only loading, whereas CrystalFlow preprocessing caches contain Python/NumPy objects. For caches you created yourself and trust, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` only in the training process; never use that setting to load untrusted caches or checkpoints. Existing caches created by another `diffcsp` project are incompatible and should be regenerated in a separate directory.

### Torch physical-path backend

Set `model.physical_path_enabled=true model.physical_path_backend=torch` for the experimental Torch optimizer. `scipy` remains the default reference backend; enabling the method is still opt-in. Torch uses fixed-step Adam and the best candidate including the straight path, and is not guaranteed to converge to the SciPy solution. Run `experiments/check_torch_path.py` for CPU/GPU invariants and `experiments/benchmark_gpu_paths.py` for the small MP-20 timing comparison. The benchmark uses different algorithms and iteration counts and must be interpreted alongside the reported action; it does not establish equal-quality speedup.

### Independent pgflow environment

Activate `conda activate pgflow`. The environment was cloned from AI4MatGen and numerical wheels pinned to NumPy 2.4.4, SciPy 1.17.1, pandas 2.3.3 and PyArrow 24.0.0; it includes unrelated inherited packages. GPU fast-development runs of both baseline and torch physical paths passed on RTX 2080 Ti. The full training entry point and cached-data caveat above apply. For the torch path enable both `model.physical_path_enabled=true` and `model.physical_path_backend=torch`. GPU/CPU action and timing evidence is in `results/pgflow_gpu_scaling_pilot.json`; the two optimizers have different budgets and are not a matched-convergence benchmark.

### Repository and data

The repository tracks the research proposal, source, scripts and lightweight experiment summaries. CrystalFlow datasets, pretrained weights, generated caches and training logs are excluded from Git. Obtain compatible datasets and weights from the [CrystalFlow project](https://github.com/ixsluo/CrystalFlow), then place them under `repos/CrystalFlow-project/data/` and the relevant model directories. The copied CrystalFlow source retains its MIT license in `repos/CrystalFlow-project/LICENSE`; the local source snapshot was supplied without Git history and is not claimed to match a verified upstream revision.

### Toy experiment suite

The runnable Toy-1–5 mechanism experiments are `toy/toy1_scan.py`, `toy/toy2_anisotropy.py` and `toy/toy2_periodic.py`, `toy/toy3_periodic_matching.py`, `toy/toy4_ablation.py` and `toy/toy4_strength_scan.py`, and `toy/toy5_periodic_particles.py`. Run them from the repository root in `pgflow`; Toy-5 uses CUDA if available. `toy/plot_toy1.py` generates the path/speed figure. Results are under `results/`; the proposal records experimental limitations and the initial unsuccessful Toy-5 solver settings. These are mechanism studies, not proof of CSP performance.

### 8-ring and two moons

Run `python toy/ring_moons.py` from the project root in `pgflow` (setting `OPENBLAS_NUM_THREADS=1` is recommended). It runs five paired seeds for straight, zero, scalar, fixed-axis anisotropic and repulsion paths, and writes `results/ring_moons.json` plus seed-0 plots. The Gaussian field is a fixed synthetic proxy, not a learned moon-normal metric; consult the proposal for results and limitations.

Run `python toy/toy4_fair_regularized.py` for the strict paired Toy-4 check. Each seed fixes endpoint pairs, time samples, random Fourier basis, ridge model and inference source/target across Straight, PG and regularized PG. The regularized path adds deviation-from-line, curvature and velocity-smoothing penalties; it writes `results/toy4_fair_regularized.json`. The included quick run uses 2 seeds and 20/10 endpoint pairs as a mechanism test, so it is not a final statistical claim.

The same script also exposes `straight_mac`, `pg_mac` and `pg_regularized_mac`: a two-stage MAC-style fit first fits the velocity model, keeps the lowest-error 70% training couplings, then refits with the same feature network. This is a local mechanism adaptation of `repos/2026_CVPR_MAC`, not the full EMA/OT implementation.

Run `conda run -n pgflow python toy/toy6_hessian_channel.py` for the Hessian-path toy. It uses an analytic two-dimensional cooperative-migration potential: Straight crosses a stiff exclusion core, while the positive-definite metric `I + alpha H(E)^T H(E)` seeks a soft channel. It reports both endpoint generation quality and physical quality along generated ODE trajectories; `results/toy6_hessian_channel.json` and `results/figures/toy6_hessian_channel_seed0.png` contain the three-seed results and visualization.

Run `conda run -n pgflow python toy/toy7_endpoint_hessian.py` for the endpoint-physics toy. It keeps Straight FM fixed and compares no residual, a frozen target-well force residual, and a frozen target-well Hessian residual on the one-step endpoint estimate. Its split-flow figures follow the QUBO coupling-study layout: source at left, reference and generated endpoints at right, purple transported trajectories, and a subtle endpoint-energy background. It writes `results/toy7_endpoint_hessian.json` and seed figures under `results/figures/`.
