# Tripartite Memory Lab

Tripartite Memory Lab is a reproducible experiment harness for comparing
associative recall from a neuron-only memory model with recall from models that
add a chemical-matrix channel. It explores the functional consequences of the
hypothesis described by Marx & Gilon (2026, *International Journal of Psychiatry
Research*, DOI [10.33425/2641-4317.1251](https://doi.org/10.33425/2641-4317.1251)):
recall, interference, and robustness to chemical noise.

These experiments test recall behavior of a modeled mechanism; they do not bear on whether any system is conscious.

## Install

Requires Python 3.11 or newer. From the repository root:

```powershell
python -m pip install -e ".[dev]"
```

NumPy, SciPy, pandas, Matplotlib, and PyYAML are runtime dependencies. The
development extra installs pytest and Ruff.

## Run experiments

Run the baseline configuration:

```powershell
python -m tml.cli run configs/baseline.yaml
```

Run one of the individual sweeps:

```powershell
python -m tml.cli run configs/sweep_capacity.yaml
python -m tml.cli run configs/sweep_noise.yaml
python -m tml.cli run configs/sweep_cue.yaml
python -m tml.cli run configs/sweep_diffusion.yaml
python -m tml.cli run configs/sweep_sparse.yaml
```

Run every YAML configuration and regenerate the CSVs, plots, and comparison
summary:

```powershell
python -m tml.cli all
```

By default, generated files are written to `results/`, which is git-ignored:

- One CSV per configuration, containing means, 95% bootstrap confidence
  intervals, interference estimates, trial counts, and seeds.
- One PNG per sweep, with confidence bands and a dashed neuron-only reference.
- `summary.md`, which lists variants that did not beat the neuron-only model as
  well as any conditions where a variant's mean was higher.

`python -m tml.cli run configs/sweep_noise.yaml --output custom/path.csv` writes
the selected CSV to another location; its plot and summary are written alongside
that file.

## Model and experiment details

The YAML files specify all model, sweep, trial, and bootstrap parameters. The
standard settings are N=200 neurons, K=20 active neurons per stored memory,
D=16 signature dimensions, cue fraction 0.3, three recall iterations, 150
trials per setting, and seed 1.

- **Neuron-only:** Stores binary patterns with exactly K active neurons and
  uses `W = P.T @ P` with a zero diagonal. Each recall iteration multiplies by
  W, divides by the population standard deviation, and selects the top K
  neurons. Equal scores are resolved by neuron index.
- **Matrix:** Assigns each memory a bipolar signature and builds a local
  imprint `T = P.T @ S`. Optional Gaussian blur is scaled by
  `noise * sqrt(M*K/N)`. Each iteration binds the active cells' imprints,
  normalizes the chemical scores, adds them with weight `alpha`, and selects
  the top K.
- **LocalDomain:** Restricts chemical binding to neuron domains and applies it
  only when at least `theta` active cells are present in a domain.
- **Diffusion:** Spreads the imprint across a square 2D neuron layout with a
  row-normalized Gaussian kernel of width `sigma`.
- **SparseSignature:** Uses sparse ternary signatures in `{-1, 0, +1}`; the
  `density` parameter specifies their nonzero fraction.
- **Gated:** Adds the chemical channel only when the synaptic top-K score margin
  is below `margin_threshold`.
- **NoAstrocyte:** Ablates LocalDomain's domain thresholding and uses the
  un-gated matrix chemical channel.

Each trial gets a reproducible memory bank, signatures, target, and cue from
seeded NumPy random streams. Sweep points reuse matched trial streams so
differences are not caused by changing the sampled memories or targets.
Capacity settings use the same seeded pattern sequence, with the smaller
memory banks forming prefixes of the larger ones. Bootstrap resampling uses a
separate deterministic stream. The CSV's `ci_low` and `ci_high` columns are
percentile bootstrap intervals for mean target overlap. Interference is the
maximum overlap with any non-target stored pattern.

## Reading the results

Recall overlap is the fraction of target cells active after recall. Higher is
better; 1.0 is a complete target match. The plots compare mean overlap over
trials, and the shaded bands show the 95% bootstrap intervals. The dashed line
is the neuron-only reference. Confidence-interval overlap is not itself a
statistical test of the difference between models.

With the included seed and configuration, the capacity sweep reproduced the
work-order reference values within ±0.05:

| Memories (M) | Neuron-only mean (95% CI) | Matrix mean (95% CI) |
|---:|---:|---:|
| 5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| 20 | 0.981 [0.969, 0.991] | 0.854 [0.826, 0.882] |
| 40 | 0.714 [0.687, 0.742] | 0.506 [0.478, 0.535] |

Across the tested noise values 0, 1, 2, 4, and 8, the Matrix means were
0.854, 0.820, 0.778, 0.747, and 0.741, respectively, versus 0.981 for the
neuron-only model. In the cue sweep, Matrix mean recall ranged from 0.693 to
0.893, while neuron-only ranged from 0.838 to 0.990. The generated
`results/summary.md` reports that no tested variant beat the neuron-only model
in mean recall in any tested condition. In particular, SparseSignature
improved as D increased, reaching 0.929 at D=64, but did not exceed the
neuron-only reference. No noise level with a positive chemical-layer advantage
was observed, so there is no tested noise breakpoint at which such an
advantage disappears.

These are outcomes for the configured model and finite seeded trials, not
claims about untested parameter settings. Negative results are retained in the
CSV, plots, and summary.

## Scope and limitations

These are functional simulations, not biological measurements. Neurons,
signatures, local imprints, diffusion, and domain activation are simplified
mathematical assumptions; they do not establish that the modeled processes
occur in a brain. The experiments evaluate recall and interference only.

The cited authors state they have not shown that AI cannot be conscious. This
project does not make or test claims about consciousness.

## Tests and lint

```powershell
python -m pytest
python -m ruff check .
```
