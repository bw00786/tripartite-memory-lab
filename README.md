# Tripartite Memory Lab

Tripartite Memory Lab is a reproducible harness for functional memory
experiments. It includes a neuron-only associative memory, chemical-matrix
variants, and a deterministic simulation of agent-team memory protocols. The
original chemical-memory experiments relate to the hypothesis described by Marx
& Gilon (2026, *International Journal of Psychiatry Research*, DOI
[10.33425/2641-4317.1251](https://doi.org/10.33425/2641-4317.1251)).

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
python -m tml.cli run configs/sweep_cue_noise.yaml
python -m tml.cli run configs/sweep_diffusion.yaml
python -m tml.cli run configs/sweep_sparse.yaml
```

Run the agent-memory protocol comparison:

```powershell
python -m tml.cli run configs/agent_memory.yaml
```

Run the held-out lexical-cue task and its exact-key ablation:

```powershell
python -m tml.cli run configs/agent_memory_heldout.yaml
python -m tml.cli run configs/sweep_agent_heldout_policy.yaml
```

Run the local Ollama pilot with the preinstalled Gemma 3 4B model:

```powershell
ollama serve
ollama list
python -m tml.cli run configs/ollama_heldout_pilot.yaml
```

The pilot requires `gemma3:4b` to be installed in Ollama and its local API to
be reachable at `http://localhost:11434`. Model downloads are not automatic.
The config fixes the model tag, seed, temperature (0), context limit, output
limit, and task set. Output records the installed model digest, per-condition
inference calls, prompt/completion tokens, parse failures, and measured
latency. `api_cost_per_call_usd` is zero for this local run; machine energy and
hardware costs are not estimated. The small pilot has too few trials for
general conclusions; increase `trials` in a copied config for a fuller study.

Run its individual communication, access-limit, shared-memory-noise,
freshness-limit, or confidence-threshold intervention:

```powershell
python -m tml.cli run configs/sweep_agent_communication.yaml
python -m tml.cli run configs/sweep_agent_access.yaml
python -m tml.cli run configs/sweep_agent_noise.yaml
python -m tml.cli run configs/sweep_agent_freshness.yaml
python -m tml.cli run configs/sweep_agent_confidence.yaml
```

Run every YAML configuration and regenerate the CSVs, plots, and comparison
summary:

```powershell
python -m tml.cli all
```

This runs deterministic configs and skips Ollama configs by default. To include
the local model pilot:

```powershell
python -m tml.cli all --include-local-models
```

By default, generated files are written to `results/`, which is git-ignored:

- One CSV per configuration, containing means, 95% bootstrap confidence
  intervals, task metrics, trial counts, and seeds.
- One PNG per sweep, with confidence bands; associative-memory plots include a
  dashed neuron-only reference, while agent-memory plots compare memory setups.
- Configuration snapshots copied beside each CSV for exact run provenance.
- `summary.md` for the associative-memory experiments and
  `agent_memory_summary.md` for agent-memory results.

`python -m tml.cli run configs/sweep_noise.yaml --output custom/path.csv` writes
the selected CSV to another location; its plot and summary are written alongside
that file.

## Agent-memory benchmark

The deterministic agent-memory benchmark isolates memory and coordination
policies without calling an LLM. An optional Ollama backend runs the version-3
cue task against a local open-weight model. It is a separate, real-model pilot:
the same model tag, task instances, system instructions, and decoding settings
are used across memory setups, while the available memory and number of agent
calls vary by setup. Results record backend, model tag/digest, policy, task-set
version, configuration snapshot, and seed.

The included agent configurations use task-set version 2. In addition to the
original tasks, it varies correction-chain length and the number of distributed
clues per seeded trial. The minimum and maximum update/clue counts are explicit
configuration values. Task-set version 1 remains available for reproducing the
earlier fixed-size scenarios; `correction_chain` requires version 2.

The benchmark compares:

- **Single agent:** one agent stores all records.
- **Private memory:** records are distributed among agents; a coordinator can
  request relevant records subject to a communication-message budget.
- **Shared memory:** all agents access a common store.
- **Noisy shared memory:** a configured fraction of writes is corrupted.
- **Access-limited shared memory:** queries can inspect only the most recent
  configured number of records.

Tasks cover delayed fact retrieval, corrected/conflicting facts, multi-step
correction chains, combining clues distributed across private agents, and
abstaining on expired or low-confidence facts. Team sizes are configured in
`agent_memory.yaml`.
Separate sweep configs vary communication budget, shared-memory access limit,
shared-write corruption, freshness limit, and confidence threshold while
keeping the task seed and retrieval policy fixed.

The version-3 held-out cue task stores records under opaque IDs and asks for a
fact using two content terms rather than the stored ID. The configured
`cue_token_overlap` policy ranks records by exact token overlap; distractors
share one query term, while the target contains both. The config supplies the
lexicon and random seed, and each trial generates a new cue and distractor set.
The policy sweep compares this simple lexical rule with the exact-key ablation.
This tests retrieval beyond direct key equality, but it does not test semantic
understanding: the terms are explicit, the scoring rule is deterministic, and
the evaluation lexicon is provided to the simulator rather than held out from a
trained model.

Agent CSVs report task accuracy, exact task retrieval, stale, conflict, and
uncertainty error rates, uncertainty Brier score, measured wall-clock latency,
and deterministic compute units (records written + scanned + messages). Each
metric has a trial-bootstrap 95% confidence interval. CSVs also include the
retrieval policy and intervention value, plus paired accuracy and
exact-retrieval differences versus `single_agent`: both setups
are scored on the same seeded scenario for each trial, then the trial-level
differences are bootstrapped. The summary distinguishes intervals above zero,
below zero, or crossing zero; these intervals are not adjusted for multiple
comparisons. The held-out-policy sweep additionally reports paired differences
versus the exact-key policy on the same scenarios. `estimated_api_cost_usd` is
zero because this simulator makes no external model calls; it is not an
estimate of local electricity or hardware costs. Wall-clock latency depends on
the machine and is not expected to match across runs.

In the default seed-1, 150-trial run, single-agent and shared-memory setups
reached 1.000 accuracy on all five tasks at each configured team-size setting.
Private memory matched that except for five-agent distributed clues (0.943;
paired difference -0.057, 95% CI -0.075 to -0.040), where the communication
budget could not reach every agent holding clues. At the three-agent setting,
noisy shared memory scored 0.927 on correction chains and 0.902 on distributed
clues. Their paired accuracy differences from single-agent were -0.073 (95% CI
-0.120 to -0.033) and -0.098 (95% CI -0.125 to -0.072), respectively.
Access-limited shared memory scored 0.887 on correction chains and 0.534 on
distributed clues, with paired differences of -0.113 (95% CI -0.167 to -0.067)
and -0.466 (95% CI -0.502 to -0.431). In threshold sweeps, a freshness limit
of 20 caused stale facts to be returned (accuracy 0.5, stale-error rate 0.5),
while a confidence threshold of 0.3 caused low-confidence facts to be returned
(accuracy 0.5, uncertainty-error rate 0.5). These are outcomes of a
deterministic symbolic simulator whose policy reads explicit keys and metadata,
not measurements of deployed agents, language understanding, or general
reasoning. The generated summary includes confidence intervals and outcomes
for every intervention.

On the separate version-3 cue task (150 trials, three-agent setting), exact-key
retrieval scored 0.000, while token overlap scored 1.000 for single-agent and
shared-memory setups; the paired difference from exact-key was 1.000 (95% CI
1.000 to 1.000). Noisy shared memory scored 0.927 with a paired difference of
0.927 (95% CI 0.880 to 0.960). Private memory scored 0.367 and access-limited
shared memory 0.613. These deliberately controlled outcomes confirm the
lexical matching implementation on this synthetic task, not broad retrieval
generalization or semantic understanding.

## Model and experiment details

The YAML files specify all model, sweep, trial, and bootstrap parameters. The
standard settings are N=200 neurons, K=20 active neurons per stored memory,
D=16 signature dimensions, cue fraction 0.3, three recall iterations, 150
trials per setting, and seed 1.

- **Neuron-only:** Stores binary patterns with exactly K active neurons and
  uses `W = P.T @ P` with a zero diagonal. Each recall iteration multiplies by
  W, divides by the population standard deviation, and selects the top K
  neurons. Equal scores are resolved by neuron index.
- **CueMatch:** Scores every stored pattern by the number of active cue cells
  it contains and returns the highest-scoring stored pattern (ties select the
  earliest stored pattern). This is direct exemplar lookup, included as an
  algorithmic retrieval reference. It has access to the full pattern bank and
  is not an apples-to-apples comparison with the neuron-only or chemical
  dynamics models.
- **Cue corruption:** `cue_noise` replaces that fraction of the initially
  cued target cells with randomly selected cells absent from the target.
  `cue_noise=0` is an uncorrupted cue; `cue_noise=1` contains no target cells.
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
seeded NumPy random streams. Sweep points reuse matched memory banks and targets
so differences are not caused by changing the sampled memories or targets.
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

The CSV also reports exact-match rate and its 95% bootstrap interval. A trial
counts as an exact match only when the recalled binary pattern equals the full
target pattern. The cue-corruption sweep varies `cue_noise` while keeping the
cue size fixed, so it tests the effect of replacing correct cue cells with
non-target distractors.

At M=20, cue fraction 0.3, and seed 1, the cue-corruption sweep produced these
exact-pattern retrieval rates (95% bootstrap CIs):

| Fraction of cue cells replaced | Neuron-only | CueMatch | Matrix |
|---:|---:|---:|---:|
| 0 | 0.853 [0.800, 0.907] | 1.000 [1.000, 1.000] | 0.373 [0.293, 0.453] |
| 0.5 | 0.153 [0.100, 0.213] | 0.840 [0.780, 0.893] | 0.047 [0.013, 0.080] |
| 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

This task shows that direct exemplar lookup is more robust to these particular
corruptions than the tested recurrent recall methods. When all cue cells are
replaced, none of the models exactly retrieves the target. Some chemical
variants have slightly higher mean overlap than neuron-only at this 100%
corruption setting, but all have zero exact-match rate; this partial overlap
does not amount to successful retrieval. These results characterize this
synthetic task and its algorithms only.

With the included seed and configuration, the capacity sweep reproduced the
work-order reference values within ±0.05:

| Memories (M) | Neuron-only mean (95% CI) | Matrix mean (95% CI) |
|---:|---:|---:|
| 5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| 20 | 0.981 [0.969, 0.991] | 0.854 [0.827, 0.882] |
| 40 | 0.714 [0.687, 0.742] | 0.506 [0.478, 0.535] |

After adding CueMatch, it scored 1.000 mean recall at M=20 and M=40 in the
configured capacity sweep and 0.984 to 1.000 over cue fractions 0.1 to 0.5.
These high values mean that this seeded pattern/cue task usually identifies
the exact stored exemplar; they should not be interpreted as evidence that
CueMatch models human memory or any other cognitive property. Directly
returning a stored item differs from reconstructing one through distributed
weights.

Across the tested noise values 0, 1, 2, 4, and 8, the Matrix means were
0.854, 0.820, 0.778, 0.747, and 0.741, respectively, versus 0.981 for the
neuron-only model. In the cue sweep, Matrix mean recall ranged from 0.693 to
0.893, while neuron-only ranged from 0.838 to 0.990. The generated
`results/summary.md` compares mean overlap across the tested conditions. No
chemical-matrix variant exceeded neuron-only at the standard, capacity, cue
size, chemical-noise, diffusion, or signature-dimension settings. At 100% cue
corruption, some variants had a small positive difference in overlap, but
exact-match rates were zero for all models. SparseSignature improved as D
increased, reaching 0.929 at D=64, but did not exceed the neuron-only reference
(95% CI 0.908 to 0.950) at the standard cue setting. No positive
chemical-layer advantage was observed in the chemical-noise sweep, so no tested
noise breakpoint for such an advantage was found.

These are outcomes for the configured model and finite seeded trials, not
claims about untested parameter settings. Negative results are retained in the
CSV, plots, and summary.

## Scope and limitations

These are functional simulations, not biological measurements. Neurons,
signatures, local imprints, diffusion, and domain activation are simplified
mathematical assumptions; they do not establish that the modeled processes
occur in a brain. The agent-memory benchmark evaluates task performance of
fixed memory policies, not the behavior of deployed AI agents. Neither
experiment evaluates consciousness.

The cited authors state they have not shown that AI cannot be conscious. This
project does not make or test claims about consciousness.

## Tests and lint

```powershell
python -m pytest
python -m ruff check .
```
