# Structure–Dynamics Analysis

Code accompanying the empirical and *in silico* analyses of structurally informed prediction of neural activity.

The empirical analysis asks whether anatomically weighted network activity contains information about a region's future activity beyond that available from its current activity and the contemporaneous population state.

## Repository contents

### Empirical Drosophila analyses

**`fly_primary_structural_input_cmi_STRICT.py`**  
Primary empirical analysis using weighted T-bar structural connectivity. Structural input is

```text
U_i(t) = sum_j W[j, i] * tanh(X_j(t))
```

with `W[source, target]`. The empirical connectome is compared with 1,000 structural surrogates that preserve each region's incoming and outgoing weighted strength, edge density within anatomical-distance bins, and total structural weight within each distance bin.

**`fly_primary_structural_input_cmi_CELLCOUNT.py`**  
Robustness analysis repeating the primary test using cell-count connectivity, defined as the number of neurons linking each source-target region pair.

**`fly_strict_per_lag.py`**  
Lag-resolved analysis of the weighted-T-bar result at 1, 2, 4, 8, and 16 imaging frames. It reports observed CMI, null mean and SD, excess CMI, one-sided permutation p-values, and max-statistic FWER-corrected p-values across the five prediction horizons.

### In-silico analysis

Add the cleaned simulation script here if the *in silico* analysis is being released in the same repository.

## External data and code

The empirical analyses use the Drosophila structural-connectivity and whole-brain functional-imaging data associated with:

Turner MH, Mann K, Clandinin TR. *The connectome predicts resting-state functional connectivity across the Drosophila brain.* Current Biology (2021).

The scripts rely on utilities from the Turner et al. SC-FC codebase, which must be available as the Python package/module `scfc`:

```text
https://github.com/mhturner/SC-FC
```

The Turner source code is not duplicated in this repository.

## Required inputs

The SC-FC configuration must define `data_dir`. Relative to that directory, the empirical scripts expect at least:

```text
branson_responses/*.pkl
template_brains/2018_999_atlas.tif
```

Structural-connectivity matrices are loaded through the SC-FC `anatomical_connectivity` utilities.

The analyses use the fine-grained Branson atlas representation. Regions without both incoming and outgoing structural connectivity are excluded before analysis.

## Python dependencies

The empirical scripts require:

```text
numpy
pandas
scipy
tifffile
```

and the external `scfc` package/codebase described above.

## Running the empirical analyses

From the directory containing the scripts:

```bash
python fly_primary_structural_input_cmi_STRICT.py
python fly_primary_structural_input_cmi_CELLCOUNT.py
python fly_strict_per_lag.py
```

Each script uses the fixed random seed `20260826` for surrogate generation.

The analyses can be computationally intensive because each structural null consists of 1,000 surrogate connectomes evaluated across all functional recordings.

## Analysis summary

Functional signals are high-pass filtered at 0.01 Hz using a first-order Butterworth filter. The first 100 frames and recording-specific artifact intervals are excluded, and lagged samples are constructed separately within retained contiguous intervals. Regional signals are standardized within recording.

For each target region, structurally weighted input is computed after applying a saturating `tanh` nonlinearity to standardized source-region activity. Conditional mutual information quantifies information about future target activity contained in structural input while conditioning on current target activity and the contemporaneous population state.

The strict structural null randomizes connectivity within ten anatomical-distance bins and then iteratively balances each surrogate to recover the empirical incoming strength, outgoing strength, and total weight in each distance bin. Because zero-valued entries are included in the within-bin permutation and multiplicative balancing does not create new nonzero entries, edge density within each distance bin is also preserved.

The lag-resolved analysis applies a one-sided empirical permutation test at each prediction horizon and controls family-wise error across the five horizons using a max-statistic permutation procedure.

## Outputs

The scripts save compressed NumPy result files:

```text
fly_primary_structural_input_cmi_STRICT_results.npz
fly_primary_structural_input_cmi_CELLCOUNT_results.npz
fly_strict_per_lag_results.npz
```

These generated result files do not need to be committed to the repository unless you specifically want to archive the numerical outputs alongside the code.

## Reproducibility notes

The empirical scripts preserve the preprocessing and atlas conventions used in the source SC-FC analysis while implementing the conditional-information and structural-surrogate analyses reported in the accompanying manuscript.

No raw Drosophila data or Turner SC-FC source files are redistributed here. Users should obtain those materials from their original sources and configure the local SC-FC installation accordingly.
