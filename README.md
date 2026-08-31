# Structure–Dynamics Analysis

Code accompanying the empirical and *in silico* analyses of structurally informed prediction of neural activity.

The empirical analysis asks whether anatomically weighted network activity contains information about a region's future activity beyond that available from its current activity and the contemporaneous population state.

## Repository contents

### Empirical *Drosophila* analyses

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

### *In silico* analysis

**`simulate_structure_dynamics.m`**  
Reproduces the synthetic-network analysis comparing the nonlinear recurrent model with a matched linear recurrent control and a structural-drive control. The script generates the circular structural networks, simulates stochastic trajectories, computes conditional mutual information, applies spatially constrained structural-label surrogates, and reports the model comparisons described in the manuscript.

## External data and code

The empirical analyses use the *Drosophila* structural-connectivity and whole-brain functional-imaging data associated with:

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

## Dependencies

### Python

The empirical scripts require:

```text
numpy
pandas
scipy
tifffile
```

as well as the external `scfc` package/codebase described above.

### MATLAB

The *in silico* analysis requires MATLAB. The simulation script uses standard MATLAB functionality and the Statistics and Machine Learning Toolbox for the nonparametric statistical tests.

## Running the analyses

### Empirical analyses

From the directory containing the scripts:

```bash
python fly_primary_structural_input_cmi_STRICT.py
python fly_primary_structural_input_cmi_CELLCOUNT.py
python fly_strict_per_lag.py
```

Each empirical script uses the fixed random seed `20260826` for surrogate generation.

The analyses can be computationally intensive because each structural null consists of 1,000 surrogate connectomes evaluated across all functional recordings.

### *In silico* analysis

In MATLAB, run:

```matlab
simulate_structure_dynamics
```

The script generates 30 independently sampled circular networks with five stochastic realizations per network and compares the nonlinear recurrent, matched linear recurrent, and structural-drive models.

## Analysis summary

### Empirical analysis

Functional signals are high-pass filtered at 0.01 Hz using a first-order Butterworth filter. The first 100 frames and recording-specific artifact intervals are excluded, and lagged samples are constructed separately within retained contiguous intervals. Regional signals are standardized within recording.

For each target region, structurally weighted input is computed after applying a saturating `tanh` nonlinearity to standardized source-region activity:

```text
U_i(t) = sum_j W[j, i] * tanh(X_j(t))
```

Conditional mutual information is then used to quantify information about future target activity contained in structural input while conditioning on current target activity and the contemporaneous population state.

The analysis uses three quantile bins for structural input, five quantile bins for regional activity, and three quantile bins for population activity. Lagged observations are thinned by a factor of two before CMI estimation.

For each recording and prediction horizon, observations are pooled across regions and retained time points to estimate a single CMI value. Lag-specific CMI is averaged across recordings, while the overall empirical statistic is averaged across both recordings and the five prediction horizons. The same aggregation procedure is applied independently to each surrogate connectome.

The strict structural null randomizes connectivity within ten anatomical-distance bins and then iteratively balances each surrogate to recover the empirical incoming strength, outgoing strength, and total weight in each distance bin. Because zero-valued entries are included in the within-bin permutation and multiplicative balancing does not create new nonzero entries, edge density within each distance bin is also preserved.

The lag-resolved analysis applies a one-sided empirical permutation test at each prediction horizon and controls family-wise error across the five horizons using a max-statistic permutation procedure.

### *In silico* analysis

The synthetic model uses circular networks of 60 equally spaced regions with symmetric, non-negative distance-dependent structural connectivity and additional node-strength and pairwise heterogeneity. Signed source efficacy is specified separately, with 30% of regions assigned negative efficacy.

Dynamics are integrated with a time step of `dt = 0.01` for 300 time units, with the first 100 discarded as burn-in. Additive Gaussian noise has amplitude 0.20. Global coupling is normalized so that the spectral radius of the effective coupling matrix is 2.5.

Information-theoretic analysis uses samples separated by 0.20 time units and a prediction horizon of 0.20 time units. Information samples are additionally thinned by a factor of five. Regional activity is standardized within region and realization. Structural position is defined by weighted anatomical node strength and discretized into three quantile bins; activity is discretized into five quantile bins.

Spatially constrained surrogate structural maps are generated by rotations and reflections of the structural coordinate around the circular network, preserving its spatial organization while disrupting the region-to-structure assignment.

## Outputs

The empirical scripts save compressed NumPy result files:

```text
fly_primary_structural_input_cmi_STRICT_results.npz
fly_primary_structural_input_cmi_CELLCOUNT_results.npz
fly_strict_per_lag_results.npz
```

These generated result files do not need to be committed to the repository unless you specifically want to archive the numerical outputs alongside the code.

The MATLAB script reports the simulation results in the MATLAB workspace/command window and may save outputs according to the options defined in the script.

## Reproducibility notes

The empirical scripts preserve the preprocessing and atlas conventions used in the source SC-FC analysis while implementing the conditional-information and structural-surrogate analyses reported in the accompanying manuscript.

No raw *Drosophila* data or Turner SC-FC source files are redistributed here. Users should obtain those materials from their original sources and configure the local SC-FC installation accordingly.
