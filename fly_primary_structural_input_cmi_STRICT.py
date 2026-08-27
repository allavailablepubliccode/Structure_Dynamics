"""
Primary Drosophila structural-input conditional mutual information analysis.

This script tests whether anatomically weighted network activity contains
information about a region's future activity beyond that available from its
current activity and the contemporaneous population state.

Structural input is defined as

    U_i(t) = sum_j W[j, i] * tanh(X_j(t)),

where W[source, target] is weighted T-bar connectivity.

The empirical connectome is compared with a strict surrogate ensemble that
preserves:
    1. outgoing weighted strength of every region,
    2. incoming weighted strength of every region,
    3. edge density within each distance bin, and
    4. total connection weight within each distance bin.

The analysis uses the Branson atlas representation and the preprocessing
conventions of Turner et al. (2021).

Expected local project structure
--------------------------------
The SC-FC package must be importable as ``scfc`` and its configuration must
define ``data_dir``. Under that directory this script expects:

    branson_responses/*.pkl
    template_brains/2018_999_atlas.tif

Output
------
fly_primary_structural_input_cmi_STRICT_results.npz
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import signal
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import pdist, squareform

from scfc import anatomical_connectivity, bridge


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------

FS = 1.2
LAGS = np.array([1, 2, 4, 8, 16], dtype=int)
THIN = 2

N_BINS_U = 3
N_BINS_X = 5
N_BINS_P = 3

N_SURR = 1000
N_DISTANCE_BINS = 10

MAX_BALANCE_ITER = 5000
BALANCE_TOL = 1e-8

MASTER_SEED = 20260826

OUTPUT_FILE = "fly_primary_structural_input_cmi_STRICT_results.npz"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def highpass_response(X: np.ndarray) -> np.ndarray:
    """Apply the first-order 0.01-Hz Butterworth high-pass filter."""
    sos = signal.butter(1, 0.01, "hp", fs=FS, output="sos")
    return signal.sosfilt(sos, X, axis=1)


def valid_segments(file_id: str, T: int) -> list[tuple[int, int]]:
    """
    Return retained contiguous frame intervals for one recording.

    The first 100 frames are removed from all recordings. The three
    recording-specific exclusions reproduce the artifact trimming used in the
    source SC-FC analysis. Keeping intervals separate prevents lagged pairs
    from crossing excluded gaps.
    """
    if file_id == "branson_2018-10-19_1":
        return [(100, 900), (1100, 2000)]

    if file_id == "branson_2017-11-08_1":
        return [(100, 1900), (2000, 4000)]

    if file_id == "branson_2018-10-20_1":
        return [(100, 1000)]

    return [(100, T)]


def make_valid_pair_times(
    segments: list[tuple[int, int]],
    lag: int,
    thin: int,
) -> np.ndarray:
    """Return present-time indices whose future partner remains in-segment."""
    chunks = [
        np.arange(start, stop - lag, thin, dtype=int)
        for start, stop in segments
        if stop - lag > start
    ]
    if not chunks:
        return np.array([], dtype=int)
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Discretization and conditional mutual information
# ---------------------------------------------------------------------------

def quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Construct robust quantile-bin edges."""
    x = np.asarray(x, dtype=float)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Ensure strictly increasing internal edges in the presence of ties.
    for k in range(1, len(edges) - 1):
        if edges[k] <= edges[k - 1]:
            edges[k] = np.nextafter(edges[k - 1], np.inf)

    return edges


def apply_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign observations to integer-valued bins."""
    return np.digitize(x, edges[1:-1], right=False).astype(np.int16)


def conditional_mutual_information(
    U: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    nU: int,
    nY: int,
    nC: int,
) -> float:
    """
    Estimate discrete conditional mutual information I(U; Y | C), in bits.

    The estimator is the plug-in estimate obtained from the empirical joint
    histogram of U, Y, and C.
    """
    U = np.asarray(U, dtype=np.int64)
    Y = np.asarray(Y, dtype=np.int64)
    C = np.asarray(C, dtype=np.int64)

    flat = U + nU * Y + nU * nY * C
    counts = np.bincount(flat, minlength=nU * nY * nC)
    counts = counts.reshape(nC, nY, nU).transpose(2, 1, 0)

    total = counts.sum()
    if total == 0:
        return np.nan

    p = counts.astype(float) / total
    p_uc = p.sum(axis=1)
    p_yc = p.sum(axis=0)
    p_c = p.sum(axis=(0, 1))

    u_idx, y_idx, c_idx = np.nonzero(p)
    vals = p[u_idx, y_idx, c_idx]

    denom = p_uc[u_idx, c_idx] * p_yc[y_idx, c_idx]
    numer = vals * p_c[c_idx]

    good = (vals > 0) & (denom > 0) & (numer > 0)
    information = np.sum(vals[good] * np.log2(numer[good] / denom[good]))
    return float(information)


# ---------------------------------------------------------------------------
# Atlas geometry and distance bins
# ---------------------------------------------------------------------------

def get_roi_centers(
    atlas_file: str,
    include_inds: np.ndarray,
) -> np.ndarray:
    """Return atlas-space centers of mass for the selected Branson parcels."""
    print("Loading Branson atlas geometry...")
    atlas = tifffile.imread(atlas_file)

    centers = center_of_mass(
        np.ones(atlas.shape, dtype=np.uint8),
        labels=atlas,
        index=list(include_inds),
    )
    centers = np.asarray(centers, dtype=float)

    if not np.all(np.isfinite(centers)):
        raise RuntimeError(
            "Could not recover finite centers for all selected Branson ROIs."
        )

    return centers


def make_distance_bins(D: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each off-diagonal region pair to an equal-count distance bin."""
    n = D.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    values = D[offdiag]

    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    bin_id = np.full(D.shape, -1, dtype=np.int16)

    for b in range(n_bins):
        if b < n_bins - 1:
            mask = offdiag & (D >= edges[b]) & (D < edges[b + 1])
        else:
            mask = offdiag & (D >= edges[b]) & (D <= edges[b + 1])
        bin_id[mask] = b

    if np.any(bin_id[offdiag] < 0):
        raise RuntimeError("At least one off-diagonal pair was not distance-binned.")

    return bin_id


# ---------------------------------------------------------------------------
# Strict structural surrogate
# ---------------------------------------------------------------------------

def constraint_errors(
    W: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    distance_bin: np.ndarray,
    bin_target: np.ndarray,
) -> tuple[float, float, float]:
    """Return maximum relative errors in the three balancing constraints."""
    row_now = W.sum(axis=1)
    col_now = W.sum(axis=0)

    row_error = np.max(
        np.abs(row_now - row_target) / np.maximum(np.abs(row_target), 1e-12)
    )
    col_error = np.max(
        np.abs(col_now - col_target) / np.maximum(np.abs(col_target), 1e-12)
    )

    bin_error = 0.0
    for b in range(N_DISTANCE_BINS):
        current = W[distance_bin == b].sum()
        target = bin_target[b]
        err = abs(current - target) / max(abs(target), 1e-12)
        bin_error = max(bin_error, err)

    return float(row_error), float(col_error), float(bin_error)


def balance_strict(
    W0: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    distance_bin: np.ndarray,
    bin_target: np.ndarray,
    max_iter: int = MAX_BALANCE_ITER,
    tol: float = BALANCE_TOL,
) -> tuple[np.ndarray, int, float, float, float]:
    """
    Iteratively balance a surrogate to the empirical network constraints.

    Multiplicative scaling is alternated across rows, columns, and distance
    bins until outgoing strength, incoming strength, and distance-bin total
    weight all meet ``tol``.
    """
    W = np.asarray(W0, dtype=np.float64).copy()
    n = W.shape[0]
    np.fill_diagonal(W, 0.0)

    for iteration in range(max_iter):
        # Match outgoing weighted strength.
        row_sum = W.sum(axis=1)
        factors = np.ones(n, dtype=float)
        good = row_sum > 0
        factors[good] = row_target[good] / row_sum[good]
        W *= factors[:, None]

        # Match incoming weighted strength.
        col_sum = W.sum(axis=0)
        factors = np.ones(n, dtype=float)
        good = col_sum > 0
        factors[good] = col_target[good] / col_sum[good]
        W *= factors[None, :]

        # Match total weight in each distance bin.
        for b in range(N_DISTANCE_BINS):
            mask = distance_bin == b
            current = W[mask].sum()
            target = bin_target[b]
            if current > 0:
                W[mask] *= target / current

        np.fill_diagonal(W, 0.0)

        # Checking every 10 iterations avoids unnecessary overhead.
        if iteration % 10 == 0 or iteration == max_iter - 1:
            row_err, col_err, bin_err = constraint_errors(
                W,
                row_target,
                col_target,
                distance_bin,
                bin_target,
            )
            if max(row_err, col_err, bin_err) < tol:
                return W, iteration + 1, row_err, col_err, bin_err

    row_err, col_err, bin_err = constraint_errors(
        W,
        row_target,
        col_target,
        distance_bin,
        bin_target,
    )
    return W, max_iter, row_err, col_err, bin_err


def make_surrogate_W(
    W_real: np.ndarray,
    distance_bin: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float, float, float]:
    """
    Generate one strict spatially and strength-constrained surrogate.

    Empirical entries, including zeros, are first permuted only among region
    pairs belonging to the same distance bin. Because multiplicative balancing
    cannot create a nonzero value from a zero entry, this preserves the number
    of zero/nonzero entries within each distance bin. The shuffled matrix is
    then balanced to recover empirical incoming strength, outgoing strength,
    and total weight in every distance bin.
    """
    W0 = np.zeros_like(W_real, dtype=np.float64)

    for b in range(N_DISTANCE_BINS):
        inds = np.where(distance_bin == b)
        vals = W_real[inds].copy()
        W0[inds] = rng.permutation(vals)

    np.fill_diagonal(W0, 0.0)

    row_target = W_real.sum(axis=1)
    col_target = W_real.sum(axis=0)
    bin_target = np.asarray(
        [W_real[distance_bin == b].sum() for b in range(N_DISTANCE_BINS)],
        dtype=float,
    )

    return balance_strict(
        W0,
        row_target,
        col_target,
        distance_bin,
        bin_target,
    )


# ---------------------------------------------------------------------------
# Functional data
# ---------------------------------------------------------------------------

def prepare_fly(
    filepath: str,
    include_inds: np.ndarray,
    active_structural: np.ndarray,
) -> dict:
    """Load, filter, trim, standardize, and transform one recording."""
    basename = os.path.basename(filepath)
    file_id = basename.replace(".pkl", "")

    raw = pd.read_pickle(filepath)
    X_full = raw.reindex(include_inds).to_numpy(dtype=np.float64)

    # Match the original functional preprocessing.
    X_full = highpass_response(X_full)
    T = X_full.shape[1]
    segments = valid_segments(file_id, T)

    # Restrict to regions with nonzero incoming and outgoing structural weight.
    X_full = X_full[active_structural, :]

    # One recording contains a completely non-finite parcel; remove such
    # parcels recording-by-recording and subset W identically downstream.
    valid_roi = np.all(np.isfinite(X_full), axis=1)
    X_full = X_full[valid_roi, :]

    valid_frames = np.concatenate(
        [np.arange(start, stop, dtype=int) for start, stop in segments]
    )

    # Standardize each regional trace over retained frames only.
    mu = np.mean(X_full[:, valid_frames], axis=1, keepdims=True)
    sd = np.std(X_full[:, valid_frames], axis=1, keepdims=True)
    sd[sd == 0] = 1.0

    Z = ((X_full - mu) / sd).astype(np.float32)

    # Same saturating nonlinearity as the theoretical/simulation model.
    PHI = np.tanh(Z).astype(np.float32)

    # Contemporaneous whole-population state used as a conditioning variable.
    P = np.mean(Z, axis=0).astype(np.float32)

    print(
        f"{file_id:25s}"
        f" | ROIs={Z.shape[0]:3d}"
        f" | raw T={T:4d}"
        f" | valid frames={len(valid_frames):4d}"
    )

    return {
        "file_id": file_id,
        "Z": Z,
        "PHI": PHI,
        "P": P,
        "segments": segments,
        "valid_roi": valid_roi,
    }


def fly_cmi_curve(fly: dict, W_active: np.ndarray) -> np.ndarray:
    """Compute the five-lag CMI curve for one recording and one connectome."""
    valid_roi = fly["valid_roi"]
    W = W_active[valid_roi, :][:, valid_roi]

    Z = fly["Z"]
    PHI = fly["PHI"]
    P = fly["P"]
    N = Z.shape[0]

    # W[source, target], hence target-wise structural input is W.T @ PHI.
    U = (W.T @ PHI).astype(np.float32)

    # Standardize structural input separately for each target region over
    # retained frames.
    valid_frames = np.concatenate(
        [
            np.arange(start, stop, dtype=int)
            for start, stop in fly["segments"]
        ]
    )
    mu_u = np.mean(U[:, valid_frames], axis=1, keepdims=True)
    sd_u = np.std(U[:, valid_frames], axis=1, keepdims=True)
    sd_u[sd_u == 0] = 1.0
    U = ((U - mu_u) / sd_u).astype(np.float32)

    curve = []

    for lag in LAGS:
        times = make_valid_pair_times(fly["segments"], lag, THIN)
        if times.size == 0:
            raise RuntimeError(
                f"No valid sample pairs for {fly['file_id']} at lag={lag}."
            )

        # Pool region-time observations within this recording.
        present = Z[:, times].reshape(-1)
        future = Z[:, times + lag].reshape(-1)
        structural_input = U[:, times].reshape(-1)

        # np.tile is correct for row-major flattening above: all times for ROI
        # 1, then all times for ROI 2, etc.
        population = np.tile(P[times], N)

        present_bin = apply_bins(
            present,
            quantile_edges(present, N_BINS_X),
        )
        future_bin = apply_bins(
            future,
            quantile_edges(future, N_BINS_X),
        )
        input_bin = apply_bins(
            structural_input,
            quantile_edges(structural_input, N_BINS_U),
        )
        population_bin = apply_bins(
            population,
            quantile_edges(population, N_BINS_P),
        )

        # Joint conditioning state C = (present-state bin, population-state bin).
        condition = present_bin + N_BINS_X * population_bin

        information = conditional_mutual_information(
            input_bin,
            future_bin,
            condition,
            N_BINS_U,
            N_BINS_X,
            N_BINS_X * N_BINS_P,
        )
        curve.append(information)

    return np.asarray(curve, dtype=float)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the weighted-T-bar strict-null empirical analysis."""
    data_dir = bridge.getUserConfiguration()["data_dir"]
    response_dir = os.path.join(data_dir, "branson_responses")
    atlas_file = os.path.join(
        data_dir,
        "template_brains",
        "2018_999_atlas.tif",
    )

    print("\n" + "=" * 76)
    print("FINAL STRICT FLY STRUCTURAL-INPUT CMI")
    print("=" * 76)

    include_inds, names = bridge.getBransonNames()

    W_df = anatomical_connectivity.getAtlasConnectivity(
        include_inds,
        names,
        "branson",
        metric="weighted_tbar",
    )
    W_full = W_df.to_numpy(dtype=np.float64)

    # Self-connections do not contribute to the cross-regional structural input.
    np.fill_diagonal(W_full, 0.0)

    out_strength = W_full.sum(axis=1)
    in_strength = W_full.sum(axis=0)
    active_structural = (out_strength > 0) & (in_strength > 0)

    print("Original Branson ROIs:", len(names))
    print("Structurally active ROIs:", int(active_structural.sum()))

    W_real = W_full[active_structural, :][:, active_structural]

    # Atlas geometry for the structurally active parcels.
    centers_full = get_roi_centers(atlas_file, include_inds)
    centers = centers_full[active_structural, :]
    D = squareform(pdist(centers))
    distance_bin = make_distance_bins(D, N_DISTANCE_BINS)

    # Load functional recordings.
    response_files = sorted(
        glob.glob(os.path.join(response_dir, "*.pkl"))
    )
    if not response_files:
        raise FileNotFoundError(
            f"No Branson response .pkl files found in: {response_dir}"
        )

    print("\nRecordings:", len(response_files))
    print("\nPreparing functional data...")

    flies = [
        prepare_fly(fp, include_inds, active_structural)
        for fp in response_files
    ]

    # -----------------------------------------------------------------------
    # Empirical connectome
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("OBSERVED REAL-CONNECTOME CMI")
    print("=" * 76)

    real_curves = []

    for fly in flies:
        curve = fly_cmi_curve(fly, W_real)
        real_curves.append(curve)

        print(
            f"{fly['file_id']:25s}"
            f" | mean CMI={curve.mean():.6f}"
            " | curve="
            + " ".join(f"{x:.6f}" for x in curve)
        )

    real_curves = np.vstack(real_curves)
    real_recording_mean = np.mean(real_curves, axis=1)
    real_pooled = float(np.mean(real_recording_mean))

    print(
        "\nMean observed CMI across recordings and lags:",
        f"{real_pooled:.6f}",
        "bits",
    )

    # -----------------------------------------------------------------------
    # Strict structural surrogate ensemble
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("GENERATING STRICT STRUCTURAL NULL")
    print("=" * 76)

    rng = np.random.default_rng(MASTER_SEED)

    null_pooled = np.zeros(N_SURR, dtype=float)
    surrogate_row_error = np.zeros(N_SURR, dtype=float)
    surrogate_col_error = np.zeros(N_SURR, dtype=float)
    surrogate_bin_error = np.zeros(N_SURR, dtype=float)
    surrogate_iterations = np.zeros(N_SURR, dtype=int)

    for q in range(N_SURR):
        (
            W_q,
            iterations,
            row_err,
            col_err,
            bin_err,
        ) = make_surrogate_W(W_real, distance_bin, rng)

        surrogate_iterations[q] = iterations
        surrogate_row_error[q] = row_err
        surrogate_col_error[q] = col_err
        surrogate_bin_error[q] = bin_err

        recording_values = [
            float(fly_cmi_curve(fly, W_q).mean())
            for fly in flies
        ]
        null_pooled[q] = float(np.mean(recording_values))

        if q == 0 or (q + 1) % 10 == 0:
            print(
                f"surrogate {q + 1:4d}/{N_SURR}"
                f" | pooled CMI={null_pooled[q]:.6f}"
                f" | iter={iterations:3d}"
                f" | row={row_err:.2e}"
                f" | col={col_err:.2e}"
                f" | bin={bin_err:.2e}"
            )

    # -----------------------------------------------------------------------
    # Primary pooled statistic
    # -----------------------------------------------------------------------
    null_mean = float(np.mean(null_pooled))
    null_sd = float(np.std(null_pooled, ddof=1))
    excess = real_pooled - null_mean

    n_exceed = int(np.sum(null_pooled >= real_pooled))
    p_value = (1 + n_exceed) / (N_SURR + 1)

    lags_seconds = LAGS / FS

    print("\n" + "=" * 76)
    print("FINAL STRICT PRIMARY FLY RESULT")
    print("=" * 76)

    print("Fixed lags (s):", np.round(lags_seconds, 3))
    print("\nObserved mean CMI:", f"{real_pooled:.6f}", "bits")
    print("Strict-null mean CMI:", f"{null_mean:.6f}", "bits")
    print("Strict-null SD:", f"{null_sd:.6f}", "bits")
    print("Excess predictive information:", f"{excess:+.6f}", "bits")
    print("Surrogates >= observed:", f"{n_exceed}/{N_SURR}")
    print("One-sided permutation p:", f"{p_value:.4f}")

    print(
        "\nMaximum outgoing-strength error:",
        f"{np.max(surrogate_row_error):.3e}",
    )
    print(
        "Maximum incoming-strength error:",
        f"{np.max(surrogate_col_error):.3e}",
    )
    print(
        "Maximum distance-bin weight error:",
        f"{np.max(surrogate_bin_error):.3e}",
    )
    print(
        "Maximum balancing iterations:",
        int(np.max(surrogate_iterations)),
    )

    if p_value < 0.05 and excess > 0:
        print(
            "\nRESULT: REAL CONNECTOME PROVIDES MORE PREDICTIVE INFORMATION "
            "THAN THE STRICT SPATIAL/STRENGTH-CONSTRAINED NULL."
        )
    else:
        print(
            "\nRESULT: NO RELIABLE EXCESS PREDICTIVE INFORMATION FOR THE "
            "REAL CONNECTOME UNDER THE STRICT NULL."
        )

    # -----------------------------------------------------------------------
    # Save compact numerical output
    # -----------------------------------------------------------------------
    np.savez_compressed(
        OUTPUT_FILE,
        file_ids=np.asarray([fly["file_id"] for fly in flies]),
        lags_frames=LAGS,
        lags_seconds=lags_seconds,
        real_curves=real_curves,
        real_recording_mean=real_recording_mean,
        real_pooled=np.array(real_pooled),
        null_pooled=null_pooled,
        null_mean=np.array(null_mean),
        null_sd=np.array(null_sd),
        excess=np.array(excess),
        n_exceed=np.array(n_exceed),
        p_value=np.array(p_value),
        active_structural=np.asarray(active_structural),
        surrogate_row_error=surrogate_row_error,
        surrogate_col_error=surrogate_col_error,
        surrogate_bin_error=surrogate_bin_error,
        surrogate_iterations=surrogate_iterations,
        structural_metric=np.array("weighted_tbar"),
        null_type=np.array("strict_strength_distance_bin"),
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print("\n" + "=" * 76)
    print("DONE")
    print("=" * 76)


if __name__ == "__main__":
    main()
