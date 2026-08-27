"""Per-lag Drosophila structural-input conditional mutual information analysis.

This script characterizes the temporal profile of the primary empirical effect
using the same preprocessing, conditional-mutual-information estimator, weighted
T-bar structural connectivity, and strict structural surrogate model as the
primary analysis.

For each prediction lag, it reports:
    - observed conditional mutual information (CMI),
    - strict-null mean and standard deviation,
    - excess CMI above the null mean,
    - raw one-sided permutation p-value, and
    - max-statistic family-wise-error-rate (FWER) corrected p-value.

The strict structural null preserves:
    1. each region's outgoing weighted strength,
    2. each region's incoming weighted strength,
    3. edge density within each anatomical-distance bin, and
    4. total structural weight within each anatomical-distance bin.

Structural input is defined as

    U_i(t) = sum_j W[j, i] * tanh(x_j(t)),

where W[source, target] is weighted T-bar connectivity.

The pooled statistic at the end is included only as a consistency check against
the primary analysis; the purpose of this script is the lag-resolved result.
"""

import os
import glob
import numpy as np
import pandas as pd

from scipy import signal
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import pdist, squareform
import tifffile

from scfc import bridge
from scfc import anatomical_connectivity


# ============================================================
# PER-LAG TEMPORAL PROFILE
#
# Same data, preprocessing, structural connectivity, estimator,
# and strict structural null as the primary analysis.
#
# The primary result is the mean across five lags; this script
# characterizes the corresponding temporal profile.
#
# For each lag it reports:
#
#   observed CMI
#   strict-null mean CMI
#   excess CMI
#   raw one-sided permutation p
#
# It also reports a max-statistic FWER-corrected p across the
# five lags.
# ============================================================


FS = 1.2

LAGS = np.array(
    [1, 2, 4, 8, 16],
    dtype=int
)

THIN = 2

N_BINS_U = 3
N_BINS_X = 5
N_BINS_P = 3

N_SURR = 1000

N_DISTANCE_BINS = 10

MAX_BALANCE_ITER = 5000
BALANCE_TOL = 1e-8

MASTER_SEED = 20260826


# ============================================================
# PATHS
# ============================================================

DATA_DIR = bridge.getUserConfiguration()["data_dir"]

RESPONSE_DIR = os.path.join(
    DATA_DIR,
    "branson_responses"
)

ATLAS_FILE = os.path.join(
    DATA_DIR,
    "template_brains",
    "2018_999_atlas.tif"
)


# ============================================================
# PREPROCESSING
# ============================================================

def highpass_response(X):
    """Apply the Turner et al.-matched first-order 0.01-Hz high-pass filter."""

    sos = signal.butter(
        1,
        0.01,
        "hp",
        fs=FS,
        output="sos"
    )

    return signal.sosfilt(
        sos,
        X,
        axis=1
    )


def valid_segments(file_id, T):
    """Return retained frame intervals, excluding known recording artifacts."""

    if file_id == "branson_2018-10-19_1":

        return [
            (100, 900),
            (1100, 2000),
        ]

    elif file_id == "branson_2017-11-08_1":

        return [
            (100, 1900),
            (2000, 4000),
        ]

    elif file_id == "branson_2018-10-20_1":

        return [
            (100, 1000),
        ]

    else:

        return [
            (100, T),
        ]


def make_valid_pair_times(
    segments,
    lag,
    thin
):
    """Construct valid lagged sample times without crossing excluded intervals."""

    pieces = []

    for start, stop in segments:

        t = np.arange(
            start,
            stop - lag,
            thin,
            dtype=int
        )

        if len(t):
            pieces.append(t)

    if not pieces:

        return np.array(
            [],
            dtype=int
        )

    return np.concatenate(
        pieces
    )


# ============================================================
# DISCRETIZATION
# ============================================================

def quantile_edges(
    x,
    n_bins
):
    """Construct monotonically increasing quantile-bin edges."""

    x = np.asarray(
        x,
        dtype=float
    )

    edges = np.quantile(
        x,
        np.linspace(
            0,
            1,
            n_bins + 1
        )
    )

    edges[0] = -np.inf
    edges[-1] = np.inf

    for k in range(
        1,
        len(edges) - 1
    ):

        if edges[k] <= edges[k - 1]:

            edges[k] = np.nextafter(
                edges[k - 1],
                np.inf
            )

    return edges


def apply_bins(
    x,
    edges
):
    """Assign observations to precomputed bin edges."""

    return np.digitize(
        x,
        edges[1:-1],
        right=False
    ).astype(
        np.int16
    )


# ============================================================
# CONDITIONAL MUTUAL INFORMATION
# ============================================================

def conditional_mutual_information(
    U,
    Y,
    C,
    nU,
    nY,
    nC
):
    """Estimate discrete conditional mutual information I(U; Y | C) in bits."""

    U = np.asarray(
        U,
        dtype=np.int64
    )

    Y = np.asarray(
        Y,
        dtype=np.int64
    )

    C = np.asarray(
        C,
        dtype=np.int64
    )

    flat = (
        U
        + nU * Y
        + nU * nY * C
    )

    counts = np.bincount(
        flat,
        minlength=nU * nY * nC
    )

    counts = counts.reshape(
        nC,
        nY,
        nU
    ).transpose(
        2,
        1,
        0
    )

    total = counts.sum()

    if total == 0:

        return np.nan

    p = (
        counts.astype(float)
        / total
    )

    p_uc = p.sum(
        axis=1
    )

    p_yc = p.sum(
        axis=0
    )

    p_c = p.sum(
        axis=(0, 1)
    )

    u_idx, y_idx, c_idx = np.nonzero(
        p
    )

    vals = p[
        u_idx,
        y_idx,
        c_idx
    ]

    denom = (
        p_uc[
            u_idx,
            c_idx
        ]
        *
        p_yc[
            y_idx,
            c_idx
        ]
    )

    numer = (
        vals
        *
        p_c[
            c_idx
        ]
    )

    good = (
        (vals > 0)
        &
        (denom > 0)
        &
        (numer > 0)
    )

    return float(
        np.sum(
            vals[good]
            *
            np.log2(
                numer[good]
                /
                denom[good]
            )
        )
    )


# ============================================================
# ROI GEOMETRY
# ============================================================

def get_roi_centers(
    include_inds
):
    """Return atlas-space centers of mass for selected Branson parcels."""

    print(
        "Loading Branson atlas geometry..."
    )

    atlas = tifffile.imread(
        ATLAS_FILE
    )

    ones = np.ones(
        atlas.shape,
        dtype=np.uint8
    )

    centers = center_of_mass(
        ones,
        labels=atlas,
        index=list(include_inds)
    )

    centers = np.asarray(
        centers,
        dtype=float
    )

    if not np.all(
        np.isfinite(
            centers
        )
    ):

        raise RuntimeError(
            "Non-finite ROI center."
        )

    return centers


# ============================================================
# DISTANCE BINS
# ============================================================

def make_distance_bins(
    D,
    n_bins
):
    """Assign off-diagonal region pairs to quantile-based distance bins."""

    n = D.shape[0]

    offdiag = ~np.eye(
        n,
        dtype=bool
    )

    values = D[
        offdiag
    ]

    edges = np.quantile(
        values,
        np.linspace(
            0,
            1,
            n_bins + 1
        )
    )

    edges[0] = -np.inf
    edges[-1] = np.inf

    bin_id = np.full(
        D.shape,
        -1,
        dtype=np.int16
    )

    for b in range(
        n_bins
    ):

        if b < n_bins - 1:

            mask = (
                offdiag
                &
                (D >= edges[b])
                &
                (D < edges[b + 1])
            )

        else:

            mask = (
                offdiag
                &
                (D >= edges[b])
                &
                (D <= edges[b + 1])
            )

        bin_id[
            mask
        ] = b

    return bin_id


# ============================================================
# STRICT SURROGATE BALANCING
# ============================================================

def constraint_errors(
    W,
    row_target,
    col_target,
    distance_bin,
    bin_target
):

    row_now = W.sum(
        axis=1
    )

    col_now = W.sum(
        axis=0
    )

    row_error = np.max(
        np.abs(
            row_now - row_target
        )
        /
        np.maximum(
            np.abs(row_target),
            1e-12
        )
    )

    col_error = np.max(
        np.abs(
            col_now - col_target
        )
        /
        np.maximum(
            np.abs(col_target),
            1e-12
        )
    )

    bin_error = 0.0

    for b in range(
        N_DISTANCE_BINS
    ):

        current = W[
            distance_bin == b
        ].sum()

        target = bin_target[
            b
        ]

        err = (
            abs(current - target)
            /
            max(
                abs(target),
                1e-12
            )
        )

        bin_error = max(
            bin_error,
            err
        )

    return (
        float(row_error),
        float(col_error),
        float(bin_error),
    )


def balance_strict(
    W0,
    row_target,
    col_target,
    distance_bin,
    bin_target,
    max_iter=MAX_BALANCE_ITER,
    tol=BALANCE_TOL
):
    """Iteratively match row strengths, column strengths, and distance-bin weight."""

    W = np.asarray(
        W0,
        dtype=np.float64
    ).copy()

    n = W.shape[0]

    np.fill_diagonal(
        W,
        0.0
    )

    for iteration in range(
        max_iter
    ):

        # outgoing strength
        row_sum = W.sum(
            axis=1
        )

        factors = np.ones(
            n
        )

        good = row_sum > 0

        factors[
            good
        ] = (
            row_target[
                good
            ]
            /
            row_sum[
                good
            ]
        )

        W *= factors[
            :,
            None
        ]


        # incoming strength
        col_sum = W.sum(
            axis=0
        )

        factors = np.ones(
            n
        )

        good = col_sum > 0

        factors[
            good
        ] = (
            col_target[
                good
            ]
            /
            col_sum[
                good
            ]
        )

        W *= factors[
            None,
            :
        ]


        # distance-bin total weight
        for b in range(
            N_DISTANCE_BINS
        ):

            mask = (
                distance_bin == b
            )

            current = W[
                mask
            ].sum()

            target = bin_target[
                b
            ]

            if current > 0:

                W[
                    mask
                ] *= (
                    target
                    /
                    current
                )

        np.fill_diagonal(
            W,
            0.0
        )

        if (
            iteration % 10 == 0
            or
            iteration == max_iter - 1
        ):

            (
                row_err,
                col_err,
                bin_err
            ) = constraint_errors(
                W,
                row_target,
                col_target,
                distance_bin,
                bin_target
            )

            if max(
                row_err,
                col_err,
                bin_err
            ) < tol:

                return W

    return W


def make_surrogate_W(
    W_real,
    distance_bin,
    rng
):
    """Generate one distance-binned, strength-balanced structural surrogate."""

    W0 = np.zeros_like(
        W_real,
        dtype=np.float64
    )

    for b in range(
        N_DISTANCE_BINS
    ):

        inds = np.where(
            distance_bin == b
        )

        vals = W_real[
            inds
        ].copy()

        W0[
            inds
        ] = rng.permutation(
            vals
        )

    np.fill_diagonal(
        W0,
        0.0
    )

    row_target = W_real.sum(
        axis=1
    )

    col_target = W_real.sum(
        axis=0
    )

    bin_target = np.asarray(
        [
            W_real[
                distance_bin == b
            ].sum()

            for b in range(
                N_DISTANCE_BINS
            )
        ],
        dtype=float
    )

    return balance_strict(
        W0,
        row_target,
        col_target,
        distance_bin,
        bin_target
    )


# ============================================================
# PREPARE ONE RECORDING
# ============================================================

def prepare_fly(
    filepath,
    include_inds,
    active_structural
):
    """Load, filter, standardize, and prepare one functional recording."""

    file_id = os.path.basename(
        filepath
    ).replace(
        ".pkl",
        ""
    )

    raw = pd.read_pickle(
        filepath
    )

    X_full = raw.reindex(
        include_inds
    ).to_numpy(
        dtype=np.float64
    )

    X_full = highpass_response(
        X_full
    )

    T = X_full.shape[1]

    segments = valid_segments(
        file_id,
        T
    )

    X_full = X_full[
        active_structural,
        :
    ]

    valid_roi = np.all(
        np.isfinite(
            X_full
        ),
        axis=1
    )

    X_full = X_full[
        valid_roi,
        :
    ]

    valid_frames = np.concatenate(
        [
            np.arange(
                start,
                stop,
                dtype=int
            )

            for start, stop
            in segments
        ]
    )

    mu = np.mean(
        X_full[
            :,
            valid_frames
        ],
        axis=1,
        keepdims=True
    )

    sd = np.std(
        X_full[
            :,
            valid_frames
        ],
        axis=1,
        keepdims=True
    )

    sd[
        sd == 0
    ] = 1.0

    Z = (
        (X_full - mu)
        /
        sd
    ).astype(
        np.float32
    )

    PHI = np.tanh(
        Z
    ).astype(
        np.float32
    )

    P = np.mean(
        Z,
        axis=0
    ).astype(
        np.float32
    )

    print(
        f"{file_id:25s}"
        f" | ROIs={Z.shape[0]:3d}"
        f" | T={T:4d}"
        f" | valid={len(valid_frames):4d}"
    )

    return {
        "file_id": file_id,
        "Z": Z,
        "PHI": PHI,
        "P": P,
        "segments": segments,
        "valid_roi": valid_roi,
    }


# ============================================================
# CMI CURVE FOR ONE RECORDING
# ============================================================

def fly_cmi_curve(
    fly,
    W_active
):
    """Compute the five-lag CMI curve for one recording and one connectome."""

    valid_roi = fly[
        "valid_roi"
    ]

    W = W_active[
        valid_roi,
        :
    ][
        :,
        valid_roi
    ]

    Z = fly[
        "Z"
    ]

    PHI = fly[
        "PHI"
    ]

    P = fly[
        "P"
    ]

    N = Z.shape[0]


    # W[source, target]
    # U[target, t] = sum_source W[source, target] * phi(source, t)
    U = (
        W.T
        @
        PHI
    ).astype(
        np.float32
    )


    valid_frames = np.concatenate(
        [
            np.arange(
                start,
                stop,
                dtype=int
            )

            for start, stop
            in fly[
                "segments"
            ]
        ]
    )


    mu_u = np.mean(
        U[
            :,
            valid_frames
        ],
        axis=1,
        keepdims=True
    )

    sd_u = np.std(
        U[
            :,
            valid_frames
        ],
        axis=1,
        keepdims=True
    )

    sd_u[
        sd_u == 0
    ] = 1.0

    U = (
        (U - mu_u)
        /
        sd_u
    ).astype(
        np.float32
    )


    curve = []


    for lag in LAGS:

        times = make_valid_pair_times(
            fly[
                "segments"
            ],
            lag,
            THIN
        )

        present = Z[
            :,
            times
        ].reshape(-1)

        future = Z[
            :,
            times + lag
        ].reshape(-1)

        structural_input = U[
            :,
            times
        ].reshape(-1)

        population = np.tile(
            P[
                times
            ],
            N
        )


        present_bin = apply_bins(
            present,
            quantile_edges(
                present,
                N_BINS_X
            )
        )

        future_bin = apply_bins(
            future,
            quantile_edges(
                future,
                N_BINS_X
            )
        )

        input_bin = apply_bins(
            structural_input,
            quantile_edges(
                structural_input,
                N_BINS_U
            )
        )

        population_bin = apply_bins(
            population,
            quantile_edges(
                population,
                N_BINS_P
            )
        )

        condition = (
            present_bin
            +
            N_BINS_X
            *
            population_bin
        )

        I = conditional_mutual_information(
            input_bin,
            future_bin,
            condition,
            N_BINS_U,
            N_BINS_X,
            N_BINS_X
            *
            N_BINS_P
        )

        curve.append(
            I
        )

    return np.asarray(
        curve,
        dtype=float
    )


# ============================================================
# LOAD EMPIRICAL STRUCTURAL CONNECTOME
# ============================================================

print(
    "\n"
    + "=" * 76
)

print(
    "STRICT FLY PER-LAG STRUCTURAL-INPUT CMI"
)

print(
    "=" * 76
)


include_inds, names = (
    bridge.getBransonNames()
)


W_df = (
    anatomical_connectivity.getAtlasConnectivity(
        include_inds,
        names,
        "branson",
        metric="weighted_tbar"
    )
)


W_full = W_df.to_numpy(
    dtype=np.float64
)

np.fill_diagonal(
    W_full,
    0.0
)


out_strength = W_full.sum(
    axis=1
)

in_strength = W_full.sum(
    axis=0
)


active_structural = (
    (out_strength > 0)
    &
    (in_strength > 0)
)


W_real = W_full[
    active_structural,
    :
][
    :,
    active_structural
]


print(
    "Original ROIs:",
    len(names)
)

print(
    "Active ROIs:",
    W_real.shape[0]
)


# ============================================================
# GEOMETRY
# ============================================================

centers_full = get_roi_centers(
    include_inds
)

centers = centers_full[
    active_structural,
    :
]

D = squareform(
    pdist(
        centers
    )
)

distance_bin = make_distance_bins(
    D,
    N_DISTANCE_BINS
)


# ============================================================
# LOAD AND PREPARE FUNCTIONAL RECORDINGS
# ============================================================

response_files = sorted(
    glob.glob(
        os.path.join(
            RESPONSE_DIR,
            "*.pkl"
        )
    )
)


print(
    "\nRecordings:",
    len(response_files)
)

print(
    "\nPreparing recordings..."
)


flies = [
    prepare_fly(
        fp,
        include_inds,
        active_structural
    )

    for fp in response_files
]


# ============================================================
# OBSERVED PER-LAG CMI CURVE
# ============================================================

print(
    "\n"
    + "=" * 76
)

print(
    "OBSERVED PER-LAG CMI"
)

print(
    "=" * 76
)


real_recording_curves = np.vstack(
    [
        fly_cmi_curve(
            fly,
            W_real
        )

        for fly in flies
    ]
)


real_curve = np.mean(
    real_recording_curves,
    axis=0
)


for lag_s, value in zip(
    LAGS / FS,
    real_curve
):

    print(
        f"tau={lag_s:6.3f}s"
        f" | observed CMI={value:.6f}"
    )


# ============================================================
# STRICT NULL: FULL FIVE-LAG CURVE FOR EACH SURROGATE
# ============================================================

print(
    "\n"
    + "=" * 76
)

print(
    "GENERATING 1000 STRICT NULL CURVES"
)

print(
    "=" * 76
)


rng = np.random.default_rng(
    MASTER_SEED
)


null_curves = np.zeros(
    (
        N_SURR,
        len(LAGS)
    ),
    dtype=float
)


for q in range(
    N_SURR
):

    Wq = make_surrogate_W(
        W_real,
        distance_bin,
        rng
    )


    curves_q = np.vstack(
        [
            fly_cmi_curve(
                fly,
                Wq
            )

            for fly in flies
        ]
    )


    null_curves[
        q,
        :
    ] = np.mean(
        curves_q,
        axis=0
    )


    if (
        q == 0
        or
        (q + 1) % 10 == 0
    ):

        print(
            f"surrogate {q+1:4d}/{N_SURR}"
            " | "
            +
            " ".join(
                f"{x:.6f}"
                for x in null_curves[
                    q
                ]
            )
        )


# ============================================================
# PER-LAG STATISTICS
# ============================================================

null_mean_curve = np.mean(
    null_curves,
    axis=0
)

null_sd_curve = np.std(
    null_curves,
    axis=0,
    ddof=1
)

excess_curve = (
    real_curve
    -
    null_mean_curve
)


# Empirical one-sided permutation p-values use the standard +1 correction.
raw_p = np.zeros(
    len(LAGS),
    dtype=float
)


for k in range(
    len(LAGS)
):

    raw_p[
        k
    ] = (
        1
        +
        np.sum(
            null_curves[
                :,
                k
            ]
            >=
            real_curve[
                k
            ]
        )
    ) / (
        N_SURR + 1
    )


# ============================================================
# MAX-STATISTIC FWER CORRECTION
#
# Null statistic is each surrogate's largest excess above the
# lag-specific null mean.
#
# Each observed lag is compared with this common distribution.
# ============================================================

null_excess = (
    null_curves
    -
    null_mean_curve[
        None,
        :
    ]
)


null_max = np.max(
    null_excess,
    axis=1
)


# Compare every observed lag-specific excess with the common null maximum.
fwer_p = np.zeros(
    len(LAGS),
    dtype=float
)


for k in range(
    len(LAGS)
):

    observed_excess = excess_curve[
        k
    ]

    fwer_p[
        k
    ] = (
        1
        +
        np.sum(
            null_max
            >=
            observed_excess
        )
    ) / (
        N_SURR
        +
        1
    )


# ============================================================
# REPORT RESULTS
# ============================================================

print(
    "\n"
    + "=" * 76
)

print(
    "STRICT PER-LAG RESULTS"
)

print(
    "=" * 76
)


print(
    "\n"
    "tau(s) | observed | null mean | null SD "
    "| excess | raw p | FWER p"
)


for k, lag in enumerate(
    LAGS
):

    print(
        f"{lag/FS:6.3f}"
        f" | {real_curve[k]:.6f}"
        f" | {null_mean_curve[k]:.6f}"
        f" | {null_sd_curve[k]:.6f}"
        f" | {excess_curve[k]:+.6f}"
        f" | {raw_p[k]:.4f}"
        f" | {fwer_p[k]:.4f}"
    )


# ============================================================
# POOLED CHECK
#
# This should reproduce the pooled primary result up to numerical precision.
# ============================================================

real_pooled = float(
    np.mean(
        real_curve
    )
)


null_pooled = np.mean(
    null_curves,
    axis=1
)


null_pooled_mean = float(
    np.mean(
        null_pooled
    )
)


pooled_excess = (
    real_pooled
    -
    null_pooled_mean
)


pooled_p = (
    1
    +
    np.sum(
        null_pooled
        >=
        real_pooled
    )
) / (
    N_SURR
    +
    1
)


print(
    "\n"
    + "-" * 76
)

print(
    "POOLED CONSISTENCY CHECK"
)

print(
    "-" * 76
)


print(
    "Observed pooled CMI:",
    f"{real_pooled:.6f}"
)

print(
    "Strict-null pooled CMI:",
    f"{null_pooled_mean:.6f}"
)

print(
    "Pooled excess:",
    f"{pooled_excess:+.6f}"
)

print(
    "Pooled permutation p:",
    f"{pooled_p:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

np.savez_compressed(
    "fly_strict_per_lag_results.npz",

    file_ids=np.asarray(
        [
            fly[
                "file_id"
            ]
            for fly in flies
        ]
    ),

    lags_frames=
        LAGS,

    lags_seconds=
        LAGS / FS,

    real_recording_curves=
        real_recording_curves,

    real_curve=
        real_curve,

    null_curves=
        null_curves,

    null_mean_curve=
        null_mean_curve,

    null_sd_curve=
        null_sd_curve,

    excess_curve=
        excess_curve,

    raw_p=
        raw_p,

    fwer_p=
        fwer_p,

    null_max=
        null_max,

    real_pooled=np.array(
        real_pooled
    ),

    null_pooled=
        null_pooled,

    null_pooled_mean=np.array(
        null_pooled_mean
    ),

    pooled_excess=np.array(
        pooled_excess
    ),

    pooled_p=np.array(
        pooled_p
    ),

    structural_metric=np.array(
        "weighted_tbar"
    ),

    null_type=np.array(
        "strict_strength_distance_bin"
    ),

    n_surrogates=np.array(
        N_SURR
    ),

    n_distance_bins=np.array(
        N_DISTANCE_BINS
    ),

    master_seed=np.array(
        MASTER_SEED
    ),
)


print(
    "\nSaved: "
    "fly_strict_per_lag_results.npz"
)

print(
    "\n"
    + "=" * 76
)

print(
    "DONE"
)

print(
    "=" * 76
)
