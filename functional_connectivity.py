"""
Functional-connectivity utilities adapted from the Turner, Mann & Clandinin
SC-FC codebase.

Original project:
    https://github.com/mhturner/SC-FC

These functions support preprocessing of regional fluorescence traces,
functional-connectivity estimation, and atlas geometry calculations.

The numerical behavior of the original implementation is preserved here; edits
are limited to formatting, documentation, and replacement of a deprecated
SciPy import path.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import pdist


def filterRegionResponse(region_response, cutoff=None, fs=None):
    """High-pass filter regional response traces.

    Parameters
    ----------
    region_response : np.ndarray
        Regional response array.
    cutoff : float or None
        High-pass cutoff frequency in Hz.
    fs : float or None
        Sampling frequency in Hz. If None, no filtering is applied.

    Returns
    -------
    np.ndarray
        Filtered regional responses.
    """
    if fs is not None:
        sos = signal.butter(
            1,
            cutoff,
            "hp",
            fs=fs,
            output="sos",
        )
        return signal.sosfilt(sos, region_response)

    return region_response


def trimRegionResponse(
    file_id,
    region_response,
    start_include=100,
    end_include=None,
):
    """Remove known artifacts and initial frames from a recording.

    Parameters
    ----------
    file_id : str
        Recording identifier.
    region_response : np.ndarray
        Either an ``n_rois x n_frames`` response matrix or a one-dimensional
        behavioral response trace.
    start_include : int
        Default first frame to retain for recordings without a specific rule.
    end_include : int or None
        Default final frame to retain.

    Returns
    -------
    np.ndarray
        Trimmed response array.
    """
    # Recording-specific retained frame indices from Turner et al.
    brains_to_trim = {
        # Transient dropout spikes.
        "2018-10-19_1": np.array(
            list(range(100, 900)) + list(range(1100, 2000))
        ),
        # Baseline shift.
        "2017-11-08_1": np.array(
            list(range(100, 1900)) + list(range(2000, 4000))
        ),
        # Dropout halfway through recording.
        "2018-10-20_1": np.array(
            list(range(100, 1000))
        ),
    }

    if file_id in brains_to_trim:
        include_inds = brains_to_trim[file_id]
        if region_response.ndim == 2:
            return region_response[:, include_inds]
        if region_response.ndim == 1:
            return region_response[include_inds]

    # Default trimming for all other recordings.
    if region_response.ndim == 2:
        return region_response[:, start_include:end_include]
    if region_response.ndim == 1:
        return region_response[start_include:end_include]

    raise ValueError(
        "region_response must be one- or two-dimensional."
    )


def getProcessedRegionResponse(resp_fp, cutoff=None, fs=None):
    """Load, high-pass filter, and trim regional response traces.

    Parameters
    ----------
    resp_fp : str
        Path to a pickled pandas DataFrame containing regional responses.
    cutoff : float or None
        High-pass cutoff frequency in Hz.
    fs : float or None
        Sampling frequency in Hz.

    Returns
    -------
    pandas.DataFrame
        Processed regional response traces.
    """
    file_id = resp_fp.split("/")[-1].replace(".pkl", "")

    region_responses = pd.read_pickle(resp_fp)

    resp = filterRegionResponse(
        region_responses.to_numpy(),
        cutoff=cutoff,
        fs=fs,
    )
    resp = trimRegionResponse(file_id, resp)

    return pd.DataFrame(
        data=resp,
        index=region_responses.index,
    )


def computeRegionResponses(brain, region_masks):
    """Compute the mean fluorescence trace within each atlas-region mask.

    Parameters
    ----------
    brain : np.ndarray
        Four-dimensional ``x, y, z, time`` brain array.
    region_masks : sequence of np.ndarray
        Boolean three-dimensional masks, one per region.

    Returns
    -------
    np.ndarray
        Matrix of regional response traces with shape
        ``n_regions x n_frames``.
    """
    region_responses = [
        np.mean(brain[mask, :], axis=0)
        for mask in region_masks
    ]

    return np.vstack(region_responses)


def getCmat(response_filepaths, include_inds, name_list):
    """Compute the mean functional-connectivity matrix across recordings.

    Regional traces are preprocessed with a 0.01-Hz high-pass filter at
    1.2-Hz sampling, matching the original SC-FC workflow. Pearson
    correlation matrices are Fisher-z transformed before averaging.

    Parameters
    ----------
    response_filepaths : sequence of str
        Paths to pickled regional-response files.
    include_inds : sequence
        ROI identifiers to retain.
    name_list : sequence of str
        Names corresponding to ``include_inds``.

    Returns
    -------
    CorrelationMatrix : pandas.DataFrame
        Mean Fisher-z-transformed functional-connectivity matrix.
    cmats_z : list of np.ndarray
        Fisher-z-transformed matrix for each recording.
    """
    cmats_z = []

    for resp_fp in response_filepaths:
        processed = getProcessedRegionResponse(
            resp_fp,
            cutoff=0.01,
            fs=1.2,
        )

        resp_included = processed.reindex(
            include_inds
        ).to_numpy()

        correlation_matrix = np.corrcoef(
            resp_included
        )

        np.fill_diagonal(
            correlation_matrix,
            np.nan,
        )

        # Fisher z transform.
        cmat_z = np.arctanh(
            correlation_matrix
        )
        cmats_z.append(cmat_z)

    mean_cmat = np.nanmean(
        np.stack(cmats_z, axis=2),
        axis=2,
    )
    np.fill_diagonal(
        mean_cmat,
        np.nan,
    )

    correlation_matrix = pd.DataFrame(
        data=mean_cmat,
        index=name_list,
        columns=name_list,
    )

    return correlation_matrix, cmats_z


def getMeanBrain(filepath):
    """Load and return a time-averaged brain volume."""
    return np.asanyarray(
        nib.load(filepath).dataobj
    ).astype("uint16")


def loadAtlasData(
    atlas_path,
    include_inds,
    name_list,
):
    """Load atlas masks for selected regions.

    Parameters
    ----------
    atlas_path : str
        Path to the atlas image.
    include_inds : sequence
        ROI identifiers to retain.
    name_list : sequence of str
        Names corresponding to ``include_inds``.

    Returns
    -------
    list of np.ndarray
        Boolean atlas mask for each requested region.
    """
    mask_brain = np.asarray(
        np.squeeze(
            nib.load(atlas_path).get_fdata()
        ),
        dtype="uint16",
    )

    roi_masks = []

    for r_ind, _ in enumerate(name_list):
        roi_masks.append(
            mask_brain == include_inds[r_ind]
        )

    return roi_masks


def getRegionGeometry(
    atlas_path,
    include_inds,
    name_list,
):
    """Compute atlas-region centers, sizes, distances, and pairwise size scale.

    Parameters
    ----------
    atlas_path : str
        Path to the atlas image.
    include_inds : sequence
        ROI identifiers to retain.
    name_list : sequence of str
        Names corresponding to ``include_inds``.

    Returns
    -------
    coms : np.ndarray
        Region centers of mass.
    roi_size : list
        Number of voxels in each region.
    DistanceMatrix : pandas.DataFrame
        Symmetric Euclidean distance matrix between region centers.
    SizeMatrix : pandas.DataFrame
        Geometric mean of region sizes for each pair.
    """
    roi_masks = loadAtlasData(
        atlas_path,
        include_inds,
        name_list,
    )

    roi_size = [
        mask.sum()
        for mask in roi_masks
    ]

    coms = np.vstack(
        [
            center_of_mass(mask)
            for mask in roi_masks
        ]
    )

    # Euclidean distance matrix between region centers of mass.
    dist_mat = np.zeros(
        (
            len(roi_masks),
            len(roi_masks),
        )
    )

    upper = np.triu_indices(
        len(roi_masks),
        k=1,
    )
    dist_mat[upper] = pdist(coms)
    dist_mat += dist_mat.T

    distance_matrix = pd.DataFrame(
        data=dist_mat,
        index=name_list,
        columns=name_list,
    )

    # Geometric mean of the sizes of each ROI pair.
    size_mat = np.sqrt(
        np.outer(
            np.asarray(roi_size),
            np.asarray(roi_size),
        )
    )

    size_matrix = pd.DataFrame(
        data=size_mat,
        index=name_list,
        columns=name_list,
    )

    return (
        coms,
        roi_size,
        distance_matrix,
        size_matrix,
    )
