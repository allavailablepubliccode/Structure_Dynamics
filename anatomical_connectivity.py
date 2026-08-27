"""
Structural-connectivity utilities adapted from the Turner, Mann & Clandinin
SC-FC codebase.

Original project:
    https://github.com/mhturner/SC-FC

NeuPrint references:
    https://connectome-neuprint.github.io/neuprint-python/docs/index.html
    https://github.com/connectome-neuprint/neuprint-python

The numerical behavior of the original implementation is preserved here.
Edits are limited to formatting, documentation, naming clarity, and explicit
description of matrix orientation.

Matrix convention
-----------------
Rows correspond to source regions and columns correspond to target regions:

    W[source, target]

This is the convention used by the empirical structural-input analyses.
"""

import os

import numpy as np
import pandas as pd
from neuprint import NeuronCriteria, fetch_neurons

from . import bridge


def getAtlasConnectivity(
    include_inds,
    name_list,
    atlas_id,
    metric="cellcount",
):
    """Load an atlas-level structural-connectivity matrix.

    Parameters
    ----------
    include_inds : sequence of int
        Atlas ROI identifiers to retain.
    name_list : sequence of str
        Region names corresponding to ``include_inds``.
    atlas_id : {"branson", "ito"}
        Atlas representation.
    metric : {"cellcount", "tbar", "weighted_tbar"}
        Structural-connectivity measure.

    Returns
    -------
    pandas.DataFrame
        Directed connectivity matrix with rows as source regions and columns
        as target regions.
    """
    data_dir = bridge.getUserConfiguration()["data_dir"]

    if atlas_id == "branson":
        filepath = os.path.join(
            data_dir,
            "hemi_2_atlas",
            f"JRC2018_branson_{metric}_matrix.csv",
        )
        full_matrix = pd.read_csv(
            filepath,
            header=0,
        ).to_numpy()[:, 1:]

        full_matrix = pd.DataFrame(
            data=full_matrix,
            index=np.arange(1, 1000),
            columns=np.arange(1, 1000),
        )

    elif atlas_id == "ito":
        filepath = os.path.join(
            data_dir,
            "hemi_2_atlas",
            f"JRC2018_ito_{metric}_matrix.csv",
        )
        full_matrix = pd.read_csv(
            filepath,
            header=0,
        ).to_numpy()[:, 1:]

        full_matrix = pd.DataFrame(
            data=full_matrix,
            index=np.arange(1, 87),
            columns=np.arange(1, 87),
        )

    else:
        raise ValueError(
            "atlas_id must be either 'branson' or 'ito'."
        )

    # Select and order the requested regions.
    connectivity = pd.DataFrame(
        data=np.zeros(
            (
                len(include_inds),
                len(include_inds),
            )
        ),
        index=name_list,
        columns=name_list,
    )

    for source_index, source_roi in enumerate(include_inds):
        for target_index, target_roi in enumerate(include_inds):
            connectivity.iloc[
                source_index,
                target_index,
            ] = full_matrix.loc[
                source_roi,
                target_roi,
            ]

    return connectivity


def getRoiCompleteness(
    neuprint_client,
    name_list,
):
    """Return pre-, post-, and combined completeness for Ito atlas regions.

    Parameters
    ----------
    neuprint_client
        Active NeuPrint client.
    name_list : sequence of str
        Ito atlas region names.

    Returns
    -------
    pandas.DataFrame
        Columns are ``frac_pre``, ``frac_post``, and their product
        ``completeness``.
    """
    completeness_neuprint = (
        neuprint_client.fetch_roi_completeness()
    )
    completeness_neuprint.index = (
        completeness_neuprint["roi"]
    )

    completeness = np.zeros(
        (
            len(name_list),
            2,
        )
    )

    for region_index, roi in enumerate(name_list):
        current_rois = completeness_neuprint.loc[
            bridge.ito_to_neuprint(roi),
            :,
        ]

        completeness[
            region_index,
            0,
        ] = (
            current_rois["roipre"].sum()
            / current_rois["totalpre"].sum()
        )

        completeness[
            region_index,
            1,
        ] = (
            current_rois["roipost"].sum()
            / current_rois["totalpost"].sum()
        )

    roi_completeness = pd.DataFrame(
        data=completeness,
        index=name_list,
        columns=[
            "frac_pre",
            "frac_post",
        ],
    )

    roi_completeness["completeness"] = (
        roi_completeness["frac_pre"]
        * roi_completeness["frac_post"]
    )

    return roi_completeness


def computeConnectivityMatrix(
    neuprint_client,
    mapping,
):
    """Compute region-to-region connectivity measures from NeuPrint.

    Parameters
    ----------
    neuprint_client
        Active NeuPrint client. Retained for compatibility with the original
        SC-FC interface.
    mapping : dict
        Mapping from atlas regions to one or more hemibrain ROI names.

    Returns
    -------
    WeakConnections : pandas.DataFrame
        Number of neurons with fewer than 3 postsynaptic inputs in the source
        ROI and at least one presynaptic output in the target ROI.
    MediumConnections : pandas.DataFrame
        Number of neurons with 3--9 postsynaptic inputs in the source ROI and
        at least one presynaptic output in the target ROI.
    StrongConnections : pandas.DataFrame
        Number of neurons with at least 10 postsynaptic inputs in the source
        ROI and at least one presynaptic output in the target ROI.
    Connectivity : pandas.DataFrame
        Sum across connecting neurons of
        ``sqrt(source postsynapses * target T-bars)``.
    WeightedSynapseNumber : pandas.DataFrame
        Sum across connecting neurons of target-region T-bars weighted by the
        fraction of that neuron's total postsynaptic inputs that lie in the
        source region.
    TBars : pandas.DataFrame
        Total number of presynaptic T-bars in the target region contributed by
        connecting neurons.
    body_ids : np.ndarray
        Unique NeuPrint body IDs contributing to at least one region pair.

    Notes
    -----
    Returned matrices use the convention ``matrix[source, target]``.
    """
    # ``neuprint_client`` is retained in the signature for compatibility with
    # the original project. ``fetch_neurons`` uses the configured global
    # NeuPrint client.
    _ = neuprint_client

    rois = sorted(
        mapping.keys()
    )

    shape = (
        len(rois),
        len(rois),
    )

    weak_connections = pd.DataFrame(
        data=np.zeros(shape),
        index=rois,
        columns=rois,
    )
    medium_connections = pd.DataFrame(
        data=np.zeros(shape),
        index=rois,
        columns=rois,
    )
    strong_connections = pd.DataFrame(
        data=np.zeros(shape),
        index=rois,
        columns=rois,
    )
    connectivity = pd.DataFrame(
        data=np.zeros(shape),
        index=rois,
        columns=rois,
    )
    weighted_synapse_number = pd.DataFrame(
        data=np.zeros(shape),
        index=rois,
        columns=rois,
    )
    tbars = pd.DataFrame(
        data=np.zeros(shape),
        index=rois,
        columns=rois,
    )

    # Track all cells contributing to at least one atlas-region connection.
    body_ids = []

    for roi_source in rois:
        for roi_target in rois:
            sources = mapping[
                roi_source
            ]
            targets = mapping[
                roi_target
            ]

            weak_neurons = 0
            medium_neurons = 0
            strong_neurons = 0

            summed_connectivity = 0
            weighted_synapses_total = 0
            tbar_total = 0

            # Multiple hemibrain source/target ROIs may be collapsed into one
            # atlas-level region by ``mapping``.
            for source_roi in sources:
                for target_roi in targets:
                    neurons, _ = fetch_neurons(
                        NeuronCriteria(
                            inputRois=source_roi,
                            outputRois=target_roi,
                            status="Traced",
                            cropped=False,
                        )
                    )

                    # Presynaptic outputs in target region.
                    outputs_in_target = np.array(
                        [
                            info[target_roi]["pre"]
                            for info in neurons.roiInfo
                        ]
                    )

                    # Postsynaptic inputs in source region.
                    inputs_in_source = np.array(
                        [
                            info[source_roi]["post"]
                            for info in neurons.roiInfo
                        ]
                    )

                    n_weak = np.sum(
                        np.logical_and(
                            outputs_in_target > 0,
                            inputs_in_source < 3,
                        )
                    )

                    n_medium = np.sum(
                        np.logical_and(
                            outputs_in_target > 0,
                            np.logical_and(
                                inputs_in_source >= 3,
                                inputs_in_source < 10,
                            ),
                        )
                    )

                    n_strong = np.sum(
                        np.logical_and(
                            outputs_in_target > 0,
                            inputs_in_source >= 10,
                        )
                    )

                    # Cell-wise connection-strength measure:
                    # sqrt(input PSDs in source * output T-bars in target).
                    connection_strengths = [
                        np.sqrt(
                            info[target_roi]["pre"]
                            * info[source_roi]["post"]
                        )
                        for info in neurons.roiInfo
                    ]

                    # Weighted T-bar measure used in the primary empirical
                    # analysis:
                    #
                    # target T-bars * (
                    #     source-region postsynaptic inputs
                    #     / total postsynaptic inputs onto that neuron
                    # )
                    weighted_synapses = [
                        (
                            neurons.roiInfo[index][target_roi]["pre"]
                            * (
                                neurons.roiInfo[index][source_roi]["post"]
                                / neurons.loc[index, "post"]
                            )
                        )
                        for index in range(
                            len(neurons)
                        )
                    ]

                    new_tbars = [
                        neurons.roiInfo[index][target_roi]["pre"]
                        for index in range(
                            len(neurons)
                        )
                    ]

                    body_ids.append(
                        neurons.bodyId.values
                    )

                    if neurons.roiInfo.shape[0] > 0:
                        summed_connectivity += np.sum(
                            connection_strengths
                        )
                        weighted_synapses_total += np.sum(
                            weighted_synapses
                        )
                        weak_neurons += n_weak
                        medium_neurons += n_medium
                        strong_neurons += n_strong
                        tbar_total += np.sum(
                            new_tbars
                        )

            weak_connections.loc[
                [roi_source],
                [roi_target],
            ] = weak_neurons

            medium_connections.loc[
                [roi_source],
                [roi_target],
            ] = medium_neurons

            strong_connections.loc[
                [roi_source],
                [roi_target],
            ] = strong_neurons

            connectivity.loc[
                [roi_source],
                [roi_target],
            ] = summed_connectivity

            weighted_synapse_number.loc[
                [roi_source],
                [roi_target],
            ] = weighted_synapses_total

            tbars.loc[
                [roi_source],
                [roi_target],
            ] = tbar_total

    if body_ids:
        body_ids = np.unique(
            np.hstack(
                body_ids
            )
        )
    else:
        body_ids = np.array(
            [],
            dtype=int,
        )

    return (
        weak_connections,
        medium_connections,
        strong_connections,
        connectivity,
        weighted_synapse_number,
        tbars,
        body_ids,
    )
