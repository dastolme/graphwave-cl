import torch
import constants
import numpy as np
import pandas as pd
import awkward as ak
from typing import List
from dataclasses import dataclass

@dataclass
class Graph:
    x: torch.Tensor
    edge_index: torch.Tensor

def _rescale_slow_waveforms(waveforms_ak: ak.Array) -> List[np.array]:
    from scipy import signal

    waveforms_list = []

    for idx, wf in enumerate(waveforms_ak):
        wf_np = ak.to_numpy(wf)

        if idx in constants.SLOW_DAQ_IDXS:
            wf_resampled = signal.resample(
                wf_np[constants.SLOW_DAQ_CUT:], 
                constants.FAST_DAQ
                )
        elif idx in constants.FAST_DAQ_IDXS:
            wf_resampled = wf_np

        waveforms_list.append(wf_resampled)

    return waveforms_list

def _reshape_waveforms_ak(waveforms_ak: ak.Array) -> pd.DataFrame:
    """Convert waveforms DataFrame to tensor format.
    
    Args:
        waveforms_df: DataFrame with 'subentry' index level, 
                    'pmt_wf_channel' column, and 'pmt_fullWaveform_Y' values
    
    Returns:
        Waveforms object containing the pivoted tensor data
    """
    waveforms_list = _rescale_slow_waveforms(waveforms_ak)
    assert all(len(wf) == constants.FAST_DAQ for wf in waveforms_list), \
    "Not all waveforms resampled to expected length"
    waveforms_df = pd.DataFrame(waveforms_list)

    return waveforms_df.transpose()

def _rebin_track(track_df: pd.DataFrame, rebin_factor: int) -> pd.DataFrame:
    """
    track_df: DataFrame with columns redpix_ix, redpix_iy, redpix_iz
    returns:  DataFrame with same columns but rebinned coordinates and summed intensity
    """
    scale = 1.0 / rebin_factor
    target_max = constants.TOTAL_NUM_PIXELS // rebin_factor - 1

    xy = track_df[['redpix_ix', 'redpix_iy']].values
    intensities = track_df['redpix_iz'].values

    scaled = np.floor(xy * scale).astype(int)
    scaled = np.clip(scaled, 0, target_max)

    coords, inverse = np.unique(scaled, axis=0, return_inverse=True)
    summed_intensities = np.bincount(inverse, weights=intensities)

    return pd.DataFrame({
        'redpix_ix': coords[:, 0],
        'redpix_iy': coords[:, 1],
        'redpix_iz': summed_intensities
    })

def _create_track_graph(track_df: pd.DataFrame) -> Graph:
    from torch_geometric.nn import knn_graph

    track_df = track_df[track_df['redpix_iz'] > 0]
    
    positions = track_df[['redpix_ix', 'redpix_iy']].values
    positions_tensor = torch.FloatTensor(positions)
    
    edge_index = knn_graph(positions_tensor, k=20, loop=False)
    
    features = track_df[['redpix_ix', 'redpix_iy', 'redpix_iz']].values
    x = torch.FloatTensor(features)
    
    return Graph(x=x, edge_index=edge_index)