import os
import torch
import uproot
import pandas as pd
import awkward as ak
from typing import Dict, Optional
from dataclasses import dataclass

RECO_PATH = "https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/RECO/Run5_Saladin/"
RECO_CMOS_VARIABLES = ['event', 'nSc', 'sc_redpixIdx', 
                       'redpix_ix', 'redpix_iy', 'redpix_iz']
RECO_PMT_VARIABLES = ['pmt_wf_event', 'pmt_wf_trigger', 'pmt_wf_sampling', 
                      'pmt_wf_channel', 'pmt_fullWaveform_Y']

@dataclass
class RecoFile:
    number: int
    cmos_tree: str
    pmts_tree: str

@dataclass
class Graph:
    event: int
    cluster_id: str
    x: torch.Tensor
    edge_index: torch.Tensor

@dataclass
class WaveformSet:
    event: int
    trigger_id: str
    waveforms: pd.DataFrame

class RecoFileReader:
    def __init__(self, reco_file: RecoFile):
        self.reco_file = reco_file
        self.uproot_file = None
        self._check_root_file_existence()

    def _check_root_file_existence(self):
        file_name = os.pathlib.join(RECO_PATH, "reco_run", self.reco_file.number, "_3D.root")
        try:
            self.uproot_file = uproot.open(file_name)
        except Exception as e:
            raise RuntimeError(f"Error opening ROOT file {file_name}: {e}")

    def get_cmos_tree(self):
        if self.uproot_file is None:
            raise RuntimeError("ROOT file not opened")
        
        cmos_tree = self.uproot_file[self.reco_file.cmos_tree].arrays(RECO_CMOS_VARIABLES)
        cmos_df = ak.to_dataframe(cmos_tree)

        return cmos_df

    def get_pmts_tree(self):
        if self.uproot_file is None:
            raise RuntimeError("ROOT file not opened")
        
        pmts_tree = self.uproot_file[self.reco_file.pmts_tree].arrays(RECO_PMT_VARIABLES)
        pmts_df = ak.to_dataframe(pmts_tree)

        return pmts_df

class RecoFilePreprocessor:
    def __init__(self, cmos_df: pd.DataFrame, pmt_df: pd.DataFrame, scaling_params: Optional[Dict]):
        self.cmos_df = cmos_df
        self.pmt_df = pmt_df
        self.scaling_params = scaling_params



