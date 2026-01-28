import os
import uproot
import awkward as ak
from typing import Dict
from dataclasses import dataclass

RECO_PATH = ""
RECO_CMOS_VARIABLES = []
RECO_PMT_VARIABLES = []

@dataclass
class RecoFile:
    number: int
    cmos_tree: str
    pmts_tree: str

class RecoFileMan:
    def __init__(self, reco_file: RecoFile):
        self.reco_file = reco_file
        self.root_file = None
        self._check_root_file_existence()

    def _check_root_file_existence(self):
        file_name = os.pathlib.join(RECO_PATH, "reco_run", self.reco_file.number, "_3D.root")
        try:
            self.root_file = uproot.open(file_name)
        except Exception as e:
            raise RuntimeError(f"Error opening ROOT file {file_name}: {e}")

    def get_cmos_tree(self):
        if self.root_file is None:
            raise RuntimeError("ROOT file not opened")
        
        cmos_tree = self.root_file[self.reco_file.cmos_tree].arrays(RECO_CMOS_VARIABLES)
        cmos_df = ak.to_dataframe(cmos_tree)

        return cmos_df

    def get_pmts_tree(self):
        if self.root_file is None:
            raise RuntimeError("ROOT file not opened")
        
        pmts_tree = self.root_file[self.reco_file.pmts_tree].arrays(RECO_PMT_VARIABLES)
        pmts_df = ak.to_dataframe(pmts_tree)

        return pmts_df



