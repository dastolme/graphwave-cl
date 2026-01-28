import uproot
from typing import Dict
from dataclasses import dataclass

@dataclass
class RecoFile:
    number: int,
    cmos_tree: str,
    pmt_tree: str

class RecoFileMan:
    def __init__(reco_file: RecoFile, scaling_params: Dict):
        self.reco_file = reco_file
        self.scaling_params = scaling_params