import uproot
import constants
import numpy as np
import pandas as pd
import awkward as ak
from dataclasses import dataclass
from typing import Tuple
from fsspec.exceptions import FSTimeoutError
from aiohttp.client_exceptions import ServerDisconnectedError

@dataclass
class PMTEvent:
    run_number: int
    event_number: int

class PMTEventManager:
    RECO_WFS_COLUMN  = "pmt_fullWaveform_Y"
    METADATA_COLUMNS = ['pmt_wf_event', 'pmt_wf_trigger', 
                        'pmt_wf_sampling', 'pmt_wf_channel']
    def __init__(self, pmt_event: PMTEvent):
        self.pmt_event = pmt_event
        self.file_path = f"{constants.RECO_URL}reco_run{self.pmt_event.run_number}_3D.root"
        self._metadata = None
        self._pmt_events_tree = None

    def _get_pmt_events_tree(self):
        if self._pmt_events_tree is None:
            try:
                self._pmt_events_tree = uproot.open(f"{self.file_path}:PMT_Events")
            except FileNotFoundError:
                print(f"File not found: {self.file_path}. Skipping event.")
                return None
            except KeyError:
                print(f"Tree 'PMT_Events' not found in: {self.file_path}. Skipping event.")
                return None
            except (ServerDisconnectedError, FSTimeoutError) as e:
                print(f"Network error while accessing {self.file_path}: {e}. Skipping.")
                return None
        return self._pmt_events_tree
        
    def _load_metadata(self):
        """Load metadata once for the entire run."""
        if self._metadata is not None:
            return self._metadata
        
        tree = self._get_pmt_events_tree()
        
        self._metadata = tree.arrays(
            self.METADATA_COLUMNS,
            library='ak'
        )
        return self._metadata
    
    def _find_entry_range(self, trigger_id: int, sampling: int) -> Tuple[int, int]:
        """Find entry_start and entry_stop for given trigger."""
        metadata = self._load_metadata()
        
        single_waveform_mask = (
            (metadata['pmt_wf_event'] == self.pmt_event.event_number) &
            (metadata['pmt_wf_trigger'] == trigger_id) &
            (metadata['pmt_wf_sampling'] == sampling)
        )
        
        indices = ak.where(single_waveform_mask)[0]
        
        if len(indices) == 0:
            raise ValueError(f"No waveforms found for trigger {trigger_id} and sampling {sampling}")
        
        entry_start = int(ak.min(indices))
        entry_stop = int(ak.max(indices)) + 1
        
        return entry_start, entry_stop
    
    def get_waveforms(self, trigger_id: int, sampling: int) -> ak.Array:
        """Get PMT waveforms for specific trigger.
        
        Returns:
            Awkward array with waveform data
        """
        entry_start, entry_stop = self._find_entry_range(trigger_id, sampling)

        tree = self._get_pmt_events_tree()

        wfs = tree[self.RECO_WFS_COLUMN].array(
            entry_start=entry_start,
            entry_stop=entry_stop,
            library='ak'
            )

        if len(wfs) == 0:
            raise ValueError(f"No waveforms found for trigger {trigger_id}")

        return wfs