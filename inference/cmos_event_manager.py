import uproot
import constants
import numpy as np
import pandas as pd
import awkward as ak
from dataclasses import dataclass
from fsspec.exceptions import FSTimeoutError
from aiohttp.client_exceptions import ServerDisconnectedError

@dataclass
class CMOSEvent:
    run_number: int
    event_number: int

class CMOSEventManager:
    INVALID_INDEX = -1
    def __init__(self, cmos_event: CMOSEvent):
        self.cmos_event = cmos_event
        self.file_path = f"{constants.RECO_URL}reco_run{self.cmos_event.run_number}_3D.root"
        self._events_tree = None

    def _get_events_tree(self):
        if self._events_tree is None:
            try:
                self._events_tree = uproot.open(f"{self.file_path}:Events")
            except FileNotFoundError:
                print(f"File not found: {self.file_path}. Skipping event.")
                return None
            except KeyError:
                print(f"Tree 'PMT_Events' not found in: {self.file_path}. Skipping event.")
                return None
            except (ServerDisconnectedError, FSTimeoutError) as e:
                print(f"Network error while accessing {self.file_path}: {e}. Skipping.")
                return None
        return self._events_tree

    def _get_sc_redpixIdx(self) -> np.ndarray:
        """Get supercluster redpix indices for this event."""
        tree = self._get_events_tree()
        
        sc_redpixIdx = tree["sc_redpixIdx"].array(
            entry_start=self.cmos_event.event_number,
            entry_stop=self.cmos_event.event_number + 1,
            library="np",
        )[0]

        sc_redpixIdx_event = sc_redpixIdx[sc_redpixIdx != self.INVALID_INDEX]

        return sc_redpixIdx_event
            
    def _get_redpix_coordinates(self) -> pd.DataFrame:
        """Get all redpix x,y,z coordinates for an event"""
        tree = self._get_events_tree()

        redpix_root_file = tree.arrays(
            filter_name="redpix_i*",
            entry_start=self.cmos_event.event_number,
            entry_stop=self.cmos_event.event_number + 1,
            library="ak",
        )
        redpix_df = ak.to_dataframe(redpix_root_file)

        return redpix_df
    
    def get_num_tracks(self) -> int:
        """Get number of tracks in this event."""
        return len(self._get_sc_redpixIdx())

    def get_track_pixels(self, track_number: int) -> pd.DataFrame:

        sc_redpixIdx_event = self._get_sc_redpixIdx()
        redpix_event_df = self._get_redpix_coordinates()        
        
        num_tracks = len(sc_redpixIdx_event) - 1
        if track_number < 0 or track_number > num_tracks:
            raise ValueError(
                f"Track {track_number} out of range. Available: 0-{num_tracks}"
            )
        
        start_idx = int(sc_redpixIdx_event[track_number])

        if track_number == num_tracks:
            track_df = redpix_event_df.iloc[start_idx:]
        else:
            end_idx = int(sc_redpixIdx_event[track_number + 1])
            track_df = redpix_event_df.iloc[start_idx:end_idx]
            
        return track_df
