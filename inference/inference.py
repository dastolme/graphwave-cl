import os
import sys
import torch
import uproot
import constants
import pandas as pd
import awkward as ak
import data_transforms
import torch.nn.functional as F
import pmt_event_manager as pem
import cmos_event_manager as cem
from dataclasses import dataclass
from data_transforms import Graph
from typing import List, Optional, Tuple
from torch_geometric.data import Data, Batch
from scipy.optimize import linear_sum_assignment

class RecoError(Exception):
    """Base class for all reco pipeline errors."""
    pass

class EmptyEventError(RecoError):
    """Event exists but produced no clusters or waveforms after cuts."""
    pass

@dataclass
class RecoFile:
    run_number: int
    path_to_file: str

class RecoFileReader:
    def __init__(self, reco_file: RecoFile):
        self.reco_file = reco_file
        self._uproot_file = None
        self._cmos_metadata: Optional[pd.DataFrame] = None
        self._pmt_metadata: Optional[pd.DataFrame] = None

    @property
    def uproot_file(self):
        """Lazy-load the ROOT file only when first accessed."""
        if self._uproot_file is None:
            file_path = os.path.join(
                self.reco_file.path_to_file,
                f"reco_run{self.reco_file.run_number}_3D.root"
            )
            self._uproot_file = uproot.open(file_path)
        return self._uproot_file

    @property
    def cmos_metadata(self) -> pd.DataFrame:
        """Lazy-load and cache CMOS metadata."""
        if self._cmos_metadata is None:
            cmos_tree = self.uproot_file["Events"].arrays(constants.CMOS_REDUCED_RECO_VARIABLES)
            self._cmos_metadata = ak.to_dataframe(cmos_tree)
        return self._cmos_metadata

    @property
    def pmt_metadata(self) -> pd.DataFrame:
        """Lazy-load and cache PMT metadata."""
        if self._pmt_metadata is None:
            pmt_tree = self.uproot_file["PMT_Events"].arrays(constants.PMT_RECO_VARIABLES)
            self._pmt_metadata = ak.to_dataframe(pmt_tree)
        return self._pmt_metadata

    @property
    def event_numbers(self) -> List[int]:
        """Return all available event numbers from the CMOS tree."""
        return self.cmos_metadata['event'].unique().tolist()

    def get_cmos_event(self, event_number: int) -> pd.DataFrame:
        df = self.cmos_metadata
        result = df[df['event'] == event_number]
        if result.empty:
            raise ValueError(f"Event {event_number} not found in CMOS tree.")
        return result.iloc[0]

    def get_pmt_event(self, event_number: int) -> pd.DataFrame:
        df = self.pmt_metadata
        result = df[df['pmt_wf_event'] == event_number]
        if result.empty:
            raise ValueError(f"Event {event_number} not found in PMT tree.")
        return result
    
    def get_cmos_manager(self, event_number: int) -> cem.CMOSEventManager:
        event = cem.CMOSEvent(
            run_number=self.reco_file.run_number,
            event_number=event_number
        )
        return cem.CMOSEventManager(event)

    def get_pmt_manager(self, event_number: int) -> pem.PMTEventManager:
        event = pem.PMTEvent(
            run_number=self.reco_file.run_number,
            event_number=event_number
        )
        return pem.PMTEventManager(event)
    
@dataclass
class CutConfig:
    pmt_query: Optional[str] = None

class RecoFileProcessor:
    def __init__(self, reader: RecoFileReader, cuts: Optional[CutConfig] = None):
        self.reader = reader
        self.cuts = cuts or CutConfig()
        self._norm_stats_loaded = False
        self.int_min = None
        self.int_max = None
        self.wave_min = None
        self.wave_max = None
        self.TOTAL_PIXEL_SIDE = None
        self.rebin_factor = None
    
    def load_normalization_stats(self, load_path: str):
        """Load and apply pre-computed normalization stats from disk."""
        try:
            stats = torch.load(load_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Normalization stats file not found: {load_path}")
        self._apply_normalization_stats(stats)
        print(f"✓ Loaded normalization stats from: {load_path}")
        return stats

    def load_normalization_stats_from_dict(self, stats: dict):
        """Apply pre-loaded normalization stats (no disk I/O)."""
        self._apply_normalization_stats(stats)
    
    def _check_norm_stats(self):
        if not self._norm_stats_loaded:
            raise RuntimeError("Normalization stats not loaded. Call load_normalization_stats() first.")

    def _apply_normalization_stats(self, stats: dict):
        """Shared logic for applying a stats dict to this processor."""
        self.int_min          = torch.tensor(stats['int_min'])  if stats['int_min']  is not None else None
        self.int_max          = torch.tensor(stats['int_max'])  if stats['int_max']  is not None else None
        self.wave_min         = torch.tensor(stats['wave_min']) if stats['wave_min'] is not None else None
        self.wave_max         = torch.tensor(stats['wave_max']) if stats['wave_max'] is not None else None
        self.TOTAL_PIXEL_SIDE = stats['TOTAL_PIXEL_SIDE']
        self.rebin_factor     = stats['rebin_factor']
        self._norm_stats_loaded = True
    
    def _apply_node_scaling(self, node_features: torch.Tensor) -> torch.Tensor:
        """Apply scaling to node features."""
        node_features[:, 0] = 2 * (node_features[:, 0] / self.TOTAL_PIXEL_SIDE) - 1
        node_features[:, 1] = 2 * (node_features[:, 1] / self.TOTAL_PIXEL_SIDE) - 1
        
        intensity_range = torch.clamp(self.int_max - self.int_min, min=1e-6)
        node_features[:, 2] = (node_features[:, 2] - self.int_min) / intensity_range
        
        return node_features
    
    def _apply_wave_scaling(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Apply scaling to waveforms."""
        wave_range = torch.clamp(self.wave_max - self.wave_min, min=1e-6)
        return (waveforms - self.wave_min) / wave_range
    
    def process_clusters(self, event_number: int) -> List[Graph]:
        """Process CMOS clusters for a given event number."""
        self._check_norm_stats()

        cmos_manager = self.reader.get_cmos_manager(event_number)
        nSc = cmos_manager.get_num_tracks()

        graphs = []
        matched_cluster_ids = []
        for cluster_id in range(nSc):
            track_df = cmos_manager.get_track_pixels(cluster_id)
            track_df_rebin = data_transforms._rebin_track(track_df, self.rebin_factor)
            graph = data_transforms._create_track_graph(track_df_rebin)
            graph.x = self._apply_node_scaling(graph.x)
            graphs.append(graph)
            matched_cluster_ids.append(int(cluster_id))

        return graphs, matched_cluster_ids

    def process_waveforms(self, event_number: int) -> List[torch.Tensor]:
        """Process PMT waveforms for a given event number."""
        self._check_norm_stats()
        pmt_event_df = self.reader.get_pmt_event(event_number)
        
        if self.cuts.pmt_query:
            pmt_event_df = pmt_event_df.query(self.cuts.pmt_query)
        
        trigger_ids = pmt_event_df['pmt_wf_trigger'].unique()
        pmt_manager  = self.reader.get_pmt_manager(event_number)

        waveforms = []
        matched_trigger_ids = []
        for trigger_id in trigger_ids:
            fast_raw = pmt_manager.get_waveforms(trigger_id, constants.FAST_DAQ)
            slow_raw = pmt_manager.get_waveforms(trigger_id, constants.SLOW_DAQ)
            raw = ak.concatenate([fast_raw,slow_raw])
            reshaped = data_transforms._reshape_waveforms_ak(raw)
            tensor = torch.tensor(reshaped.to_numpy(), dtype=torch.float32)
            tensor = tensor.transpose(0, 1)
            waveforms.append(self._apply_wave_scaling(tensor))
            matched_trigger_ids.append(int(trigger_id))

        return waveforms, matched_trigger_ids
    
    def process_event(self, event_number: int) -> Tuple[List[Graph], List[torch.Tensor]]:
        """Process both CMOS clusters and PMT waveforms for a single event."""
        graphs,cluster_ids = self.process_clusters(event_number)
        waveforms,trigger_ids = self.process_waveforms(event_number)

        if not graphs:
            raise EmptyEventError(f"Event {event_number}: no CMOS clusters found.")
        if not waveforms:
            raise EmptyEventError(f"Event {event_number}: no PMT waveforms found after cuts.")
            
        return graphs, cluster_ids, waveforms, trigger_ids
    
@dataclass
class MatchResult:
    cluster_id: int
    trigger_id: str
    similarity_score: float

class GraphWaveformMatcher:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        
    
    def load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        node_in_dim = checkpoint['node_in_dim']
        wave_in_dim = checkpoint['wave_in_dim']
        emb_dim = checkpoint['args']['emb_dim']
        temperature = checkpoint['args']['temperature']
        
        sys.path.insert(1, '../')
        from model import GraphWaveModel
        self.model = GraphWaveModel(
            node_in_dim=node_in_dim,
            wave_in_dim=wave_in_dim,
            emb_dim=emb_dim,
            temperature=temperature
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()

        self.tau = temperature
        
        print(f"✓ Loaded model from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
        
        return self.model

    def score(
        self,
        graphs: List[Graph],
        cluster_ids: List[int],
        waveforms: List[torch.Tensor],
        trigger_ids: List[int]
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        Compute the raw similarity matrix without matching.
        Returns (similarity_matrix, cluster_ids, trigger_ids).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not graphs:
            raise ValueError("Cannot score: graphs list is empty.")
        if not waveforms:
            raise ValueError("Cannot score: waveforms list is empty.")

        with torch.no_grad():
            pyg_graphs = [Data(x=g.x, edge_index=g.edge_index).to(self.device) for g in graphs]
            zG = self.model.encode_graphs(Batch.from_data_list(pyg_graphs))
            zW = self.model.encode_waves(torch.stack([w.to(self.device) for w in waveforms]))
            similarity_matrix = (zG @ zW.T) / self.tau

        return similarity_matrix.cpu(), cluster_ids, trigger_ids

    def score_batch(
        self,
        batch: List[Tuple[List[Graph], List[int], List[torch.Tensor], List[int]]]
    ) -> List[Tuple[torch.Tensor, List[int], List[int]]]:
        """
        Run one GPU forward pass over all events in a run.
        Each entry in batch is (graphs, cluster_ids, waveforms, trigger_ids).
        Returns a list of (similarity_matrix, cluster_ids, trigger_ids) — one per event,
        where similarity is only computed between clusters and waveforms of the same event.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not batch:
            return []

        # Flatten all graphs/waveforms and record per-event slice boundaries
        all_graphs, all_waveforms = [], []
        graph_slices, wave_slices = [], []
        g_cursor, w_cursor = 0, 0

        for graphs, cluster_ids, waveforms, trigger_ids in batch:
            all_graphs.extend(
                Data(x=g.x, edge_index=g.edge_index).to(self.device) for g in graphs
            )
            all_waveforms.extend(w.to(self.device) for w in waveforms)
            graph_slices.append((g_cursor, g_cursor + len(graphs)))
            wave_slices.append((w_cursor,  w_cursor  + len(waveforms)))
            g_cursor += len(graphs)
            w_cursor += len(waveforms)

        # Single forward pass for the entire run
        with torch.no_grad():
            zG = self.model.encode_graphs(Batch.from_data_list(all_graphs))  # (total_clusters, emb)
            zW = self.model.encode_waves(torch.stack(all_waveforms))          # (total_triggers, emb)

        # Slice embeddings back per event — cross-event similarity is never computed
        results = []
        for (gs, ge), (ws, we), (_, cluster_ids, _, trigger_ids) in zip(
            graph_slices, wave_slices, batch
        ):
            sim = (zG[gs:ge] @ zW[ws:we].T) / self.tau
            results.append((sim.cpu(), cluster_ids, trigger_ids))

        return results

    def match(
        self,
        graphs: List[Graph],
        cluster_ids: List[int],
        waveforms: List[torch.Tensor],
        trigger_ids: List[int]
    ) -> Tuple[List[MatchResult], List[int], torch.Tensor]:
        similarity_matrix, cluster_ids, trigger_ids = self.score(
            graphs, cluster_ids, waveforms, trigger_ids
        )

        row_ids, col_ids = linear_sum_assignment(-similarity_matrix.numpy())

        matches = [
            MatchResult(
                cluster_id=cluster_ids[r],
                trigger_id=trigger_ids[c],
                similarity_score=float(similarity_matrix[r, c])
            )
            for r, c in zip(row_ids, col_ids)
        ]

        matched_cluster_ids = set(row_ids)
        unmatched_cluster_ids = [
            cluster_ids[i] for i in range(len(graphs)) if i not in matched_cluster_ids
        ]

        return matches, unmatched_cluster_ids, similarity_matrix
                

