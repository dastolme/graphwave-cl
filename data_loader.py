import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import h5py
import numpy as np
from typing import List, Tuple

class HDF5GraphWaveDataset(Dataset):
    """
    Dataset loader for HDF5 files containing graph-waveform pairs.
    """
    def __init__(self, hdf5_path: str, apply_scaling: bool = False):
        self.apply_scaling = apply_scaling
        self.TOTAL_PIXEL_SIDE = 2304
        
        print("Loading entire dataset into memory...")
        self.samples = self._load_all_samples(hdf5_path)
        
        print(f"Loaded {len(self.samples)} samples into memory")
        print(f"Node feature dimension: {self.node_dim}")
        print(f"Waveform channels: {self.wave_channels}")
        print(f"Waveform length: {self.wave_length}")
        
        if self.apply_scaling:
            self._initialize_scaling_parameters()
        else:
            self.int_min = self.int_max = None
            self.wave_min = self.wave_max = None

    def _load_all_samples(self, hdf5_path: str) -> list:
        """Load all samples from HDF5 file into memory."""
        samples = []
        
        with h5py.File(hdf5_path, 'r') as f:
            keys = list(f.keys())
            
            # Get dimensions from first sample
            self._extract_dimensions(f[keys[0]])
            
            # Load all samples
            for i, key in enumerate(keys):
                if i % 1000 == 0:
                    print(f"Loading sample {i}/{len(keys)}...")
                
                sample = self._load_single_sample(f[key])
                samples.append(sample)
        
        return samples

    def _extract_dimensions(self, group):
        """Extract data dimensions from a sample group."""
        self.node_dim = group['graph_x'].shape[1]
        wf_shape = group['waveforms'].shape
        self.wave_length = wf_shape[0]
        self.wave_channels = wf_shape[1]

    def _load_single_sample(self, group) -> dict:
        """Load a single sample from HDF5 group."""
        return {
            'node_features': torch.tensor(group['graph_x'][:], dtype=torch.float32),
            'edge_index': torch.tensor(group['graph_edge_index'][:], dtype=torch.long),
            'waveforms': torch.tensor(group['waveforms'][:], dtype=torch.float32)
        }

    def _initialize_scaling_parameters(self):
        """Compute and store scaling parameters."""
        self.int_min, self.int_max = self._compute_node_stats()
        self.wave_min, self.wave_max = self._compute_wave_stats()
        
        print(f"Intensity will be scaled to [0, 1]")
        print(f"Waveforms will be scaled to [0, 1]")

    def _compute_node_stats(self):
        """Compute min/max for intensity column from in-memory data."""
        all_intensities = torch.cat([
            sample['node_features'][:, 2] 
            for sample in self.samples
        ])
        
        min_val = all_intensities.min()
        max_val = all_intensities.max()
        
        print(f"Node intensity range: [{min_val:.4f}, {max_val:.4f}]")
        return min_val, max_val

    def _compute_wave_stats(self):
        """Compute global min/max from in-memory data."""
        all_waves = torch.cat([
            sample['waveforms'].flatten() 
            for sample in self.samples
        ])
        
        min_val = all_waves.min()
        max_val = all_waves.max()
        
        print(f"Waveform range: [{min_val:.4f}, {max_val:.4f}]")
        return min_val, max_val
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        node_features = sample['node_features'].clone()
        edge_index = sample['edge_index']
        waveforms = sample['waveforms'].clone()
        
        if self.apply_scaling:
            node_features = self._apply_node_scaling(node_features)
            waveforms = self._apply_wave_scaling(waveforms)
        
        waveforms = waveforms.permute(1, 0)
        graph = Data(x=node_features, edge_index=edge_index)
        
        return graph, waveforms
    
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

def collate_fn(batch: List[Tuple[Data, torch.Tensor]]):
    """Custom collate function for DataLoader"""
    graphs, waves = zip(*batch)
    
    batched_graphs = Batch.from_data_list(graphs)
    
    batched_waves = torch.stack(waves)
    
    return batched_graphs, batched_waves

def create_dataloaders(hdf5_path: str,
                       apply_scaling: str = False, 
                       batch_size_train: int = 32,
                       batch_size_val: int = 8,  
                       train_split: float = 0.8,
                       num_workers: int = 0,
                       pin_memory: bool = True):
    """
    Create train and validation dataloaders.
    
    Args:
        hdf5_path: Path to HDF5 dataset
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (use True for GPU)
    
    Returns:
        train_loader, val_loader, dataset
    """
    dataset = HDF5GraphWaveDataset(hdf5_path, apply_scaling)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, dataset