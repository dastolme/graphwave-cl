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
    def __init__(self, hdf5_path: str, apply_scaling: False):
        """
        Args:
            hdf5_path: Path to HDF5 file created by build_dataset
            apply_scaling: Whether to apply normalization to features
        """
        self.hdf5_path = hdf5_path
        self.apply_scaling = apply_scaling
        self.TOTAL_PIXEL_SIDE = 2304
        
        self.keys = self._get_hdf5_keys(hdf5_path)
        self.node_dim, self.wave_dim = self._get_input_dim(hdf5_path, self.keys)

        if self.apply_scaling:
            self.int_min, self.int_max = self._compute_node_stats()
            self.wave_min, self.wave_max = self._compute_wave_stats()
            print(f"Intensity will be scaled to [-1, 1]")
            print(f"Waveforms will be scaled to [0, 1]")
        else:
            self.int_min = self.int_max = None
            self.wave_min = self.wave_max = None
            
        print(f"Loaded {len(self.keys)} samples")
        print(f"Node feature dimension: {self.node_dim}")
        print(f"Waveform dimension: {self.wave_dim}")

    @staticmethod
    def _get_hdf5_keys(hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            keys = list(f.keys())

        return keys

    @staticmethod
    def _get_input_dim(hdf5_path, keys):
        with h5py.File(hdf5_path, 'r') as f:
            first_group = f[keys[0]]

            if 'graph_x' not in first_group:
                raise ValueError("graph_x not found in HDF5")

            node_dim = first_group['graph_x'].shape[1]
            
            if 'waveforms' not in first_group:
                raise ValueError("waveforms not found in HDF5")

            wf_shape = first_group['waveforms'].shape
            wave_dim = np.prod(wf_shape)

        return node_dim, wave_dim

    def _compute_node_stats(self):
        """Compute min/max for intensity column (column 2) of node features"""
        all_intensities = []

        with h5py.File(self.hdf5_path, 'r') as f:
            for key in list(f.keys()):
                node = f[key]['graph_x'][:]
                all_intensities.append(node[:, 2])

        all_intensities = np.concatenate(all_intensities)
        min_val = torch.tensor(all_intensities.min(), dtype=torch.float32)
        max_val = torch.tensor(all_intensities.max(), dtype=torch.float32)

        print(f"Node intensity range: [{min_val:.4f}, {max_val:.4f}]")
        return min_val, max_val

    def _compute_wave_stats(self):
        """Compute global min/max across all waveforms and PMTs"""
        all_waves = []

        with h5py.File(self.hdf5_path, 'r') as f:
            for key in list(f.keys()):
                wf = f[key]['waveforms'][:]
                all_waves.append(wf.flatten())

        all_waves = np.concatenate(all_waves)
        min_val = torch.tensor(all_waves.min(), dtype=torch.float32)
        max_val = torch.tensor(all_waves.max(), dtype=torch.float32)

        print(f"Waveform range: [{min_val:.4f}, {max_val:.4f}]")
        return min_val, max_val
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[key]
            
            node_features = torch.tensor(
                group['graph_x'][:], dtype=torch.float32
            )

            edge_index = torch.tensor(
                group['graph_edge_index'][:], dtype=torch.long
            )

            waveforms = torch.tensor(
                group['waveforms'][:], dtype=torch.float32
            )

            if self.apply_scaling:
                node_features[:, 0] = 2 * (node_features[:, 0] / self.TOTAL_PIXEL_SIDE) - 1
                node_features[:, 1] = 2 * (node_features[:, 1] / self.TOTAL_PIXEL_SIDE) - 1

                intensity_range = self.int_max - self.int_min
                intensity_range = torch.clamp(intensity_range, min=1e-6)
                node_features[:, 2] = 2 * ((node_features[:, 2] - self.int_min) / intensity_range) - 1

                wave_range = self.wave_max - self.wave_min
                wave_range = torch.clamp(wave_range, min=1e-6)
                waveforms = (waveforms - self.wave_min) / wave_range
        
        
        waveforms = waveforms.flatten()
        graph = Data(x=node_features, edge_index=edge_index)
        
        return graph, waveforms

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