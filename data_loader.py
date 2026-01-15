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
    def __init__(self, hdf5_path: str):
        """
        Args:
            hdf5_path: Path to HDF5 file created by build_dataset
        """
        self.hdf5_path = hdf5_path
        
        with h5py.File(hdf5_path, 'r') as f:
            self.keys = list(f.keys())

        with h5py.File(hdf5_path, 'r') as f:
            first_group = f[self.keys[0]]

            if 'graph_x' not in first_group:
                raise ValueError("graph_x not found in HDF5")

            self.node_dim = first_group['graph_x'].shape[1]
            
            if 'waveforms' not in first_group:
                raise ValueError("waveforms not found in HDF5")

            wf_shape = first_group['waveforms'].shape
            self.wave_dim = np.prod(wf_shape)
        
        print(f"Loaded {len(self.keys)} samples")
        print(f"Node feature dimension: {self.node_dim}")
        print(f"Waveform dimension: {self.wave_dim}")
    
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
            ).flatten()
        
        # Create PyG Data object
        graph = Data(x=node_features, edge_index=edge_index)
        
        return graph, waveforms

def collate_fn(batch: List[Tuple[Data, torch.Tensor]]):
    """Custom collate function for DataLoader"""
    graphs, waves = zip(*batch)
    
    batched_graphs = Batch.from_data_list(graphs)
    
    batched_waves = torch.stack(waves)
    
    return batched_graphs, batched_waves

def create_dataloaders(hdf5_path: str, 
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
    dataset = HDF5GraphWaveDataset(hdf5_path)
    
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