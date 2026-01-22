import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphEncoder(nn.Module):
    """Graph encoder using GCN layers"""
    def __init__(self, node_in_dim, hidden_dim=64, out_dim=128,):
        super().__init__()
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        g = global_mean_pool(x, batch)
        z = self.lin(g)
        return z

class WaveEncoder(nn.Module):
    """Waveform encoder using 1D CNN for 4 PMTs with waveforms of length 1024"""
    def __init__(self, wave_in_dim=4, hidden_dim=128, out_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=wave_in_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, padding=1),
        )
        
    def forward(self, w):
        x = self.conv_layers(w)
        x = x.mean(dim=-1)
        return x

class GraphWaveModel(nn.Module):
    """
    Contrastive learning model for graph-waveform pairs.
    Uses symmetric loss to align graph and waveform embeddings.
    """
    def __init__(self, node_in_dim, wave_in_dim, emb_dim=128, temperature=0.1):
        super().__init__()
        self.graph_encoder = GraphEncoder(node_in_dim, hidden_dim=64, out_dim=emb_dim)
        self.wave_encoder = WaveEncoder(wave_in_dim, hidden_dim=128, out_dim=emb_dim)
        self.tau = temperature
    
    def encode_graphs(self, data):
        """Encode graphs to normalized embeddings"""
        zG = self.graph_encoder(data.x, data.edge_index, data.batch)
        return F.normalize(zG, dim=-1)
    
    def encode_waves(self, w):
        """Encode waveforms to normalized embeddings"""
        zW = self.wave_encoder(w)
        return F.normalize(zW, dim=-1)
    
    def forward(self, data, waves):
        """
        Forward pass computing similarity logits.
        
        Args:
            data: PyG Batch of graphs
            waves: Tensor of waveforms [B, wave_dim]
        
        Returns:
            logits: Similarity matrix [B, B]
        """
        zG = self.encode_graphs(data)
        zW = self.encode_waves(waves)
        logits = (zG @ zW.T) / self.tau
        return logits
    
    def compute_loss(self, logits):
        """
        Compute symmetric contrastive loss.
        
        Args:
            logits: Similarity matrix [B, B]
        
        Returns:
            loss: Scalar loss value
        """
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_g2w = F.cross_entropy(logits, labels)
        loss_w2g = F.cross_entropy(logits.T, labels)
        loss = (loss_g2w + loss_w2g) / 2
        return loss

    @torch.no_grad()
    def compute_metrics(self, logits):
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)

        pred_g2w = logits.argmax(dim=1)
        pred_w2g = logits.argmax(dim=0)

        acc_g2w = (pred_g2w == labels).float().mean()
        acc_w2g = (pred_w2g == labels).float().mean()
        acc = 0.5 * (acc_g2w + acc_w2g)

        return {
            "acc": acc,
            "acc_g2w": acc_g2w,
            "acc_w2g": acc_w2g,
        }
