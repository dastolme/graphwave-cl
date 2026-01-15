import torch
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from data_loader import create_dataloaders
from model import GraphWaveModel

def train_epoch(model, train_loader, optimizer, device, writer=None, epoch=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (batch_graphs, batch_waves) in enumerate(train_loader):
        batch_graphs = batch_graphs.to(device)
        batch_waves = batch_waves.to(device)
        
        logits = model(batch_graphs, batch_waves)
        loss = model.compute_loss(logits)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
    
    return total_loss / num_batches

def validate(model, val_loader, device, writer=None, epoch=0):
    """Validation loop"""
    model.eval()
    total_loss = 0
    acc_sum = 0
    acc_g2w_sum = 0
    acc_w2g_sum = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_graphs, batch_waves in val_loader:
            batch_graphs = batch_graphs.to(device)
            batch_waves = batch_waves.to(device)
            
            logits = model(batch_graphs, batch_waves)
            loss = model.compute_loss(logits)

            metrics = model.compute_metrics(logits)
            
            total_loss += loss.item()
            acc_sum += metrics["acc"].item()
            acc_g2w_sum += metrics["acc_g2w"].item()
            acc_w2g_sum += metrics["acc_w2g"].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = acc_sum / num_batches
    avg_acc_g2w = acc_g2w_sum / num_batches
    avg_acc_w2g = acc_w2g_sum / num_batches
    
    if writer is not None:
        writer.add_scalar('Loss/validation', avg_loss, epoch)
        writer.add_scalar('Accuracy/validation', avg_acc, epoch)
        writer.add_scalar('Accuracy/validation_g2w', avg_acc_g2w, epoch)
        writer.add_scalar('Accuracy/validation_w2g', avg_acc_w2g, epoch)
    
    return avg_loss, avg_acc

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = output_dir / 'runs' / f'experiment_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\nTensorBoard log directory: {log_dir}")
    print(f"Run: tensorboard --logdir={output_dir / 'runs'}")
    
    print("\nLoading dataset...")
    train_loader, val_loader, dataset = create_dataloaders(
        hdf5_path=args.data_path,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_val,
        train_split=args.train_split,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print("\nInitializing model...")
    model = GraphWaveModel(
        node_in_dim=dataset.node_dim,
        wave_in_dim=dataset.wave_dim,
        emb_dim=args.emb_dim,
        temperature=args.temperature
    ).to(device)
    
    print(f"Node input dim: {dataset.node_dim}")
    print(f"Wave input dim: {dataset.wave_dim}")
    print(f"Embedding dim: {args.emb_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    writer.add_text('Model/Architecture', str(model), 0)
    writer.add_text('Model/Parameters', f"Total: {sum(p.numel() for p in model.parameters()):,}", 0)
    
    writer.add_text('Hyperparameters/batch_size_train', str(args.batch_size_train), 0)
    writer.add_text('Hyperparameters/batch_size_val', str(args.batch_size_val), 0)
    writer.add_text('Hyperparameters/learning_rate', str(args.lr), 0)
    writer.add_text('Hyperparameters/embedding_dim', str(args.emb_dim), 0)
    writer.add_text('Hyperparameters/temperature', str(args.temperature), 0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, writer, epoch)
        val_loss, val_acc = validate(model, val_loader, device, writer, epoch)
        
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalars('Loss/train_vs_val', {
            'train': train_loss,
            'validation': val_loss
        }, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f"Epoch {epoch+1:3d}/{args.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            
            writer.add_scalar('Best/validation_loss', best_val_loss, epoch)
        
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    writer.add_hparams(
        {
            'lr': args.lr,
            'batch_size_train': args.batch_size_train,
            'batch_size_val': args.batch_size_val,
            'emb_dim': args.emb_dim,
            'temperature': args.temperature,
        },
        {
            'best_val_loss': best_val_loss,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
        }
    )
    
    writer.close()
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train contrastive graph-wave model')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--batch_size_train', type=int, default=32,
                        help='Batch size training')
    parser.add_argument('--batch_size_val', type=int, default=8,
                        help='Batch size validation')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/val split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive loss')
    
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save models')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)