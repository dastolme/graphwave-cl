import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

from data_loader import create_dataloaders
from model import GraphWaveModel


# ── JSON serialization ────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Metadata ──────────────────────────────────────────────────────────────────

def load_event_metadata(hdf5_path, indices):
    metadata = []
    with h5py.File(hdf5_path, 'r') as f:
        keys = list(f.keys())
        for idx in indices:
            if idx < len(keys):
                key   = keys[idx]
                group = f[key]
                parts = key.split('_')
                metadata.append({
                    'hdf5_key':    key,
                    'dataset_idx': idx,
                    'run':         parts[0],
                    'event':       parts[1],
                    'num_nodes':   int(group['graph_x'].shape[0]),
                    'num_edges':   int(group['graph_edge_index'].shape[1]),
                })
            else:
                metadata.append({'dataset_idx': idx, 'error': 'index out of range'})
    return metadata


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_similarity_matrix(similarity_matrix, batch_metadata, save_path,
                           hungarian_pred=None, title_suffix=""):
    B   = similarity_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, B * 0.8), max(8, B * 0.8)))
    sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, square=True, ax=ax,
                cbar_kws={'label': 'Similarity'},
                linewidths=0.5, linecolor='gray')

    labels = []
    for i, meta in enumerate(batch_metadata):
        if 'run' in meta and 'event' in meta:
            label = f"{i}: R{meta['run']}_E{meta['event']}\n({meta['num_nodes']}n)"
        else:
            label = f"{i}: Idx{meta['dataset_idx']}"
        labels.append(label)

    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)

    for i in range(B):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='blue', lw=3, linestyle='--'))
    if hungarian_pred is not None:
        for i in range(B):
            pred = hungarian_pred[i]
            if pred != i:
                ax.add_patch(plt.Rectangle((pred, i), 1, 1, fill=False,
                                           edgecolor='red', lw=2))

    ax.set_xlabel('Waveform Index', fontsize=10, fontweight='bold')
    ax.set_ylabel('Graph Index',    fontsize=10, fontweight='bold')
    ax.set_title(f'Graph-Waveform Similarity Matrix{title_suffix}\n'
                 'Blue dashed = Correct matches, Red = Wrong Hungarian assignment',
                 fontsize=11, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_diagonal_histogram(split_data, output_dir):
    """
    One subplot per split sharing the x-axis.
    Correct/failed split is Hungarian-based.
    Vertical lines at p5, p10, p25 of each split.
    """
    split_names = list(split_data.keys())
    n_splits    = len(split_names)
    cut_pcts    = [5, 10, 25]
    cut_colors  = ['#e74c3c', '#e67e22', '#27ae60']

    all_scores   = np.concatenate([np.array(split_data[s]['scores']) for s in split_names])
    x_min, x_max = all_scores.min(), all_scores.max()
    bins         = np.linspace(x_min, x_max, 60)

    fig, axes = plt.subplots(n_splits, 1,
                             figsize=(10, 4 * n_splits),
                             sharex=True)
    if n_splits == 1:
        axes = [axes]

    for ax, split_name in zip(axes, split_names):
        scores  = np.array(split_data[split_name]['scores'])
        correct = np.array(split_data[split_name]['correct'])
        acc     = 100 * correct.mean() if len(correct) > 0 else 0.0

        scores_ok   = scores[correct]
        scores_fail = scores[~correct]

        ax.hist(scores_ok,   bins=bins, alpha=0.65, color='steelblue',
                label=f'Correct  (n={len(scores_ok)})',  density=True)
        ax.hist(scores_fail, bins=bins, alpha=0.65, color='salmon',
                label=f'Failed   (n={len(scores_fail)})', density=True)

        for pct, col in zip(cut_pcts, cut_colors):
            val      = np.percentile(scores, pct)
            pct_pass = 100 * np.mean(scores >= val)
            ax.axvline(val, color=col, lw=1.8, ls='--',
                       label=f'p{pct} = {val:.3f}  ({pct_pass:.1f}% pass)')

        stats_txt = (f'mean={scores.mean():.3f}  std={scores.std():.3f}  '
                     f'min={scores.min():.3f}  max={scores.max():.3f}')
        ax.text(0.02, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_title(f'{split_name.upper()} split  –  '
                     f'N={len(scores)},  Hungarian acc={acc:.1f}%',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 0.88))
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel('Diagonal similarity score  s(graph_i, wave_i)', fontsize=10)
    fig.suptitle('Diagonal Score Distribution  –  val & test\n'
                 'Correct/Failed split by Hungarian assignment  |  '
                 'vertical lines = candidate quality-cut thresholds',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = Path(output_dir) / 'diagonal_score_histogram.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Diagonal histogram   : {out_path}")


# ── Single-loader evaluation ──────────────────────────────────────────────────

def evaluate_loader(model, loader, device, hdf5_path, output_dir,
                    split_name, save_heatmaps=False):
    """
    Run inference on every batch of *loader* in a single pass.
    Diagonal scores and Hungarian correctness are collected here —
    no second pass needed.

    Returns
    -------
    all_batch_results : list of per-batch dicts
    all_failures      : list of per-event failure dicts
    diag              : dict {'scores': [...], 'correct': [...]}
    summary           : dict with aggregate metrics
    """
    model.eval()

    dataset = loader.dataset
    indices = (dataset.indices
               if hasattr(dataset, 'indices')
               else list(range(len(dataset))))

    all_batch_results  = []
    all_failures       = []
    diag_scores        = []
    diag_correct       = []
    total_events       = 0
    total_correct_hung = 0
    total_correct_g2w  = 0
    failed_batches     = []
    batch_start        = 0

    with torch.no_grad():
        for batch_num, (batch_graphs, batch_waves) in enumerate(
                tqdm(loader, desc=f"  {split_name}")):

            B             = batch_graphs.num_graphs
            batch_indices = indices[batch_start: batch_start + B]
            batch_start  += B

            batch_graphs = batch_graphs.to(device)
            batch_waves  = batch_waves.to(device)

            logits = model(batch_graphs, batch_waves)   # [B, B]
            sim    = logits.cpu().numpy()

            # Hungarian assignment
            _, col_ind        = linear_sum_assignment(-sim)
            hungarian_pred    = col_ind
            correct_hungarian = (hungarian_pred == np.arange(B))

            # Top-1 for reference
            pred_g2w    = logits.argmax(dim=1).cpu().numpy()
            correct_g2w = (pred_g2w == np.arange(B))

            # Collect diagonal scores in this same pass
            for i in range(B):
                diag_scores.append(float(sim[i, i]))
                diag_correct.append(bool(correct_hungarian[i]))

            batch_metadata = load_event_metadata(hdf5_path, batch_indices)

            batch_result = {
                'batch_num':             batch_num,
                'batch_size':            B,
                'similarity_matrix':     sim.tolist(),
                'hungarian_pred':        hungarian_pred.tolist(),
                'predictions_g2w':       pred_g2w.tolist(),
                'correct_hungarian':     correct_hungarian.tolist(),
                'correct_g2w':           correct_g2w.tolist(),
                'num_correct_hungarian': int(correct_hungarian.sum()),
                'num_correct_g2w':       int(correct_g2w.sum()),
                'events':                [],
            }

            for i in range(B):
                sorted_idx = np.argsort(sim[i])[::-1]
                rank       = int(np.where(sorted_idx == i)[0][0]) + 1

                ev = {
                    'batch_position':       i,
                    'dataset_idx':          int(batch_indices[i]),
                    'metadata':             batch_metadata[i],
                    'correct_hungarian':    bool(correct_hungarian[i]),
                    'correct_g2w':          bool(correct_g2w[i]),
                    'hungarian_pred_match': int(hungarian_pred[i]),
                    'predicted_match':      int(pred_g2w[i]),
                    'self_similarity':      float(sim[i, i]),
                    'hungarian_similarity': float(sim[i, hungarian_pred[i]]),
                    'rank_of_correct':      rank,
                    'top3':                 rank <= 3,
                    'top5':                 rank <= 5,
                }
                if not correct_hungarian[i]:
                    wrong = int(hungarian_pred[i])
                    ev['wrong_match_metadata'] = batch_metadata[wrong]
                    ev['similarity_to_wrong']  = float(sim[i, wrong])
                    all_failures.append({**ev, 'batch_num': batch_num})

                batch_result['events'].append(ev)

            all_batch_results.append(batch_result)
            total_events       += B
            total_correct_hung += int(correct_hungarian.sum())
            total_correct_g2w  += int(correct_g2w.sum())

            has_failures = not all(correct_hungarian)
            if has_failures:
                failed_batches.append(batch_num)

            if save_heatmaps and (has_failures or batch_num < 5):
                plot_path = (Path(output_dir) /
                             f'{split_name}_batch_{batch_num:04d}_similarity_matrix.png')
                suffix = " [HAS FAILURES]" if has_failures else ""
                plot_similarity_matrix(sim, batch_metadata, plot_path,
                                       hungarian_pred, suffix)
                batch_result['plot_path'] = str(plot_path)

    summary = {
        'split':             split_name,
        'total_events':      total_events,
        'correct_hungarian': total_correct_hung,
        'correct_g2w':       total_correct_g2w,
        'acc_hungarian':     total_correct_hung / total_events,
        'acc_g2w':           total_correct_g2w  / total_events,
        'failed_batches':    failed_batches,
        'num_failed_events': len(all_failures),
    }

    diag = {'scores': diag_scores, 'correct': diag_correct}

    return all_batch_results, all_failures, diag, summary


# ── Main evaluation routine ───────────────────────────────────────────────────

def detailed_test_evaluation(model, val_loader, test_loader,
                             device, hdf5_path, output_dir):
    print("\n" + "=" * 80)
    print("EVALUATION  –  VAL + TEST")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning val set...")
    val_batches, val_failures, val_diag, val_summary = evaluate_loader(
        model, val_loader, device, hdf5_path, output_dir,
        split_name='val', save_heatmaps=False)

    print("Running test set...")
    test_batches, test_failures, test_diag, test_summary = evaluate_loader(
        model, test_loader, device, hdf5_path, output_dir,
        split_name='test', save_heatmaps=True)

    # Histogram uses data already collected in the loops above — no extra pass
    split_data = {'val': val_diag, 'test': test_diag}
    plot_diagonal_histogram(split_data, output_dir)

    # Print summaries
    for summary in [val_summary, test_summary]:
        n = summary['total_events']
        print(f"\n{'='*80}")
        print(f"{summary['split'].upper()} SUMMARY")
        print(f"{'='*80}")
        print(f"Total events         : {n}")
        print(f"Hungarian accuracy   : {summary['correct_hungarian']}/{n} "
              f"({100 * summary['acc_hungarian']:.2f}%)")
        print(f"Top-1 G→W accuracy   : {summary['correct_g2w']}/{n} "
              f"({100 * summary['acc_g2w']:.2f}%)")
        print(f"Batches with failures: {len(summary['failed_batches'])}")
        print(f"Failed events        : {summary['num_failed_events']}")

    # Failure details (test)
    if test_failures:
        print(f"\n{'='*80}")
        print("TEST FAILURE DETAILS  (top 20 lowest self-similarity)")
        print(f"{'='*80}")
        failures_sorted = sorted(test_failures, key=lambda x: x['self_similarity'])

        for i, f in enumerate(failures_sorted[:20], 1):
            meta = f['metadata']
            print(f"\n{i:2d}. Batch {f['batch_num']}, position {f['batch_position']}")
            if 'run' in meta:
                print(f"    Run {meta['run']}, Event {meta['event']}")
            print(f"    Nodes: {meta['num_nodes']}, Edges: {meta['num_edges']}")
            print(f"    Self-similarity      : {f['self_similarity']:.4f}")
            print(f"    Hungarian assignment : {f['hungarian_pred_match']} "
                  f"(score {f['hungarian_similarity']:.4f})")
            print(f"    Rank of correct      : {f['rank_of_correct']}")
            if 'wrong_match_metadata' in f:
                wm = f['wrong_match_metadata']
                if 'run' in wm:
                    print(f"    Matched to           : Run {wm['run']}, "
                          f"Event {wm['event']} ({wm['num_nodes']} nodes)")

        scores_fail = [f['self_similarity'] for f in test_failures]
        ranks_fail  = [f['rank_of_correct']  for f in test_failures]
        top3 = sum(f['top3'] for f in test_failures)
        top5 = sum(f['top5'] for f in test_failures)
        print(f"\nAvg self-similarity  : {np.mean(scores_fail):.4f}")
        print(f"Avg rank of correct  : {np.mean(ranks_fail):.2f}")
        print(f"In top-3             : {top3}/{len(test_failures)} "
              f"({100 * top3 / len(test_failures):.1f}%)")
        print(f"In top-5             : {top5}/{len(test_failures)} "
              f"({100 * top5 / len(test_failures):.1f}%)")
    else:
        failures_sorted = []

    # Save outputs
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    diag_out = {}
    for split_name, diag in split_data.items():
        scores  = np.array(diag['scores'])
        correct = np.array(diag['correct'])
        diag_out[split_name] = {
            'scores':  scores.tolist(),
            'correct': correct.tolist(),
            'stats': {
                'n':    int(len(scores)),
                'mean': float(scores.mean()),
                'std':  float(scores.std()),
                'min':  float(scores.min()),
                'max':  float(scores.max()),
                **{f'p{p}': float(np.percentile(scores, p))
                   for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
            },
        }
    diag_json = output_dir / 'diagonal_scores.json'
    with open(diag_json, 'w') as f:
        json.dump(diag_out, f, indent=2, cls=NumpyEncoder)
    print(f"✓ Diagonal scores      : {diag_json}")

    results_file = output_dir / 'test_batch_results.json'
    with open(results_file, 'w') as f:
        json.dump(test_batches, f, indent=2, cls=NumpyEncoder)
    print(f"✓ Test batch results   : {results_file}")

    if failures_sorted:
        failures_file = output_dir / 'test_failures.json'
        with open(failures_file, 'w') as f:
            json.dump(failures_sorted, f, indent=2, cls=NumpyEncoder)
        print(f"✓ Test failures        : {failures_file}")

    summary_file = output_dir / 'evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("EVALUATION SUMMARY\n" + "=" * 80 + "\n\n")
        for summary in [val_summary, test_summary]:
            n = summary['total_events']
            f.write(f"{summary['split'].upper()}\n")
            f.write(f"  Total events       : {n}\n")
            f.write(f"  Hungarian acc      : "
                    f"{summary['correct_hungarian']}/{n} "
                    f"({100 * summary['acc_hungarian']:.2f}%)\n")
            f.write(f"  Top-1 G→W acc      : "
                    f"{summary['correct_g2w']}/{n} "
                    f"({100 * summary['acc_g2w']:.2f}%)\n")
            f.write(f"  Failed events      : {summary['num_failed_events']}\n\n")
        f.write("Diagonal score stats per split:\n")
        for split_name, info in diag_out.items():
            s = info['stats']
            f.write(f"  {split_name}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                    f"p5={s['p5']:.4f}, p10={s['p10']:.4f}, p25={s['p25']:.4f}\n")
    print(f"✓ Summary              : {summary_file}")
    print(f"✓ Heatmap plots        : {output_dir}")
    print("\n" + "=" * 80)

    return test_batches, test_failures


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.model_path, map_location=device)
    ckpt_args  = checkpoint['args']

    rebin_factor = ckpt_args.get('rebin_factor',  1)
    train_split  = ckpt_args.get('train_split',   args.train_split)
    val_split    = ckpt_args.get('val_split',      args.val_split)
    random_seed  = ckpt_args.get('random_seed',   args.random_seed)
    emb_dim      = ckpt_args.get('emb_dim',        128)
    temperature  = ckpt_args.get('temperature',    0.1)

    print(f"  Epoch        : {checkpoint['epoch']}")
    print(f"  Val loss     : {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Val acc      : {checkpoint.get('val_acc',  'N/A'):.4f}")
    print(f"  rebin_factor : {rebin_factor}")
    print(f"  train_split  : {train_split}  |  val_split : {val_split}")
    print(f"  random_seed  : {random_seed}")

    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        hdf5_path        = args.data_path,
        apply_scaling    = True,
        rebin_factor     = rebin_factor,
        batch_size_train = 32,
        batch_size_val   = args.batch_size_eval,
        batch_size_test  = args.batch_size_eval,
        train_split      = train_split,
        val_split        = val_split,
        num_workers      = args.num_workers,
        pin_memory       = torch.cuda.is_available(),
        random_seed      = random_seed,
    )
    print(f"  Train / Val / Test : "
          f"{len(train_loader.dataset)} / "
          f"{len(val_loader.dataset)} / "
          f"{len(test_loader.dataset)} events")

    model = GraphWaveModel(
        node_in_dim = dataset.node_dim,
        wave_in_dim = dataset.wave_channels,
        emb_dim     = emb_dim,
        temperature = temperature,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nModel loaded  "
          f"({sum(p.numel() for p in model.parameters()):,} parameters)")

    detailed_test_evaluation(
        model, val_loader, test_loader,
        device, args.data_path, output_dir,
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate graph-wave contrastive model with Hungarian accuracy '
                    'and diagonal-score distribution plots.')

    parser.add_argument('--model_path',      type=str, required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--data_path',       type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--output_dir',      type=str,   default='./test_analysis')
    parser.add_argument('--batch_size_eval', type=int,   default=8,
                        help='Batch size for val and test')
    parser.add_argument('--train_split',     type=float, default=0.75,
                        help='Fallback — overridden by checkpoint')
    parser.add_argument('--val_split',       type=float, default=0.20,
                        help='Fallback — overridden by checkpoint')
    parser.add_argument('--random_seed',     type=int,   default=42,
                        help='Fallback — overridden by checkpoint')
    parser.add_argument('--num_workers',     type=int,   default=4)

    args = parser.parse_args()
    main(args)