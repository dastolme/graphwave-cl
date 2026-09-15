"""
Check the matcher against pedestal clusters. Standalone — drop it in the
GraphWave-CL directory, next to model.py and data_loader.py, and run it there.

    python check_matching.py \
        --data_path data/dataset_rebinned.h5 \
        --pedestal_path data/dataset_pedestal.h5 \
        --checkpoint results/best_model.pth \
        --norm_stats results/normalization_stats.pth

Embeds the VALIDATION split's pairs and the pedestal graphs ONCE, then every
later question is numpy. That is exact, not an approximation: nothing in either
encoder mixes samples — GCNConv passes messages inside each graph of a disjoint
union, global_mean_pool is per graph, the Conv1d stack and the Linear act per
sample, and there is no BatchNorm or dropout — so a score assembled from cached
embeddings is bit-for-bit what an 8x8 forward pass would have produced.
(Verified: encoding the same graphs at batch size 8, 32, 64 and in shuffled
order gives max|diff| = 0.)

Writes embeddings.npz so the later questions cost nothing to ask again.

Read the output in this order:

  1. the dataset fingerprint. normalization_stats.pth holds min/max over the
     WHOLE dataset, so recomputing them must reproduce the stored values. If
     they differ this is not the file the model was trained on, and the
     "validation" split below overlaps the training data.
  2. the OOD table. If the two pools barely overlap in scaled intensity, the
     encoder can separate them on brightness alone and the null measures that,
     not whether the model recognises junk.
  3. the score distributions: true vs pedestal vs cross-event.
  4. the dilution curve. Not how high junk scores — how often junk WINS, as a
     function of how much of it there is. That is the number which transfers to
     a sample with a different junk density.
  5. the diluted batches. The validation protocol gave every waveform a
     counterpart; here most have none, which is the deployment case.
"""
# postpones annotation evaluation, so `str | Path` and `list[str]` work on
# Python 3.8/3.9 as well as 3.10+
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


class H5Pairs:
    """Raw contents of one dataset file, in file-key order.

    Key order matters: HDF5GraphWaveDataset builds its sample list from
    list(f.keys()), and random_split indexes into that list. Reproducing the
    validation subset means reproducing this order, so it is preserved here
    rather than sorted.
    """

    def __init__(self, path: str | Path, keys: Optional[list[str]] = None):
        import h5py

        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.keys = list(f.keys()) if keys is None else list(keys)
            self.nodes, self.edges, self.waves = [], [], []
            for k in self.keys:
                g = f[k]
                self.nodes.append(np.asarray(g["graph_x"]))
                self.edges.append(np.asarray(g["graph_edge_index"]))
                # A pedestal cluster has no trigger, so the file may carry no
                # waveforms at all. Only the graphs are needed from such a
                # pool — it supplies candidates, never queries.
                if "waveforms" in g:
                    self.waves.append(np.asarray(g["waveforms"]))
        self.node_dim = self.nodes[0].shape[1]
        self.has_waves = len(self.waves) == len(self.keys) and bool(self.waves)
        if self.has_waves:
            self.wave_length, self.wave_channels = self.waves[0].shape
        else:
            self.wave_length = self.wave_channels = None
            if self.waves:
                raise ValueError(
                    f"{self.path}: {len(self.waves)} of {len(self.keys)} "
                    f"samples have waveforms — a partially populated file is "
                    f"more likely a broken write than a design, and the "
                    f"indices would not line up with the graphs.")
            self.waves = []

    def __len__(self) -> int:
        return len(self.keys)

    def subset(self, idx) -> "H5Pairs":
        out = object.__new__(H5Pairs)
        out.path = self.path
        idx = list(idx)
        out.keys = [self.keys[i] for i in idx]
        out.nodes = [self.nodes[i] for i in idx]
        out.edges = [self.edges[i] for i in idx]
        out.waves = [self.waves[i] for i in idx] if self.has_waves else []
        out.node_dim = self.node_dim
        out.has_waves = self.has_waves
        out.wave_length, out.wave_channels = self.wave_length, self.wave_channels
        return out

    def all_nodes(self) -> np.ndarray:
        """Every node of every sample, unscaled — for ood_report."""
        return np.concatenate(self.nodes)

    # ── torch-side ────────────────────────────────────────────────────────
    def graphs(self, stats: dict):
        """Scaled PyG Data objects, using the supplied constants."""
        import torch
        from torch_geometric.data import Data

        return [Data(x=torch.tensor(scale_node_features(n, stats),
                                    dtype=torch.float32),
                     edge_index=torch.tensor(e, dtype=torch.long))
                for n, e in zip(self.nodes, self.edges)]

    def waveforms(self, stats: dict):
        """Scaled waveform tensor (n_samples, channels, length).

        Note the permute: the file stores (length, channels) and the model's
        Conv1d wants (channels, length), exactly as __getitem__ does it.
        """
        import torch

        if not self.has_waves:
            raise ValueError(f"{self.path} carries no waveforms — it can "
                             f"supply candidate graphs but not queries.")
        w = torch.tensor(np.stack(self.waves), dtype=torch.float32)
        rng_ = max(float(stats["wave_max"]) - float(stats["wave_min"]), 1e-6)
        w = (w - float(stats["wave_min"])) / rng_
        return w.permute(0, 2, 1)


def validation_indices(n_total: int, train_split: float = 0.75,
                       val_split: float = 0.20, random_seed: int = 42
                       ) -> list[int]:
    """The exact indices create_dataloaders put in the validation split.

    Uses torch.utils.data.random_split on a range of the same length with the
    same seed, rather than reimplementing the permutation — the split is only
    reproducible if it is produced the same way, and a re-drawn "validation"
    set contains data the model trained on.
    """
    import torch

    train_size = int(train_split * n_total)
    val_size = int(val_split * n_total)
    test_size = n_total - train_size - val_size
    gen = torch.Generator().manual_seed(random_seed)
    _, val, _ = torch.utils.data.random_split(
        range(n_total), [train_size, val_size, test_size], generator=gen)
    return list(val.indices)


def load_norm_stats(path) -> dict:
    """normalization_stats.pth, as train.py wrote it.

    These constants must be REUSED for any new pool, never recomputed on it.
    Recomputing would map a pedestal pool's own range onto [0, 1] and hand the
    encoder a distribution that looks like the training one — which is the
    opposite of what the null is supposed to measure.
    """
    import torch
    return torch.load(path, map_location="cpu")


def scale_node_features(nodes: np.ndarray, stats: dict) -> np.ndarray:
    """HDF5GraphWaveDataset._apply_node_scaling, on numpy.

    Columns 0 and 1 are pixel coordinates mapped to [-1, 1] by the rebinned
    sensor side; column 2 is intensity mapped by the dataset's global min/max.

    Note what is NOT here: a clamp. A cluster fainter than anything in the
    training set lands below 0, brighter lands above 1. That is faithful to the
    loader, and it is exactly why ood_report() exists — a pedestal pool sitting
    outside the trained range scores low because the encoder has never seen
    such inputs, not because it recognises them as junk, and a null built that
    way is optimistic.
    """
    out = np.asarray(nodes, dtype=np.float64).copy()
    side = stats["TOTAL_PIXEL_SIDE"]
    out[:, 0] = 2 * (out[:, 0] / side) - 1
    out[:, 1] = 2 * (out[:, 1] / side) - 1
    rng_ = max(stats["int_max"] - stats["int_min"], 1e-6)
    out[:, 2] = (out[:, 2] - stats["int_min"]) / rng_
    return out


def ood_report(val_nodes: np.ndarray, pool_nodes: np.ndarray,
               stats: dict) -> pd.DataFrame:
    """Are the pool's scaled features inside the range the encoder was trained
    on? Compares the validation clusters against the candidate pool.

    Read the `frac_outside` row first. If a large fraction of pedestal node
    intensities fall outside [0, 1] while the validation ones do not, the two
    populations are separable on scaling alone and any threshold derived from
    the null is measuring the extrapolation, not the physics.
    """
    rows = []
    for label, nodes in (("validation", val_nodes), ("pool", pool_nodes)):
        sc = scale_node_features(nodes, stats)
        inten = sc[:, 2]
        rows.append({
            "population": label, "n_nodes": len(sc),
            "x_min": sc[:, 0].min(), "x_max": sc[:, 0].max(),
            "int_p1": np.percentile(inten, 1),
            "int_med": np.median(inten),
            "int_p99": np.percentile(inten, 99),
            "frac_outside": float(((inten < 0) | (inten > 1)).mean()),
            # overlap of the two intensity distributions, 1 = identical.
            # frac_outside alone misses the case that matters here: with
            # int_max set by one bright outlier, EVERY cluster scales to a
            # small number, so a pedestal pool ten times fainter than the
            # training clusters is still comfortably inside [0, 1] and looks
            # fine by that test while being trivially separable.
            "_inten": inten,
        })
    df = pd.DataFrame(rows)
    a, b = df.pop("_inten").tolist()
    lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, 200) if hi > lo else np.array([lo, lo + 1])
    ha = np.histogram(a, edges, density=True)[0]
    hb = np.histogram(b, edges, density=True)[0]
    df["overlap"] = float(np.minimum(ha, hb).sum() * np.diff(edges)[0])
    return df


def make_graphs(node_features: list[np.ndarray], edge_indices: list[np.ndarray],
                stats: dict):
    """Scaled PyG Data objects from raw (x, y, intensity, ...) node arrays.

    edge_indices must come from the SAME construction that built the training
    HDF5 — the rule lives in whatever script wrote graph_edge_index, not in
    data_loader.py, which only reads it. A pool wired up differently is a pool
    the encoder has never seen.
    """
    import torch
    from torch_geometric.data import Data

    return [Data(x=torch.tensor(scale_node_features(n, stats),
                                dtype=torch.float32),
                 edge_index=torch.tensor(e, dtype=torch.long))
            for n, e in zip(node_features, edge_indices)]


# ══════════════════════════════════════════════════════════════════════════
# scores — pure numpy from here on
# ══════════════════════════════════════════════════════════════════════════

def logits(zG: np.ndarray, zW: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """(n_graphs, n_waves), exactly as GraphWaveModel.forward computes it."""
    return (zG @ zW.T) / tau


def null_scores(zW: np.ndarray, zP: np.ndarray, tau: float = 0.1,
                n_pairs: int = 200_000, seed: int = 0) -> np.ndarray:
    """Scores of definitionally-wrong pairs: true waveforms against pedestal
    clusters. A floor on the wrong-score distribution, not the whole of it —
    a physics image also holds real out-of-time clusters, which carry real
    light and score higher than sensor noise. Bracket with cross_null().
    """
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(zW), n_pairs)
    j = rng.integers(0, len(zP), n_pairs)
    return np.einsum("ij,ij->i", zW[i], zP[j]) / tau


def cross_null(zW: np.ndarray, zG: np.ndarray, tau: float = 0.1,
               n_pairs: int = 200_000, seed: int = 0) -> np.ndarray:
    """Scores of true waveforms against REAL clusters belonging to a different
    pair. Overshoots the wrong-score distribution — some of these are genuine
    counterparts of something — so the truth sits between this and
    null_scores(). If the two land in the same place, the distinction stops
    mattering.
    """
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(zW), n_pairs)
    j = rng.integers(0, len(zG), n_pairs)
    same = i == j
    j[same] = (j[same] + 1) % len(zG)
    return np.einsum("ij,ij->i", zW[i], zG[j]) / tau


def true_scores(zW: np.ndarray, zG: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Score of each true pair — the diagonal."""
    return np.einsum("ij,ij->i", zW, zG) / tau


def threshold_at(null: np.ndarray, far: float = 0.01) -> float:
    """The score s0 at which a fraction `far` of wrong pairs survive.

    Calibrated on the null OF THE SAMPLE IT WILL BE APPLIED TO. A threshold
    carried over from iron is a threshold set on a different noise population,
    a different occupancy and a different set of detector conditions.
    """
    return float(np.quantile(null, 1.0 - far))


def efficiency_at(true: np.ndarray, s0: float) -> float:
    """Fraction of true pairs that survive s0. The other half of the story:
    a threshold is only as good as the efficiency it costs."""
    return float((true >= s0).mean())


# ══════════════════════════════════════════════════════════════════════════
# dilution
# ══════════════════════════════════════════════════════════════════════════

def dilution_curve(zW: np.ndarray, zG: np.ndarray, zP: np.ndarray,
                   pool_sizes: Iterable[int] = (1, 2, 5, 10, 20, 50, 100),
                   n_trials: int = 2000, tau: float = 0.1,
                   seed: int = 0) -> pd.DataFrame:
    """Accuracy when one true cluster hides among N pedestal ones.

    The question a null distribution alone cannot answer: not how high junk
    scores, but how often junk WINS. That is what enters the count, and it is
    what differs between two samples whose images have different junk
    densities. If this curve is flat across the range of pool sizes your two
    samples span, the occupancy difference does not matter; if it falls, the
    correction is read straight off it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n_pool in pool_sizes:
        idx = rng.integers(0, len(zW), n_trials)
        s_true = np.einsum("ij,ij->i", zW[idx], zG[idx]) / tau
        # the best of n_pool independent pedestal distractors per trial
        best_junk = np.full(n_trials, -np.inf)
        for _ in range(int(n_pool)):
            j = rng.integers(0, len(zP), n_trials)
            best_junk = np.maximum(best_junk,
                                   np.einsum("ij,ij->i", zW[idx], zP[j]) / tau)
        won = s_true > best_junk
        rows.append({
            "n_pedestal": int(n_pool),
            "top1_acc": float(won.mean()),
            "true_med": float(np.median(s_true)),
            "best_junk_med": float(np.median(best_junk)),
            "margin_med": float(np.median(s_true - best_junk)),
            "margin_p10": float(np.percentile(s_true - best_junk, 10)),
        })
    return pd.DataFrame(rows)


def batch_trials(zW: np.ndarray, zG: np.ndarray, zP: np.ndarray,
                 n_true: int = 2, n_pool: int = 8, n_trials: int = 1000,
                 tau: float = 0.1, s0: Optional[float] = None,
                 seed: int = 0) -> pd.DataFrame:
    """The 8x8-style test: n_true true pairs and (n_pool - n_true) pedestal
    clusters, assigned with the Hungarian algorithm.

    Deliberately NOT the validation protocol. There the matrix was square with
    a perfect assignment guaranteed to exist, so every row had a right answer
    and the solver could only redistribute. Here most waveforms have no
    counterpart in the batch at all, which is the deployment case, and the
    interesting failure is a waveform confidently taking a pedestal cluster.

    Returns one row per trial:
        n_correct      true pairs recovered (of n_true)
        n_false        waveforms with no counterpart that took a cluster anyway
                       (only meaningful when s0 is set — without a rejection
                       rule EVERY such waveform is assigned something)
        best_true, best_false   scores of the two populations
    """
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(seed)
    n_wave = n_pool                     # square batch, as in validation
    n_junk = n_pool - n_true
    if n_junk < 0:
        raise ValueError("n_true cannot exceed n_pool")

    rows = []
    for _ in range(n_trials):
        w_idx = rng.choice(len(zW), n_wave, replace=False)
        t_idx = w_idx[:n_true]                       # their true clusters
        p_idx = rng.choice(len(zP), n_junk, replace=False)

        G = np.vstack([zG[t_idx], zP[p_idx]])        # rows: clusters
        W = zW[w_idx]                                # cols: waveforms
        S = (G @ W.T) / tau

        cost = -S.astype(np.float64)
        if s0 is not None:
            cost = np.vstack([cost, np.full((n_wave, n_wave), -float(s0))])
        r, c = linear_sum_assignment(cost)
        assign = np.full(n_wave, -1)
        for rr, cc in zip(r, c):
            assign[cc] = rr

        has_truth = np.arange(n_wave) < n_true       # by construction
        correct = (assign[:n_true] == np.arange(n_true))
        took_real = (assign >= 0) & (assign < S.shape[0])
        false_pos = took_real & ~has_truth

        rows.append({
            "n_correct": int(correct.sum()),
            "n_true": n_true,
            "n_false": int(false_pos.sum()),
            "n_orphan": int((~has_truth).sum()),
            "best_true": float(np.median(S[np.arange(n_true), np.arange(n_true)])),
            "best_false": float(np.median(S[:, n_true:].max(axis=0)))
            if n_wave > n_true else np.nan,
        })
    df = pd.DataFrame(rows)
    n_orphan = int(df["n_orphan"].sum())
    df.attrs["summary"] = {
        "recall": float(df["n_correct"].sum() / df["n_true"].sum()),
        # n_true == n_pool is the validation protocol: every waveform has its
        # counterpart, so there are no orphans and no false-assignment rate to
        # quote. That is precisely the case the 98% was measured in.
        "false_per_orphan": (float(df["n_false"].sum() / n_orphan)
                             if n_orphan else float("nan")),
        "n_orphan": n_orphan,
    }
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Recall and false-assignment rate from batch_trials output."""
    return df.attrs.get("summary", {})


def verify_dataset(pairs: H5Pairs, stats: dict, tol: float = 1e-4) -> bool:
    """Is this the file the checkpoint was trained on?

    normalization_stats.pth holds min/max computed over the WHOLE dataset, so
    recomputing them here must reproduce the stored values. If they differ,
    this is a different file — and then validation_indices() returns a split of
    a different population, so part of the "held-out" sample is data the model
    trained on and every number below is optimistic for the dullest possible
    reason.
    """
    inten = np.concatenate([n[:, 2] for n in pairs.nodes])
    got = {"int_min": float(inten.min()), "int_max": float(inten.max())}
    if pairs.has_waves:
        waves = np.concatenate([w.ravel() for w in pairs.waves])
        got.update({"wave_min": float(waves.min()),
                    "wave_max": float(waves.max())})   # dict |= is 3.9+
    rows, ok = [], True
    for k, v in got.items():
        want = stats.get(k)
        if want is None:
            continue
        want = float(want)
        scale = max(abs(want), 1.0)
        match = abs(v - want) / scale < tol
        ok &= match
        rows.append({"stat": k, "stored": want, "recomputed": v,
                     "match": match})
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda x: f"{x:.6f}"))
    return ok


def main(args):
    import torch
    from model import GraphWaveModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_norm_stats(args.norm_stats)
    print(f"scaling constants from {args.norm_stats}:")
    print(f"  intensity [{stats['int_min']:.4f}, {stats['int_max']:.4f}]  "
          f"waveform [{stats['wave_min']:.4f}, {stats['wave_max']:.4f}]  "
          f"side {stats['TOTAL_PIXEL_SIDE']}")
    if stats.get("rebin_factor") not in (None, args.rebin_factor):
        raise SystemExit(
            f"checkpoint was trained with rebin_factor="
            f"{stats['rebin_factor']}, you passed {args.rebin_factor} — the "
            f"coordinate scaling would differ between the pools.")

    # ── data ──────────────────────────────────────────────────────────────
    print(f"\nloading {args.data_path}")
    full = H5Pairs(args.data_path)
    print("\n── is this the dataset the checkpoint was trained on? ────")
    if not verify_dataset(full, stats):
        msg = ("the recomputed min/max do not match normalization_stats.pth — "
               "this is not the file the model was trained on, so the "
               "validation split below is a split of a different population "
               "and will overlap the training data.")
        if args.force:
            print(f"  ⚠ {msg}\n  continuing because --force was given.")
        else:
            raise SystemExit(f"  {msg}\n  The checkpoint records the path it "
                             f"used: torch.load(ck)['args']['data_path'].\n"
                             f"  Pass --force to proceed anyway.")
    val_idx = validation_indices(len(full), args.train_split, args.val_split,
                                 args.random_seed)
    val = full.subset(val_idx)
    print(f"  {len(full)} samples, validation split {len(val)}")

    print(f"loading {args.pedestal_path}")
    ped = H5Pairs(args.pedestal_path)
    print(f"  {len(ped)} pedestal clusters"
          + ("" if ped.has_waves else " (no waveforms — graphs only, which is "
                                      "all a candidate pool needs)"))
    if ped.node_dim != full.node_dim:
        raise SystemExit(f"node_dim differs: {full.node_dim} vs {ped.node_dim}")

    # ── 1. are the two pools even comparable? ─────────────────────────────
    print("\n── scaled node features ──────────────────────────────────")
    ood = ood_report(val.all_nodes(), ped.all_nodes(), stats)
    print(ood.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if ood.loc[1, "frac_outside"] > 0.2 and ood.loc[0, "frac_outside"] < 0.05:
        print("\n  ⚠ the pedestal pool sits largely outside the trained "
              "intensity range while the validation pool does not. A low null "
              "score then reflects extrapolation, not recognition — treat the "
              "pedestal null as a LOWER bound and read the cross-event one.")
    if abs(ood.loc[1, "x_max"]) > 1.01 or abs(ood.loc[1, "x_min"]) > 1.01:
        print("\n  ⚠ pedestal coordinates fall outside [-1, 1] after scaling "
              f"by TOTAL_PIXEL_SIDE={stats['TOTAL_PIXEL_SIDE']}. The pedestal "
              f"file was almost certainly built without the rebin_factor="
              f"{stats.get('rebin_factor')} applied to the training set.")
    if ood.loc[0, "overlap"] < 0.3:
        print(f"\n  ⚠ the two intensity distributions barely overlap "
              f"({ood.loc[0, 'overlap']:.2f}). Both may sit inside [0, 1] — "
              f"int_max is set by one bright cluster, so everything scales "
              f"small — but the encoder can separate these pools on intensity "
              f"alone. The null then measures how different the populations "
              f"are, which is not the same as how well the model recognises "
              f"junk. Compare the medians, not just frac_outside.")

    # ── 2. embed, once ────────────────────────────────────────────────────
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = GraphWaveModel(node_in_dim=ck["node_in_dim"],
                           wave_in_dim=ck["wave_in_dim"],
                           emb_dim=ck["args"]["emb_dim"],
                           temperature=ck["args"]["temperature"])
    model.load_state_dict(ck["model_state_dict"])
    tau = ck["args"]["temperature"]
    print(f"\ncheckpoint epoch {ck['epoch']}, val_loss {ck['val_loss']:.4f}, "
          f"hungarian {ck.get('hung_val_acc', float('nan')):.4f}")
    print(f"  wave_in_dim {ck['wave_in_dim']}  node_in_dim {ck['node_in_dim']}")
    tr_args = ck.get("args", {})
    print(f"  trained on: {tr_args.get('data_path')}  "
          f"rebin {tr_args.get('rebin_factor')}")
    for k in ("train_split", "val_split", "random_seed"):
        if k in tr_args and getattr(args, k) != tr_args[k]:
            raise SystemExit(
                f"  {k}={getattr(args, k)} but the checkpoint was trained with "
                f"{tr_args[k]} — the validation split would not be the same "
                f"one. Pass --{k} {tr_args[k]}.")

    from torch_geometric.loader import DataLoader as GeoLoader

    def encode_graphs(pairs):
        model.eval().to(device)
        out = []
        with torch.no_grad():
            for b in GeoLoader(pairs.graphs(stats), batch_size=256,
                               shuffle=False):
                out.append(model.encode_graphs(b.to(device)).cpu().numpy())
        return np.concatenate(out)

    def encode_waves(pairs):
        model.eval().to(device)
        w = pairs.waveforms(stats)
        out = []
        with torch.no_grad():
            for i in range(0, len(w), 256):
                out.append(model.encode_waves(w[i:i + 256].to(device))
                           .cpu().numpy())
        return np.concatenate(out)

    print("\nembedding (once — batch composition cannot change these)")
    zG = encode_graphs(val)
    zW = encode_waves(val)
    zP = encode_graphs(ped)
    np.savez(out_dir / "embeddings.npz", zG=zG, zW=zW, zP=zP,
             val_idx=np.array(val_idx), tau=tau)
    print(f"  {zW.shape[0]} validation pairs, {zP.shape[0]} pedestal clusters "
          f"-> {out_dir / 'embeddings.npz'}")

    # ── 3. scores ─────────────────────────────────────────────────────────
    true = true_scores(zW, zG, tau)
    null = null_scores(zW, zP, tau)
    cross = cross_null(zW, zG, tau)
    print("\n── score distributions ───────────────────────────────────")
    print(pd.DataFrame({
        "population": ["true pair", "pedestal null", "cross-event null"],
        "median": [np.median(true), np.median(null), np.median(cross)],
        "p90": [np.percentile(true, 90), np.percentile(null, 90),
                np.percentile(cross, 90)],
        "p99": [np.percentile(true, 99), np.percentile(null, 99),
                np.percentile(cross, 99)],
    }).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n── rejection threshold ───────────────────────────────────")
    rows = []
    for far in (0.10, 0.05, 0.01, 0.001):
        for name, nl in (("pedestal", null), ("cross-event", cross)):
            s0 = threshold_at(nl, far)
            rows.append({"null": name, "false_accept": far, "s0": s0,
                         "true_efficiency": efficiency_at(true, s0)})
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))
    s0 = threshold_at(cross, args.far)
    print(f"\n  using s0 = {s0:.3f} (cross-event null at "
          f"{100 * args.far:g}% false accept)")

    # ── 4. dilution ───────────────────────────────────────────────────────
    print("\n── dilution: one true cluster among N pedestal ───────────")
    curve = dilution_curve(zW, zG, zP, tau=tau,
                              pool_sizes=args.pool_sizes, n_trials=args.n_trials)
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    curve.to_csv(out_dir / "dilution_curve.csv", index=False)

    # ── 5. diluted batches ────────────────────────────────────────────────
    print(f"\n── {args.n_pool}x{args.n_pool} batches ────────────────────")
    rows = []
    for n_true in args.n_true:
        for lab, thr in (("none", None), (f"s0={s0:.2f}", s0)):
            d = batch_trials(zW, zG, zP, n_true=n_true, n_pool=args.n_pool,
                                n_trials=args.n_trials, tau=tau, s0=thr)
            r = summarise(d)
            rows.append({"n_true": n_true, "n_pedestal": args.n_pool - n_true,
                         "rejection": lab, "recall": r["recall"],
                         "false_per_orphan": r["false_per_orphan"]})
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.3f}"))
    pd.DataFrame(rows).to_csv(out_dir / "batch_trials.csv", index=False)
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_path", required=True,
                    help="the HDF5 the model was trained on")
    ap.add_argument("--pedestal_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--norm_stats", required=True)
    ap.add_argument("--output_dir", default="./matching_check")
    ap.add_argument("--rebin_factor", type=int, default=1)
    # must match the training run, or the validation split is not the same one
    ap.add_argument("--train_split", type=float, default=0.75)
    ap.add_argument("--val_split", type=float, default=0.20)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the dataset fingerprint disagrees")
    ap.add_argument("--far", type=float, default=0.01,
                    help="false-accept rate used to pick s0")
    ap.add_argument("--n_pool", type=int, default=8)
    ap.add_argument("--n_true", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--pool_sizes", type=int, nargs="+",
                    default=[1, 2, 5, 10, 20, 50, 100])
    ap.add_argument("--n_trials", type=int, default=5000)
    main(ap.parse_args())