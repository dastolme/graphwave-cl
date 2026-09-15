"""
Batch GraphWave-CL inference driven by a (run, event) preselection.

Instead of scoring every event of every run in the logbook, this reads a table of
(run, event) pairs that survived the upstream cuts and scores only those.

Resume is per-event: a group already present and complete in the HDF5 is never
recomputed, so a crash mid-run costs at most the events of that run that had not
been written yet.
"""

import os
import time
import h5py
import torch
import subprocess
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
import inference as infer

# ── Constants ────────────────────────────────────────────────────────────────
NORM_STATS_PATH = "normalization_stats.pth"
CLOUD_RECO_PATH = "https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/RECO/Run5_Saladin/"
RECO_PATH       = "/raid/home/dastolfo/GraphWave-CL/Run5_LG/"
OUTPUT_H5       = f"{RECO_PATH}test_score.h5"

# Table of surviving (run, event) pairs from the upstream selection.
# Must contain at least the columns `run` and `event`.
PRESELECTION    = "nr_preselection.parquet"

DATASETS = ("score_matrix", "cluster_ids", "trigger_ids")


def load_run_events(path):
    """Return {run: sorted unique event array} from the preselection table."""
    presel = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = {"run", "event"} - set(presel.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")

    presel = presel[["run", "event"]].astype(np.int64).drop_duplicates()
    return {int(r): np.sort(g["event"].to_numpy())
            for r, g in presel.groupby("run", sort=True)}


def complete_keys(h5f):
    """Keys of groups that were fully written (all three datasets present)."""
    return {k for k in h5f
            if isinstance(h5f[k], h5py.Group) and all(d in h5f[k] for d in DATASETS)}


# ── Setup ────────────────────────────────────────────────────────────────────
matcher = infer.GraphWaveformMatcher(device='cuda')
matcher.load_model("best_model.pth")

norm_stats = torch.load(NORM_STATS_PATH)
print(f"✓ Loaded normalization stats from: {NORM_STATS_PATH}")

run_events = load_run_events(PRESELECTION)
n_requested = sum(len(v) for v in run_events.values())
print(f"✓ Preselection: {n_requested} events over {len(run_events)} runs")

t_read = t_proc = t_gpu = t_dl = 0.0
n_scored = 0

with h5py.File(OUTPUT_H5, "a") as h5f:

    # Built once — checking membership per event is O(1) from here on.
    existing = complete_keys(h5f)
    if existing:
        print(f"✓ Resuming: {len(existing)} events already scored, skipping them")

    for run, wanted_events in tqdm(run_events.items(), desc="Runs"):

        # ── Drop events already on disk BEFORE any download / read ───────────
        todo = np.array([e for e in wanted_events
                         if f"run{run}_event{e}" not in existing], dtype=np.int64)
        if todo.size == 0:
            continue

        destination_file = f"{RECO_PATH}reco_run{run}_3D.root"
        file_url         = f"{CLOUD_RECO_PATH}reco_run{run}_3D.root"
        fetched          = False

        try:
            # ── Download only if not already local ───────────────────────────
            if not os.path.exists(destination_file):
                t0 = time.perf_counter()
                subprocess.run(["wget", "-q", "-nc", "-P", RECO_PATH, file_url])
                t_dl += time.perf_counter() - t0
                fetched = True
                # Judge success by the file being present, not by wget's exit
                # code: `-nc` returns non-zero when the file already exists.
                if not os.path.exists(destination_file):
                    print(f"[WARN] Failed to download {file_url}")
                    continue

            # ── Load file & reader ──────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                reco_file  = infer.RecoFile(run_number=run, path_to_file=RECO_PATH)
                reader     = infer.RecoFileReader(reco_file)
                processor  = infer.RecoFileProcessor(reader)
                processor.load_normalization_stats_from_dict(norm_stats)
                cmos_metadata_filtered = reader.cmos_metadata.query('nSc > 0')
            except FileNotFoundError:
                print(f"[WARN] File not found for run {run}, skipping.")
                continue
            t_read += time.perf_counter() - t0

            # ── Keep only requested events that actually exist with nSc > 0 ──
            available = cmos_metadata_filtered['event'].to_numpy().astype(np.int64)
            todo      = np.intersect1d(todo, available, assume_unique=False)
            if todo.size == 0:
                continue

            # ── Build the batch ─────────────────────────────────────────────
            t0 = time.perf_counter()
            pending_events, pending_data = [], []
            for event_number in todo:
                try:
                    graphs, cluster_ids, waveforms, trigger_ids = \
                        processor.process_event(int(event_number))
                except (infer.EmptyEventError, ValueError):
                    continue
                pending_events.append(int(event_number))
                pending_data.append((graphs, cluster_ids, waveforms, trigger_ids))
            t_proc += time.perf_counter() - t0

            if not pending_data:
                continue

            # ── Single GPU pass, results grouped per event ──────────────────
            t0 = time.perf_counter()
            scored = matcher.score_batch(pending_data)
            t_gpu += time.perf_counter() - t0

            # ── Write ───────────────────────────────────────────────────────
            for event_number, (sim_matrix, cluster_ids, trigger_ids) in zip(pending_events, scored):
                key = f"run{run}_event{event_number}"

                # A key that exists but is incomplete is the debris of a crash
                # mid-write; it is the only thing worth deleting.
                if key in h5f:
                    del h5f[key]

                grp = h5f.create_group(key)
                grp.attrs['run']   = run
                grp.attrs['event'] = event_number
                grp.create_dataset('score_matrix', data=sim_matrix.numpy())  # (n_clusters, n_triggers)
                grp.create_dataset('cluster_ids',  data=np.asarray(cluster_ids))
                grp.create_dataset('trigger_ids',  data=np.asarray(trigger_ids))
                existing.add(key)
                n_scored += 1

            h5f.flush()

        except Exception as e:
            # One bad run must not take down the whole batch. Nothing incomplete
            # was committed, so the missing events are retried on the next resume.
            print(f"[ERROR] run {run} failed: {e}")
            traceback.print_exc()
            continue

        finally:
            # Only remove what this session downloaded — never a pre-existing
            # local copy.
            if fetched and os.path.exists(destination_file):
                os.remove(destination_file)

print(f"\n✓ {n_scored} events scored this session → {OUTPUT_H5}")
print(f"  download {t_dl:7.1f} s | read {t_read:7.1f} s | "
      f"process {t_proc:7.1f} s | gpu {t_gpu:7.1f} s")