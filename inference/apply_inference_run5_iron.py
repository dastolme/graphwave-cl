import os
import h5py
import torch
import subprocess
import traceback
import cygno as cy
import numpy as np
from tqdm import tqdm
import inference as infer

# Setup matcher
matcher = infer.GraphWaveformMatcher(device='cuda')
matcher.load_model("best_model.pth")

# Constants
NORM_STATS_PATH = "normalization_stats.pth"
CLOUD_RECO_PATH = "https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/RECO/Run5_Saladin/"
RECO_PATH       = "/raid/home/dastolfo/GraphWave-CL/Run5_LG/"
OUTPUT_H5       = f"{RECO_PATH}run5_iron_lg_score_matrices.h5"

# Read logbook
run5_logbook = cy.read_cygno_logbook(start_run=59253, end_run=95999)
mask = run5_logbook["run_description"].str.startswith("Daily Calibration", na=False)
run_list = run5_logbook.loc[mask, "run_number"].values

# Load norm stats once — reused across all runs without disk I/O
norm_stats = torch.load(NORM_STATS_PATH)
print(f"✓ Loaded normalization stats from: {NORM_STATS_PATH}")

# "a" (append) instead of "w": existing results survive a restart.
with h5py.File(OUTPUT_H5, "a") as h5f:

    # Runs fully committed in a previous session — skip them outright.
    done_runs = {int(k[len("done_run"):]) for k in h5f.attrs if k.startswith("done_run")}
    if done_runs:
        print(f"✓ Resuming: {len(done_runs)} runs already complete, skipping them")

    for run in tqdm(run_list, desc="Runs"):
        run = int(run)

        # ── Skip completed runs BEFORE any download / S3 work ─────────────────
        if run in done_runs:
            continue

        file_url         = f"{CLOUD_RECO_PATH}reco_run{run}_3D.root"
        destination_file = f"{RECO_PATH}reco_run{run}_3D.root"

        try:
            # ── Download ──────────────────────────────────────────────────────
            subprocess.run(["wget", "-nc", "-P", RECO_PATH, file_url])
            # Judge success by the file being present, not by wget's exit code:
            # `-nc` returns non-zero when the file already exists, which is fine.
            if not os.path.exists(destination_file):
                print(f"[WARN] Failed to download {file_url}")
                continue

            # ── Load file & reader ────────────────────────────────────────────
            try:
                reco_file  = infer.RecoFile(run_number=run, path_to_file=RECO_PATH)
                reader     = infer.RecoFileReader(reco_file)
                processor  = infer.RecoFileProcessor(reader)
                processor.load_normalization_stats_from_dict(norm_stats)  # no disk I/O
                cmos_metadata_filtered = reader.cmos_metadata.query('nSc > 0')
            except FileNotFoundError:
                print(f"[WARN] File not found for run {run}, skipping.")
                continue

            # ── Collect all valid events for this run ─────────────────────────
            pending_events, pending_data = [], []
            for event_number in cmos_metadata_filtered['event']:
                try:
                    graphs, cluster_ids, waveforms, trigger_ids = processor.process_event(event_number)
                except (infer.EmptyEventError, ValueError):
                    continue
                pending_events.append(event_number)
                pending_data.append((graphs, cluster_ids, waveforms, trigger_ids))

            # ── Single GPU pass for the whole run, results grouped per event ──
            scored = matcher.score_batch(pending_data)

            # ── Drop any partial groups left by a previous crashed attempt ────
            for stale in [g for g in h5f if g.startswith(f"run{run}_event")]:
                del h5f[stale]

            # ── Write every event for this run ────────────────────────────────
            for event_number, (sim_matrix, cluster_ids, trigger_ids) in zip(pending_events, scored):
                key = f"run{run}_event{int(event_number)}"
                grp = h5f.create_group(key)
                grp.attrs['run']   = run
                grp.attrs['event'] = int(event_number)
                grp.create_dataset('score_matrix', data=sim_matrix.numpy())  # (n_clusters, n_triggers)
                grp.create_dataset('cluster_ids',  data=np.array(cluster_ids))
                grp.create_dataset('trigger_ids',  data=np.array(trigger_ids))

            # ── Commit point: mark run done, then force everything to disk ────
            h5f.attrs[f"done_run{run}"] = True
            h5f.flush()

        except Exception as e:
            # One bad run must not take down the whole batch. Log loudly and move on;
            # nothing was committed for this run, so it'll be retried on the next resume.
            print(f"[ERROR] run {run} failed: {e}")
            traceback.print_exc()
            continue

        finally:
            # ── Clean up local file whether the run succeeded or not ──────────
            if os.path.exists(destination_file):
                subprocess.run(["rm", "-rf", destination_file])

print(f"\n✓ Score matrices saved to: {OUTPUT_H5}")