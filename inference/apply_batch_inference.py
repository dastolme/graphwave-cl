import h5py
import torch
import subprocess
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
RECO_PATH       = "/raid/home/dastolfo/GraphWave-CL/AmBe_LG/"
OUTPUT_H5       = f"{RECO_PATH}ambe_lg_score_matrices.h5"

# Read logbook
ambe_logbook = cy.read_cygno_logbook(start_run=96373, end_run=99248)
run_list = (
    ambe_logbook
    .query("run_description == 'AmBe Campaign'")
    ['run_number']
    .values
)

# Load norm stats once — reused across all runs without disk I/O
norm_stats = torch.load(NORM_STATS_PATH)
print(f"✓ Loaded normalization stats from: {NORM_STATS_PATH}")

with h5py.File(OUTPUT_H5, "w") as h5f:
    for run in tqdm(run_list, desc="Runs"):
        file_url         = f"{CLOUD_RECO_PATH}reco_run{run}_3D.root"
        destination_file = f"{RECO_PATH}reco_run{run}_3D.root"

        # ── Download ──────────────────────────────────────────────────────────
        result = subprocess.run(["wget", "-nc", "-P", RECO_PATH, file_url])
        if result.returncode != 0:
            print(f"[WARN] Failed to download {file_url}")
            continue

        # ── Load file & reader ────────────────────────────────────────────────
        try:
            reco_file  = infer.RecoFile(run_number=run, path_to_file=RECO_PATH)
            reader     = infer.RecoFileReader(reco_file)
            processor  = infer.RecoFileProcessor(reader)
            processor.load_normalization_stats_from_dict(norm_stats)  # no disk I/O
            cmos_metadata_filtered = reader.cmos_metadata.query('nSc > 0')
        except FileNotFoundError:
            print(f"[WARN] File not found for run {run}, skipping.")
            continue

        # ── Collect all valid events for this run ─────────────────────────────
        pending_events, pending_data = [], []
        for event_number in cmos_metadata_filtered['event']:
            try:
                graphs, cluster_ids, waveforms, trigger_ids = processor.process_event(event_number)
            except (infer.EmptyEventError, ValueError):
                continue
            pending_events.append(event_number)
            pending_data.append((graphs, cluster_ids, waveforms, trigger_ids))

        # ── Single GPU pass for the whole run, results grouped per event ──────
        scored = matcher.score_batch(pending_data)
        for event_number, (sim_matrix, cluster_ids, trigger_ids) in zip(pending_events, scored):
            key = f"run{int(run)}_event{int(event_number)}"
            grp = h5f.create_group(key)
            grp.attrs['run']   = int(run)
            grp.attrs['event'] = int(event_number)
            grp.create_dataset('score_matrix', data=sim_matrix.numpy())  # (n_clusters, n_triggers)
            grp.create_dataset('cluster_ids',  data=np.array(cluster_ids))
            grp.create_dataset('trigger_ids',  data=np.array(trigger_ids))

        # ── Clean up ──────────────────────────────────────────────────────────
        result = subprocess.run(["rm", "-rf", destination_file])
        if result.returncode != 0:
            print(f"[WARN] Failed to remove {destination_file}")

print(f"\n✓ Score matrices saved to: {OUTPUT_H5}")