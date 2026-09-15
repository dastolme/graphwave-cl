#!/usr/bin/env python3
"""GraphWave-CL batch inference over the Run4 high-gain Daily Calibration runs.

Downloads one reco per run, scores every event on the GPU, writes the
similarity matrices to HDF5, deletes the ROOT. Resumable: a run is only marked
done after its events are committed and flushed.

IMPORTANT — which prefix to score
---------------------------------
The model reads clusters (Events) and waveforms (PMT_Events) from the SAME
file, so the matrices it produces are only meaningful against a store whose
cluster_id and trigger_id come from that same reconstruction.

The Run4 store was built with clusters from Run4/ and waveforms from
Run4_Saladin/. Matrices from either prefix therefore line up on one side and
not the other, and a trigger renumbering between the two reconstructions
would join the wrong waveform to the right cluster with no visible symptom.

Two consistent options:
  (a) score Run4/ AND rebuild the store from Run4/ alone (WF_REMOTE = None).
      Gets all 167 runs, one reconstruction throughout. Recommended unless
      something is specifically wrong with Run4/ waveforms.
  (b) score Run4_Saladin/ AND rebuild the store from Run4_Saladin/ alone.
      53 runs, but the newer reconstruction.
Either way both sides come from one prefix. Set PREFIX accordingly.
"""
import os
import subprocess
import traceback

import cygno as cy
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List

import inference as infer

# ══════════════════════════════════════════════════════════════════════════
PREFIX = "Run4_Saladin"                # holds 53 of the 167 iron runs (48593-49955)
CLOUD_RECO_PATH = f"https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/RECO/{PREFIX}/"
RECO_PATH = "/raid/home/dastolfo/GraphWave-CL/Run4_HG/"
OUTPUT_H5 = f"{RECO_PATH}run4_iron_hg_score_matrices.h5"

NORM_STATS_PATH = "normalization_stats.pth"
MODEL_PATH = "best_model.pth"

# Run selection. If the catalogue built by build_store_run4_hg.py is reachable
# here, it is authoritative — it is exactly the set the store was built from,
# already filtered for gain and pedestal runs. Otherwise fall back to the
# logbook with the same filters.
RUN_CATALOG = f"{RECO_PATH}run4_hg_catalog.csv"
RUN_MIN, RUN_MAX = 48500, 55093
DESCRIPTION_PREFIX = "Daily Calibration"
EXCLUDE_TOKEN = "Low Gain"             # the low-gain runs share the prefix
DROP_PARKING = True                    # source retracted, no step — not iron
# ══════════════════════════════════════════════════════════════════════════


def run_list() -> List[int]:
    if os.path.exists(RUN_CATALOG):
        cat = pd.read_csv(RUN_CATALOG)
        n_all = len(cat)
        if DROP_PARKING:
            # filter on the parsed step when the catalogue carries it, on the
            # description otherwise — an older catalogue may predate the column
            if "step" in cat.columns:
                cat = cat[cat["step"] > 0]
            else:
                cat = cat[~cat["run_description"].str.contains(
                    "parking", case=False, na=False)]
        runs = sorted(int(r) for r in cat["run_number"])
        print(f"✓ {len(runs)} runs from {RUN_CATALOG} ({runs[0]}-{runs[-1]})")
        if DROP_PARKING and len(runs) < n_all:
            print(f"  dropped {n_all - len(runs)} parking runs")
        return runs

    lb = cy.read_cygno_logbook(start_run=RUN_MIN, end_run=RUN_MAX)
    desc = lb["run_description"].fillna("")
    mask = desc.str.startswith(DESCRIPTION_PREFIX) & (lb["pedestal_run"] == 0)
    if EXCLUDE_TOKEN:
        mask &= ~desc.str.contains(EXCLUDE_TOKEN, case=False, na=False)
    if DROP_PARKING:
        mask &= ~desc.str.contains("parking", case=False, na=False)
    runs = sorted(int(r) for r in lb.loc[mask, "run_number"])
    print(f"✓ {len(runs)} runs from the logbook ({runs[0]}-{runs[-1]})")
    print(f"  (no catalogue at {RUN_CATALOG}; copy it over to guarantee the "
          f"same set the store was built from)")
    return runs


matcher = infer.GraphWaveformMatcher(device="cuda")
matcher.load_model(MODEL_PATH)

norm_stats = torch.load(NORM_STATS_PATH)
print(f"✓ Loaded normalization stats from: {NORM_STATS_PATH}")

os.makedirs(RECO_PATH, exist_ok=True)
runs_to_do = run_list()
print(f"✓ Scoring {PREFIX} -> {OUTPUT_H5}")

with h5py.File(OUTPUT_H5, "a") as h5f:

    # Record which reconstruction these matrices came from, and refuse to mix.
    # Matrices from two prefixes in one file cannot be told apart afterwards,
    # and their cluster_id/trigger_id mean different things.
    if "prefix" in h5f.attrs:
        if h5f.attrs["prefix"] != PREFIX:
            raise SystemExit(
                f"{OUTPUT_H5} holds matrices scored from "
                f"{h5f.attrs['prefix']!r}, not {PREFIX!r}. Use a separate file "
                f"per prefix — the ids are not interchangeable."
            )
    else:
        h5f.attrs["prefix"] = PREFIX

    done_runs = {int(k[len("done_run"):]) for k in h5f.attrs
                 if k.startswith("done_run")}
    if done_runs:
        print(f"✓ Resuming: {len(done_runs)} runs already complete, skipping them")

    for run in tqdm(runs_to_do, desc="Runs"):
        run = int(run)

        # ── Skip completed runs BEFORE any download / S3 work ─────────────────
        if run in done_runs:
            continue

        file_url = f"{CLOUD_RECO_PATH}reco_run{run}_3D.root"
        destination_file = f"{RECO_PATH}reco_run{run}_3D.root"

        try:
            # ── Download ──────────────────────────────────────────────────────
            subprocess.run(["wget", "-nc", "-P", RECO_PATH, file_url])
            # Judge success by the file being present, not by wget's exit code:
            # `-nc` returns non-zero when the file already exists, which is fine.
            if not os.path.exists(destination_file):
                # Expected for runs absent from this prefix — Run4_Saladin holds
                # only part of the sample. Not an error, just skip.
                print(f"[WARN] not on {PREFIX}: run {run}")
                continue

            # ── Load file & reader ────────────────────────────────────────────
            try:
                reco_file = infer.RecoFile(run_number=run, path_to_file=RECO_PATH)
                reader = infer.RecoFileReader(reco_file)
                processor = infer.RecoFileProcessor(reader)
                processor.load_normalization_stats_from_dict(norm_stats)
                cmos_metadata_filtered = reader.cmos_metadata.query("nSc > 0")
            except FileNotFoundError:
                print(f"[WARN] File not found for run {run}, skipping.")
                continue

            # ── Collect all valid events for this run ─────────────────────────
            pending_events, pending_data = [], []
            for event_number in cmos_metadata_filtered["event"]:
                try:
                    graphs, cluster_ids, waveforms, trigger_ids = \
                        processor.process_event(event_number)
                except (infer.EmptyEventError, ValueError):
                    continue
                pending_events.append(event_number)
                pending_data.append((graphs, cluster_ids, waveforms, trigger_ids))

            if not pending_data:
                # nothing scorable, but the run IS done — mark it so a resume
                # does not download it again
                h5f.attrs[f"done_run{run}"] = True
                h5f.flush()
                continue

            # ── Single GPU pass for the whole run, results grouped per event ──
            scored = matcher.score_batch(pending_data)

            # ── Drop any partial groups left by a previous crashed attempt ────
            for stale in [g for g in h5f if g.startswith(f"run{run}_event")]:
                del h5f[stale]

            # ── Write every event for this run ────────────────────────────────
            for event_number, (sim_matrix, cluster_ids, trigger_ids) in zip(
                    pending_events, scored):
                key = f"run{run}_event{int(event_number)}"
                grp = h5f.create_group(key)
                grp.attrs["run"] = run
                grp.attrs["event"] = int(event_number)
                grp.create_dataset("score_matrix", data=sim_matrix.numpy())
                grp.create_dataset("cluster_ids", data=np.array(cluster_ids))
                grp.create_dataset("trigger_ids", data=np.array(trigger_ids))

            # ── Commit point: mark run done, then force everything to disk ────
            h5f.attrs[f"done_run{run}"] = True
            h5f.flush()

        except Exception as e:
            # One bad run must not take down the whole batch. Log loudly and
            # move on; nothing was committed, so a resume retries it.
            print(f"[ERROR] run {run} failed: {e}")
            traceback.print_exc()
            continue

        finally:
            # ── Clean up local file whether the run succeeded or not ──────────
            if os.path.exists(destination_file):
                subprocess.run(["rm", "-rf", destination_file])

print(f"\n✓ Score matrices saved to: {OUTPUT_H5}")