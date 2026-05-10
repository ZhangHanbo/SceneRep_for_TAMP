# Workflow: evaluation

Turn per-frame pose dumps into ADD / ADD-S tables, success-rate plots, and
side-by-side comparisons with baselines.

## Inputs

A dataset where `<dataset>/eval/` contains both:

* Ground-truth pose `*.txt` files (one per object).
* Predicted pose files written by a tracker run.

The default file format is whitespace-separated `(timestamp, qw, qx, qy,
qz, tx, ty, tz)` per line.  See the loader in `eval/eval_all.py`.

## Single-dataset evaluation

```bash
python eval/eval_all.py \
    --dataset datasets/<traj> \
    --eval-objects eval/eval_objects.json
```

Outputs:

* `eval_<traj>.csv` — per-frame ADD / ADD-S, success rate, per-object stats.
* Stdout summary (mean error, success rate at the 2 cm / 5 cm thresholds).

`eval_objects.json` is the per-object metadata file; the included
`eval/eval_objects.json` covers the canonical apple / tray / bowl set.
Use `eval/eval_objects_multi.json` for multi-object scenes.

## Batch over many datasets

```bash
python eval/eval_batch.py \
    --root datasets/ \
    --eval-objects eval/eval_objects.json
```

Iterates every dataset under `--root` and concatenates results into
`global_results_single.csv`.  Use the `_multi` variant for multi-object
scenes.

## Compare against external baselines

```bash
python eval/eval_ours.py \
    --dataset datasets/apple_1 \
    --transform eval/transform_matrix.txt
```

`eval_ours.py` is hardcoded for the `apple_1` trajectory and produces the
direct comparison plot against FoundationPose / BundleSDF / MidFusion used
in the paper.  `transform_matrix.txt` (and the `_apple1` variant) align the
external baselines' world frames into ours.

## Render predicted poses on RGB

```bash
python eval/render_poses.py \
    --dataset datasets/<traj> \
    --object-id 0 --object-name apple
```

Produces a directory of overlay PNGs — green for ground truth, red for
prediction, with per-frame error annotations.  Useful for finding the exact
frames where the tracker fails.

## Build report tables

```bash
# One dataset, simple table
python eval/generate_single_table.py --csv eval_<traj>.csv

# All datasets, combined
python eval/generate_combined_table.py --csv global_results_single.csv

# All datasets with extra columns (per-axis errors etc.)
python eval/generate_combined_table_expanded.py --csv global_results_single.csv

# Heat-map cell colors (ADD-S vs threshold)
python eval/generate_colored_table.py --csv global_results_single.csv
```

Output is markdown / LaTeX (`output_table.tex`) ready to drop into a paper.

## Plots

```bash
python eval/plot_add_sr.py --csv global_results_single.csv
```

Generates the ADD-success-rate-vs-threshold curves used in the paper.

## Modifying success thresholds

`eval/success_threshold.txt` is a one-line file holding the success
threshold (metres) used by every report generator.  Edit it once and
re-run any of the table scripts.

## What is "success"?

A frame is a success if `ADD-S < threshold`.  ADD-S is the symmetric
version of ADD (closest-point distance averaged over the model surface),
necessary for axis-symmetric objects (cups, bowls).  The exact formula is
the standard one — see `eval/eval_all.py::compute_metrics`.

## Auxiliary scripts

| Script | Purpose |
|---|---|
| `pad_data.py` | Backfills missing per-object columns when a tracker drops a track mid-trajectory. |
| `patch.py` | One-off metric corrections for legacy CSVs. |
| `update_success_rate.py` | Recomputes the success-rate column after editing the threshold. |
| `table_transposed.py` | Swaps rows and columns of a generated table. |

```{seealso}
* [Offline pipeline](offline_pipeline.md) — produce the predicted-pose dumps
  these tools consume.
```
