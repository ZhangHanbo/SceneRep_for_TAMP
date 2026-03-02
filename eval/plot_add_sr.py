#!/usr/bin/env python3
"""plot_add_sr.py

Per-object *per-case* visualization of ADD / ADD-S success-rate (SR) curves.

Your requirement:
- CSV 中每三行代表同一个 case 的三种方法结果（3 methods）。
- 不要把同一个物体的所有 case 平均掉。
- 对每个物体的每个 case 画一张图：同一张图里叠加三种方法的曲线。

默认把 `dataset` 列作为 case 标识（你也可以用 --case_col 改成别的列）。

Outputs:
- out_dir/plots_add(or plots_adds)/<object_name>/<case_id>.png
- out_dir/summary_per_case_add(or adds).csv

Run:
  python plot_add_sr.py --csv global_results.csv --out_dir out_add_sr --metric add
  python plot_add_sr.py --csv global_results.csv --out_dir out_add_sr --metric adds

Optional:
  --case_col dataset
  --show
"""

import argparse
import hashlib
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_sr_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Return columns like add_sr_1mm ... sorted by threshold."""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)mm$")
    cols = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda x: x[0])
    return [c for _, c in cols]


def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-\.\s]+", "_", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s if s else "unnamed"


def short_case_id(case_value: str, keep_basename: bool = True) -> str:
    """Make a stable, short id for filenames (basename + md5 prefix)."""
    s = str(case_value)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    if keep_basename:
        try:
            base = Path(s).name
            base = base if base else "case"
        except Exception:
            base = "case"
        base = safe_filename(base)[:50]
        return f"{base}__{h}"
    return f"case__{h}"


def auc_from_curve(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def plot_case(
    obj: str,
    case_value: str,
    g_case: pd.DataFrame,
    thresholds_mm: np.ndarray,
    sr_cols: List[str],
    out_path: Path,
    title_prefix: str,
) -> None:
    """Plot one figure for one (object, case), overlaying methods."""
    plt.figure()

    for method, gm in g_case.groupby("method", dropna=False):
        # if duplicates per method within same case, average them
        y = gm[sr_cols].mean(axis=0, skipna=True).to_numpy(dtype=float)
        plt.plot(thresholds_mm, y, label=str(method))

    plt.xlabel("Threshold (mm)")
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)

    case_id = short_case_id(case_value, keep_basename=True)
    plt.title(f"{title_prefix} vs threshold — {obj}\ncase: {case_id}")
    plt.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to global_results.csv")
    ap.add_argument("--out_dir", type=str, default="out_add_sr", help="Output directory")
    ap.add_argument("--metric", type=str, default="add", choices=["add", "adds"],
                    help="Plot ADD SR or ADD-S SR")
    ap.add_argument("--case_col", type=str, default="dataset",
                    help="Column that identifies a case (default: dataset)")
    ap.add_argument("--show", action="store_true", help="Show a few plots interactively (also saves)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"object_name", "method", args.case_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    prefix = "add_sr_" if args.metric == "add" else "adds_sr_"
    sr_cols = find_sr_columns(df, prefix=prefix)
    if not sr_cols:
        raise ValueError(f"No columns found like {prefix}{{N}}mm in CSV.")

    thresholds_mm = np.array([int(re.findall(r"(\d+)mm$", c)[0]) for c in sr_cols], dtype=float)

    # numeric SR
    for c in sr_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    fig_dir = out_dir / f"plots_{args.metric}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # summary per (object, case, method)
    sr_at = [10, 20, 30, 50]
    sr_idx = {t: int(np.where(thresholds_mm == t)[0][0]) for t in sr_at if t in set(thresholds_mm.astype(int))}
    summary_rows = []

    # loop per (object, case)
    for (obj, case_value), g_case in df.groupby(["object_name", args.case_col], dropna=False):
        obj_str = str(obj)
        case_str = str(case_value)

        obj_dir = fig_dir / safe_filename(obj_str)
        case_id = short_case_id(case_str, keep_basename=True)
        out_path = obj_dir / f"{case_id}.png"

        plot_case(
            obj=obj_str,
            case_value=case_str,
            g_case=g_case,
            thresholds_mm=thresholds_mm,
            sr_cols=sr_cols,
            out_path=out_path,
            title_prefix="ADD SR" if args.metric == "add" else "ADD-S SR",
        )

        # per-method stats for this case
        for method, gm in g_case.groupby("method", dropna=False):
            y = gm[sr_cols].mean(axis=0, skipna=True).to_numpy(dtype=float)
            summary_rows.append({
                "object_name": obj_str,
                args.case_col: case_str,
                "case_id": case_id,
                "method": str(method),
                "auc": auc_from_curve(thresholds_mm, y),
                "sr_10mm": y[sr_idx[10]] if 10 in sr_idx else np.nan,
                "sr_20mm": y[sr_idx[20]] if 20 in sr_idx else np.nan,
                "sr_30mm": y[sr_idx[30]] if 30 in sr_idx else np.nan,
                "sr_50mm": y[sr_idx[50]] if 50 in sr_idx else np.nan,
                "total_frames": float(pd.to_numeric(gm.get("total_frames", np.nan), errors="coerce").mean()),
                "num_segments": float(pd.to_numeric(gm.get("num_segments", np.nan), errors="coerce").mean()),
                "time": str(gm.get("time", pd.Series([np.nan])).iloc[0]) if "time" in gm.columns else "",
            })

    summary = pd.DataFrame(summary_rows).sort_values(["object_name", args.case_col, "method"])
    summary_path = out_dir / f"summary_per_case_{args.metric}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[OK] Saved per-case plots to: {fig_dir}")
    print(f"[OK] Saved per-case summary to: {summary_path}")

    if args.show:
        shown = 0
        for p in sorted(fig_dir.rglob("*.png")):
            img = plt.imread(p)
            plt.figure()
            plt.imshow(img)
            plt.axis("off")
            plt.title(p.stem)
            shown += 1
            if shown >= 12:
                break
        plt.show()


if __name__ == "__main__":
    main()
