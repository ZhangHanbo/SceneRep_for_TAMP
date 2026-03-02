#!/usr/bin/env python3

import os
import json
from glob import glob
from pathlib import Path
from eval_all import PoseEvaluator
import csv
import datetime

DATASETS_ROOT = "/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/dataset_done_multi"
META_FILE = "eval_objects_debug.json"     # 统一meta文件
OUT_SUMMARY_FILE = "batch_eval_summary_multi.json"


# -----------------------------------------------------------
# 1. object meta 统一加载
# -----------------------------------------------------------

def load_global_object_meta(meta_path):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"统一meta文件不存在: {meta_path}")

    with open(meta_path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------
# 2. 从 eval 文件发现当前 dataset 的 object_id
# -----------------------------------------------------------

def auto_find_object_ids(dataset_dir):
    ids = []
    for f in glob(os.path.join(dataset_dir, "eval", "object_*.txt")):
        name = os.path.basename(f)
        num = name.replace("object_", "").replace(".txt", "")
        if num.isdigit():
            ids.append(int(num))
    return sorted(ids)

def append_global_csv(csv_path, row_dict, fieldnames):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)



# -----------------------------------------------------------
# 3. 主驱动：完全由 JSON 控制
# -----------------------------------------------------------

def main():

    print("\n======= 批量评测启动（JSON 严格控制模式） =======")

    meta = load_global_object_meta(META_FILE)

    if not meta:
        print("[ERROR] eval_objects.json 内容为空，无法执行评测")
        return

    all_results = []

    for dataset_name, obj_dict in meta.items():

        dataset_dir = os.path.join(DATASETS_ROOT, dataset_name)

        print("\n" + "=" * 68)
        print(f"Dataset: {dataset_name}")

        # ---------- dataset 不存在 ----------
        if not os.path.isdir(dataset_dir):
            print(f"[WARN] dataset 不存在, skip: {dataset_dir}")
            continue

        # ---------- 必要子目录检查 ----------
        if not (Path(dataset_dir) / "eval").is_dir() or \
           not (Path(dataset_dir) / "pose_txt").is_dir():
            print("[WARN] dataset 缺少必要目录(eval/pose_txt)，skip")
            continue

        evaluator = PoseEvaluator(dataset_dir)

        # 扫描磁盘实际存在的 object_id
        disk_object_ids = auto_find_object_ids(dataset_dir)
        print(f"磁盘中发现 object_id: {disk_object_ids}")

        for sid, obj_name in obj_dict.items():

            oid = int(sid)

            print(f"\n--- Evaluating [{dataset_name}] Object id={oid}, name={obj_name}")

            # ---------- eval 文件检查 ----------
            eval_file = os.path.join(dataset_dir, "eval", f"object_{oid}.txt")

            if not os.path.exists(eval_file):
                print(f"[WARN] 找不到 {eval_file}, skip")
                continue

            try:
                evaluator.evaluate(
                    object_id=oid,
                    object_name=obj_name
                )

                res_file = os.path.join(
                    dataset_dir,
                    "eval",
                    "evaluation_results.json"
                )

                if os.path.exists(res_file):
                    with open(res_file) as f:
                        res = json.load(f)
                else:
                    res = None

                all_results.append({
                    "dataset": dataset_name,
                    "object_id": oid,
                    "object_name": obj_name,
                    "results": res
                })

            except Exception as e:
                print(f"[ERROR] 失败: {e}")
                all_results.append({
                    "dataset": dataset_name,
                    "object_id": oid,
                    "object_name": obj_name,
                    "error": str(e)
                })


    # -----------------------------------------------------------
    # 4. 保存全局汇总
    # -----------------------------------------------------------

    out_path = os.path.join(os.path.dirname(__file__), OUT_SUMMARY_FILE)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("--------------------------------")
    print(f"📄 汇总文件写入: {out_path}")


if __name__ == "__main__":
    main()
