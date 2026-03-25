import os
import glob
import argparse
from typing import Dict, Tuple, Optional


def read_pose_map(p: str) -> Tuple[Dict[int, str], Optional[int], Optional[int]]:
    """
    Read poses like:
      idx tx ty tz qx qy qz qw
    Return: {idx: "tx ty tz qx qy qz qw"}, min_idx, max_idx
    """
    pose_map: Dict[int, str] = {}
    min_idx, max_idx = None, None

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 8:
                # skip malformed line
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            pose_str = " ".join(parts[1:8])  # keep 7 numbers as string
            pose_map[idx] = pose_str
            min_idx = idx if min_idx is None else min(min_idx, idx)
            max_idx = idx if max_idx is None else max(max_idx, idx)

    return pose_map, min_idx, max_idx


def read_eval_bounds(p: str) -> Tuple[Optional[int], Optional[int]]:
    """Read both min and max idx from eval/object_X.txt (first column)."""
    if not os.path.isfile(p):
        return None, None
    
    indices = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                idx = int(parts[0])
                indices.append(idx)
            except ValueError:
                continue
                
    if not indices:
        return None, None
        
    return min(indices), max(indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/dataset_done_multi",
        help="dataset_done root",
    )
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    # pattern = os.path.join(root, "*", "eval_foundationpose", "object_*.txt")
    pattern = os.path.join(root, "*", "eval_tsdfpp", "object_*.txt")
    src_files = sorted(glob.glob(pattern))

    if not src_files:
        print(f"[WARN] No files matched: {pattern}")
        return

    for src in src_files:
        seq_dir = os.path.dirname(os.path.dirname(src))  # .../XXX
        base = os.path.basename(src)                     # object_X.txt

        eval_file = os.path.join(seq_dir, "eval", base)
        # dst_dir = os.path.join(seq_dir, "eval_foundationpose_comp")
        dst_dir = os.path.join(seq_dir, "eval_tsdfpp_comp")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, base)

        pose_map, min_src, max_src = read_pose_map(src)
        if not pose_map or min_src is None:
            print(f"[SKIP] empty or invalid: {src}")
            continue

        # 获取 Ground Truth (或 reference) 的帧范围
        min_eval, max_eval = read_eval_bounds(eval_file)
        
        # 确定最终输出的起始和结束帧
        start_target = min_eval if min_eval is not None else min_src
        end_target = max_eval if max_eval is not None else max_src

        # 提取第一个可用的位姿，用于“向前 pad”
        first_pose = pose_map[min_src]
        last_pose = None
        wrote = 0
        
        with open(dst, "w", encoding="utf-8") as f:
            for idx in range(start_target, end_target + 1):
                if idx in pose_map:
                    # 如果当前帧有数据，更新 last_pose 为当前位姿
                    last_pose = pose_map[idx]
                    current_pose = last_pose
                else:
                    # 如果没有数据：
                    # 1. 如果 last_pose 还是 None，说明还没遇到第一个有效帧，向前 pad (用 first_pose)
                    # 2. 如果 last_pose 不是 None，说明是中间或末尾断档，向后 pad (用 last_pose)
                    current_pose = last_pose if last_pose is not None else first_pose
                    
                f.write(f"{idx} {current_pose}\n")
                wrote += 1

        print(f"[OK] {src}  ->  {dst}   (range={start_target} to {end_target}, lines={wrote})")


if __name__ == "__main__":
    main()



# import os

# # ================= 配置参数 =================
# # 请在此处输入文件的绝对路径（Windows 路径建议在引号前加 r）
# FILE_PATH = r'/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/dataset_done/tomato_3/eval_midfusion/object_1.txt' 

# # 想要达到的最终总行数
# TARGET_ROWS = 335 
# # ===========================================

# def extend_file_by_path(path, target):
#     if not os.path.isabs(path):
#         print(f"提示: 您提供的路径 '{path}' 可能不是绝对路径，请检查。")
    
#     if not os.path.exists(path):
#         print(f"错误: 找不到文件 {path}")
#         return

#     # 1. 读取所有行
#     with open(path, 'r', encoding='utf-8') as f:
#         lines = [line.strip() for line in f if line.strip()]

#     current_count = len(lines)
#     if current_count >= target:
#         print(f"无需操作：当前已有 {current_count} 行，目标为 {target} 行。")
#         return

#     # 2. 解析最后一行
#     last_line = lines[-1]
#     parts = last_line.split()
    
#     try:
#         # 假设第一列是整数序号
#         last_index = int(parts[0])
#         # 提取后面所有的列并合并为字符串
#         data_content = " ".join(parts[1:])
#     except (ValueError, IndexError):
#         print("错误：无法解析最后一行的序号，请检查文件格式。")
#         return

#     # 3. 生成新行并追加
#     print(f"正在从第 {current_count} 行填充至 {target} 行...")
    
#     with open(path, 'a', encoding='utf-8') as f:
#         # 如果原文件末尾没有换行符，先补一个
#         f.write('\n') 
#         for i in range(1, target - current_count + 1):
#             new_index = last_index + i
#             f.write(f"{new_index} {data_content}\n")

#     print(f"恭喜！文件已成功更新至 {target} 行。")

# if __name__ == "__main__":
#     extend_file_by_path(FILE_PATH, TARGET_ROWS)

















# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import glob
# import argparse
# from typing import Dict, Tuple, Optional


# def read_pose_map(p: str) -> Tuple[Dict[int, str], Optional[int], Optional[int]]:
#     """
#     Read poses like:
#       idx tx ty tz qx qy qz qw
#     Return: {idx: "tx ty tz qx qy qz qw"}, min_idx, max_idx
#     """
#     pose_map: Dict[int, str] = {}
#     min_idx, max_idx = None, None

#     with open(p, "r", encoding="utf-8") as f:
#         for line in f:
#             s = line.strip()
#             if not s:
#                 continue
#             parts = s.split()
#             if len(parts) < 8:
#                 # skip malformed line
#                 continue
#             try:
#                 idx = int(parts[0])
#             except ValueError:
#                 continue
#             pose_str = " ".join(parts[1:8])  # keep 7 numbers as string
#             pose_map[idx] = pose_str
#             min_idx = idx if min_idx is None else min(min_idx, idx)
#             max_idx = idx if max_idx is None else max(max_idx, idx)

#     return pose_map, min_idx, max_idx


# def read_max_idx(p: str) -> Optional[int]:
#     """Read max idx from eval/object_X.txt (take max of first column)."""
#     if not os.path.isfile(p):
#         return None
#     max_idx = None
#     with open(p, "r", encoding="utf-8") as f:
#         for line in f:
#             s = line.strip()
#             if not s:
#                 continue
#             parts = s.split()
#             try:
#                 idx = int(parts[0])
#             except ValueError:
#                 continue
#             max_idx = idx if max_idx is None else max(max_idx, idx)
#     return max_idx


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--root",
#         default="/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/dataset_done_multi",
#         help="dataset_done root",
#     )
#     args = ap.parse_args()
#     root = os.path.abspath(args.root)

#     # pattern = os.path.join(root, "*", "eval_foundationpose", "object_*.txt")
#     pattern = os.path.join(root, "*", "eval_tsdfpp", "object_*.txt")
#     src_files = sorted(glob.glob(pattern))

#     if not src_files:
#         print(f"[WARN] No files matched: {pattern}")
#         return

#     for src in src_files:
#         seq_dir = os.path.dirname(os.path.dirname(src))  # .../XXX
#         base = os.path.basename(src)                     # object_X.txt

#         eval_file = os.path.join(seq_dir, "eval", base)
#         # dst_dir = os.path.join(seq_dir, "eval_foundationpose_comp")
#         dst_dir = os.path.join(seq_dir, "eval_tsdfpp_comp")
#         os.makedirs(dst_dir, exist_ok=True)
#         dst = os.path.join(dst_dir, base)

#         pose_map, min_src, max_src = read_pose_map(src)
#         if not pose_map or min_src is None:
#             print(f"[SKIP] empty or invalid: {src}")
#             continue

#         max_eval = read_max_idx(eval_file)
#         max_target = max_eval if max_eval is not None else max_src

#         # Start from the first available idx in eval_bundlesdf (向后 pad)
#         start_idx = min_src

#         last_pose = None
#         wrote = 0
#         with open(dst, "w", encoding="utf-8") as f:
#             for idx in range(start_idx, max_target + 1):
#                 if idx in pose_map:
#                     last_pose = pose_map[idx]
#                 else:
#                     if last_pose is None:
#                         # theoretically shouldn't happen since start_idx = min_src exists
#                         continue
#                 f.write(f"{idx} {last_pose}\n")
#                 wrote += 1

#         print(f"[OK] {src}  ->  {dst}   (start={start_idx}, max={max_target}, lines={wrote})")


# if __name__ == "__main__":
#     main()
