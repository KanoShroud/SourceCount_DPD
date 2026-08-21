"""
trim_exp_pt.py — 去掉实验 pt 文件中不需要的标签，减小文件体积
"""
import argparse
import gc
import glob
import os
import shutil

import torch

from chapter_runtime import (
    data_dir as runtime_data_dir,
    output_dir as runtime_output_dir,
)

REMOVE_KEYS = ['hyp_mask', 'hyp_mask_single', 'gauss_label', 'gauss_multi']

def trim_file(pt_path, output_path):
    d = torch.load(pt_path, weights_only=False)
    removed = []
    for key in REMOVE_KEYS:
        if key in d:
            del d[key]
            removed.append(key)
    if not removed:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(pt_path, output_path)
        print(f"  {os.path.basename(pt_path)}: 已是精简版，复制到隔离目录")
        del d
        return os.path.getsize(output_path) / 1e6

    old_size = os.path.getsize(pt_path) / 1e6

    # 始终写入隔离目录，不替换输入文件。
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + '.tmp'
    torch.save(d, tmp_path)
    del d
    gc.collect()

    # 验证临时文件可读
    try:
        check = torch.load(tmp_path, weights_only=False)
        assert 'fine_dpd' in check and 'pos_label' in check
        del check
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(pt_path)}: 验证失败 {e}，保留原文件")
        os.remove(tmp_path)
        return 0.0

    os.replace(tmp_path, output_path)
    new_size = os.path.getsize(output_path) / 1e6
    print(
        f"  {os.path.basename(pt_path)}: {old_size:.0f}MB → {new_size:.0f}MB  "
        f"(去掉 {', '.join(removed)}) → {output_path}"
    )
    return new_size

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data_dir', type=str, default=str(runtime_data_dir()))
    pa.add_argument('--output_dir', type=str, default=None,
                    help='精简文件输出目录；默认按 smoke/formal 模式隔离，绝不覆盖输入')
    args = pa.parse_args()

    exp_dir = os.path.join(args.data_dir, 'exp')
    output_dir = args.output_dir or str(runtime_output_dir('trim_exp_pt'))
    input_root = os.path.abspath(exp_dir)
    output_root = os.path.abspath(output_dir)
    try:
        output_inside_input = os.path.commonpath([input_root, output_root]) == input_root
    except ValueError:
        output_inside_input = False
    if output_inside_input:
        raise ValueError('output_dir 不得位于输入 exp 目录内部，以免覆盖或重复处理原始文件')
    pt_files = sorted(glob.glob(os.path.join(exp_dir, '**', 'exp_*.pt'), recursive=True))
    pt_files = [f for f in pt_files if 'index' not in f]

    print(f"找到 {len(pt_files)} 个实验数据文件")
    total_before, total_after = 0, 0
    for f in pt_files:
        before = os.path.getsize(f) / 1e6
        total_before += before
        relative_path = os.path.relpath(f, exp_dir)
        output_path = os.path.join(output_dir, relative_path)
        total_after += trim_file(f, output_path)

    print(f"\n总计: {total_before/1e3:.1f}GB → {total_after/1e3:.1f}GB")
    print("Done!")

if __name__ == '__main__':
    main()
