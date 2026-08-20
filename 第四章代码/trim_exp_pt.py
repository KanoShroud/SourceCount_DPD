"""
trim_exp_pt.py — 去掉实验 pt 文件中不需要的标签，减小文件体积
"""
import torch, os, argparse, glob, gc

REMOVE_KEYS = ['hyp_mask', 'hyp_mask_single', 'gauss_label', 'gauss_multi']

def trim_file(pt_path):
    d = torch.load(pt_path, weights_only=False)
    removed = []
    for key in REMOVE_KEYS:
        if key in d:
            del d[key]
            removed.append(key)
    if not removed:
        print(f"  {os.path.basename(pt_path)}: 已是精简版，跳过")
        del d
        return

    old_size = os.path.getsize(pt_path) / 1e6

    # 先保存到临时文件，成功后再替换原文件
    tmp_path = pt_path + '.tmp'
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
        return

    # 替换原文件
    os.replace(tmp_path, pt_path)
    new_size = os.path.getsize(pt_path) / 1e6
    print(f"  {os.path.basename(pt_path)}: {old_size:.0f}MB → {new_size:.0f}MB  (去掉 {', '.join(removed)})")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data_dir', type=str, default='/mnt/data/ltzdata_loc')
    args = pa.parse_args()

    exp_dir = os.path.join(args.data_dir, 'exp')
    pt_files = sorted(glob.glob(os.path.join(exp_dir, '**', 'exp_*.pt'), recursive=True))
    pt_files = [f for f in pt_files if 'index' not in f]

    print(f"找到 {len(pt_files)} 个实验数据文件")
    total_before, total_after = 0, 0
    for f in pt_files:
        before = os.path.getsize(f) / 1e6
        total_before += before
        trim_file(f)
        total_after += os.path.getsize(f) / 1e6

    print(f"\n总计: {total_before/1e3:.1f}GB → {total_after/1e3:.1f}GB")
    print("Done!")

if __name__ == '__main__':
    main()