#!/usr/bin/env python3
"""Run the RVC training pipeline (v2) end-to-end from the CLI."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_config(root: Path, version: str, sr: str, exp_dir: Path) -> None:
    if version == "v1" or sr == "40k":
        cfg_rel = Path("configs/inuse/v1") / f"{sr}.json"
        src_rel = Path("configs/v1") / f"{sr}.json"
    else:
        cfg_rel = Path("configs/inuse/v2") / f"{sr}.json"
        src_rel = Path("configs/v2") / f"{sr}.json"

    cfg_path = root / cfg_rel
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / src_rel, cfg_path)

    cfg_out = exp_dir / "config.json"
    if not cfg_out.exists():
        cfg_out.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")


def build_filelist(
    root: Path,
    exp_dir: Path,
    version: str,
    sr: str,
    spk_id: int,
    use_f0: bool,
) -> None:
    gt_wavs = exp_dir / "0_gt_wavs"
    feat_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"

    names = {
        p.stem for p in gt_wavs.glob("*.wav")
    } & {p.stem for p in feat_dir.glob("*.npy")}

    lines: list[str] = []
    for name in sorted(names):
        if use_f0:
            lines.append(
                f"{gt_wavs}/{name}.wav|{feat_dir}/{name}.npy|"
                f"{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
            )
        else:
            lines.append(f"{gt_wavs}/{name}.wav|{feat_dir}/{name}.npy|{spk_id}")

    fea_dim = 256 if version == "v1" else 768
    mute_root = root / "logs" / "mute"
    if use_f0:
        for _ in range(2):
            lines.append(
                f"{mute_root}/0_gt_wavs/mute{sr}.wav|"
                f"{mute_root}/3_feature{fea_dim}/mute.npy|"
                f"{mute_root}/2a_f0/mute.wav.npy|"
                f"{mute_root}/2b-f0nsf/mute.wav.npy|{spk_id}"
            )
    else:
        for _ in range(2):
            lines.append(
                f"{mute_root}/0_gt_wavs/mute{sr}.wav|"
                f"{mute_root}/3_feature{fea_dim}/mute.npy|{spk_id}"
            )

    random.shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")


def train_index(exp_dir: Path, version: str) -> None:
    import numpy as np
    import faiss
    from sklearn.cluster import MiniBatchKMeans

    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    npys = [np.load(p) for p in sorted(feature_dir.glob("*.npy"))]
    if not npys:
        raise RuntimeError("No feature files found; run feature extraction first.")
    big = np.concatenate(npys, 0)
    idx = np.arange(big.shape[0])
    np.random.shuffle(idx)
    big = big[idx]

    if big.shape[0] > 2e5:
        big = MiniBatchKMeans(
            n_clusters=10000,
            batch_size=256 * 4,
            compute_labels=False,
            init="random",
        ).fit(big).cluster_centers_

    np.save(exp_dir / "total_fea.npy", big)
    n_ivf = min(int(16 * (big.shape[0] ** 0.5)), big.shape[0] // 39)
    dim = 256 if version == "v1" else 768
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big)

    trained = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir.name}_{version}.index"
    added = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir.name}_{version}.index"
    faiss.write_index(index, str(trained))

    batch = 8192
    for i in range(0, big.shape[0], batch):
        index.add(big[i : i + batch])
    faiss.write_index(index, str(added))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RVC v2 training pipeline via CLI.")
    parser.add_argument("--exp", default="wolf_knight_vc")
    parser.add_argument(
        "--dataset",
        default="/home/zheyuanchen/wolf-knight-bot/GPT-SoVITS/output/new_slicer_opt",
    )
    parser.add_argument("--sr", default="48k", choices=["32k", "40k", "48k"])
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--f0", type=int, default=1, choices=[0, 1])
    parser.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "pm", "harvest", "dio"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--n-p", type=int, default=8, help="CPU workers for preprocessing")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--total-epoch", type=int, default=120)
    parser.add_argument("--save-every-epoch", type=int, default=10)
    parser.add_argument("--save-latest", type=int, default=1, choices=[0, 1])
    parser.add_argument("--cache-gpu", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save-weights", type=int, default=1, choices=[0, 1])
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--slice-per", type=float, default=3.7)
    parser.add_argument("--is-half", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    exp_dir = root / "logs" / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"Dataset folder not found: {dataset}")

    required = [
        root / "assets" / "hubert" / "hubert_base.pt",
    ]
    if args.f0 == 1 and args.f0_method == "rmvpe":
        required.append(root / "assets" / "rmvpe" / "rmvpe.pt")

    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing required model: {path}")

    pretrain_dir = root / "assets" / ("pretrained_v2" if args.version == "v2" else "pretrained")
    pretrain_g = pretrain_dir / f"f0G{args.sr}.pth"
    pretrain_d = pretrain_dir / f"f0D{args.sr}.pth"
    if args.f0 == 1:
        if not pretrain_g.exists() or not pretrain_d.exists():
            raise SystemExit(f"Missing pretrained weights: {pretrain_g} / {pretrain_d}")

    run(
        [
            sys.executable,
            "infer/modules/train/preprocess.py",
            str(dataset),
            "40000" if args.sr == "40k" else ("32000" if args.sr == "32k" else "48000"),
            str(args.n_p),
            str(exp_dir),
            "True" if args.no_parallel else "False",
            f"{args.slice_per:.1f}",
        ],
        root,
    )

    if args.f0 == 1:
        if args.f0_method == "rmvpe":
            run(
                [
                    sys.executable,
                    "infer/modules/train/extract/extract_f0_rmvpe.py",
                    "1",
                    "0",
                    str(args.gpu),
                    str(exp_dir),
                    "True" if args.is_half else "False",
                ],
                root,
            )
        else:
            run(
                [
                    sys.executable,
                    "infer/modules/train/extract/extract_f0_print.py",
                    str(exp_dir),
                    str(args.n_p),
                    args.f0_method,
                ],
                root,
            )

    run(
        [
            sys.executable,
            "infer/modules/train/extract_feature_print.py",
            "cuda:0",
            "1",
            "0",
            str(args.gpu),
            str(exp_dir),
            args.version,
            "True" if args.is_half else "False",
        ],
        root,
    )

    ensure_config(root, args.version, args.sr, exp_dir)
    build_filelist(root, exp_dir, args.version, args.sr, 0, args.f0 == 1)

    train_cmd = [
        sys.executable,
        "infer/modules/train/train.py",
        "-e",
        args.exp,
        "-sr",
        args.sr,
        "-f0",
        str(args.f0),
        "-bs",
        str(args.batch_size),
        "-g",
        str(args.gpu),
        "-te",
        str(args.total_epoch),
        "-se",
        str(args.save_every_epoch),
        "-l",
        str(args.save_latest),
        "-c",
        str(args.cache_gpu),
        "-sw",
        str(args.save_weights),
        "-v",
        args.version,
    ]
    if args.f0 == 1:
        train_cmd += ["-pg", str(pretrain_g), "-pd", str(pretrain_d)]
    run(train_cmd, root)

    if not args.skip_index:
        train_index(exp_dir, args.version)


if __name__ == "__main__":
    main()
