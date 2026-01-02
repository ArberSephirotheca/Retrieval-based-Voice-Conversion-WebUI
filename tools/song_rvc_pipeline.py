#!/usr/bin/env python3
"""Batch UVR5 separation + RVC conversion for all songs in a folder.

Skips work if outputs already exist and records progress in a JSON state file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from configs.config import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho
from infer.modules.vc.modules import VC

try:
    import ffmpeg
except Exception:
    ffmpeg = None

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".webm"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "songs": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "songs": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def iter_songs(songs_dir: Path, output_dir: Path) -> list[Path]:
    songs = []
    for path in sorted(songs_dir.rglob("*")):
        if not path.is_file():
            continue
        if output_dir in path.parents:
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        songs.append(path)
    return songs


def ensure_ffmpeg() -> None:
    if ffmpeg is None:
        raise RuntimeError("ffmpeg-python not available; install ffmpeg-python.")


def reformat_if_needed(src: Path, tmp_dir: Path) -> Path:
    ensure_ffmpeg()
    need_reformat = True
    try:
        info = ffmpeg.probe(str(src), cmd="ffprobe")
        if (
            info["streams"][0].get("channels") == 2
            and info["streams"][0].get("sample_rate") == "44100"
        ):
            need_reformat = False
    except Exception:
        need_reformat = True

    if not need_reformat:
        return src

    tmp_path = tmp_dir / f"{src.name}.reformatted.wav"
    cmd = [
        "ffmpeg",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(tmp_path),
        "-y",
    ]
    subprocess.run(cmd, check=True)
    return tmp_path


def build_uvr_model(model_name: str, device: str, is_half: bool, agg: int) -> object:
    if model_name == "onnx_dereverb_By_FoxJoy":
        return MDXNetDereverb(15, device)

    model_path = Path(os.getenv("weight_uvr5_root", "assets/uvr5_weights"))
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model_file = model_path / f"{model_name}.pth"
    if not model_file.exists():
        raise FileNotFoundError(f"UVR5 model not found: {model_file}")

    cls = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
    return cls(agg=agg, model_path=str(model_file), device=device, is_half=is_half)


def expected_uvr_names(song_name: str, agg: int, fmt: str, model_name: str) -> tuple[str, str]:
    if model_name == "onnx_dereverb_By_FoxJoy":
        return f"{song_name}_main_vocal.{fmt}", f"{song_name}_others.{fmt}"

    is_hp3 = "HP3" in model_name
    suffix = f"{song_name}_{agg}.{fmt}"
    vocal_name = f"instrument_{suffix}" if is_hp3 else f"vocal_{suffix}"
    inst_name = f"vocal_{suffix}" if is_hp3 else f"instrument_{suffix}"
    return vocal_name, inst_name


def resolve_uvr_outputs(
    song_name: str,
    vocals_dir: Path,
    inst_dir: Path,
    agg: int,
    fmt: str,
    model_name: str,
) -> tuple[Path | None, Path | None]:
    vocal_name, inst_name = expected_uvr_names(song_name, agg, fmt, model_name)
    vocal_path = vocals_dir / vocal_name
    inst_path = inst_dir / inst_name
    if vocal_path.exists() and inst_path.exists():
        return vocal_path, inst_path

    # Try swapped (some models invert vocal/inst outputs).
    swapped_vocal = inst_dir / vocal_name
    swapped_inst = vocals_dir / inst_name
    if swapped_vocal.exists() and swapped_inst.exists():
        return swapped_vocal, swapped_inst

    # Fallback: look for any matching files containing the song name.
    vocal_candidates = sorted(vocals_dir.glob(f"*{song_name}*.{fmt}"))
    inst_candidates = sorted(inst_dir.glob(f"*{song_name}*.{fmt}"))
    if vocal_candidates and inst_candidates:
        return vocal_candidates[0], inst_candidates[0]

    return None, None


def rvc_convert(
    vc: VC,
    vocal_path: Path,
    out_path: Path,
    f0up_key: int,
    f0_method: str,
    index_path: str,
    index_rate: float,
    filter_radius: int,
    resample_sr: int,
    rms_mix_rate: float,
    protect: float,
) -> None:
    from scipy.io import wavfile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, wav_opt = vc.vc_single(
        0,
        str(vocal_path),
        f0up_key,
        None,
        f0_method,
        index_path,
        None,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    )
    wavfile.write(str(out_path), wav_opt[0], wav_opt[1])


def mix_audio(vocal_path: Path, inst_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(vocal_path),
        "-i",
        str(inst_path),
        "-filter_complex",
        "amix=inputs=2:normalize=0",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch UVR5 + RVC pipeline for songs.")
    parser.add_argument(
        "--songs-dir",
        default="/home/zheyuanchen/wolf-knight-bot/songs",
        help="Folder containing full-mix songs",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output base directory (defaults to <songs-dir>/uvr)",
    )
    parser.add_argument("--exp", default="wolf_knight_vc", help="Training exp name")
    parser.add_argument("--model-name", default="", help="Model filename in assets/weights")
    parser.add_argument("--index-path", default="", help="Index path (added_*.index)")
    parser.add_argument("--index-rate", type=float, default=0.66)
    parser.add_argument("--f0method", default="rmvpe")
    parser.add_argument("--f0up-key", type=int, default=0)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=1.0)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--is-half", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--uvr-model", default="HP5_only_main_vocal")
    parser.add_argument("--uvr-agg", type=int, default=10)
    parser.add_argument("--uvr-format", default="wav", choices=["wav", "flac", "mp3"])
    parser.add_argument("--state", default="", help="State file path")
    parser.add_argument("--force", action="store_true", help="Redo all steps")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    songs_dir = Path(args.songs_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else songs_dir / "uvr"
    vocals_dir = output_dir / "vocals"
    inst_dir = output_dir / "instrumental"
    rvc_dir = output_dir / "vocals_rvc"
    final_dir = output_dir / "final"
    state_path = Path(args.state).expanduser().resolve() if args.state else output_dir / "state.json"

    songs = iter_songs(songs_dir, output_dir)
    if not songs:
        print(f"No songs found in {songs_dir}")
        return

    state = load_state(state_path)

    weight_root = Path(os.getenv("weight_root", "assets/weights"))
    if not weight_root.is_absolute():
        weight_root = ROOT / weight_root

    model_name = args.model_name or f"{args.exp}.pth"
    if ("/" in model_name or "\\" in model_name):
        model_name = Path(model_name).name
    model_path = weight_root / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    index_path = args.index_path
    if args.index_rate <= 0:
        index_path = ""
    elif not index_path:
        candidates = sorted((ROOT / "logs" / args.exp).glob("added_*.index"))
        if not candidates:
            raise FileNotFoundError(
                f"No index found in logs/{args.exp}. Provide --index-path or set --index-rate 0."
            )
        index_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
    elif not Path(index_path).is_absolute():
        index_path = str((ROOT / index_path).resolve())

    # Prepare models
    config = Config()
    config.device = args.device
    if args.fp32:
        config.is_half = False
    elif args.is_half:
        config.is_half = True

    uvr_model = build_uvr_model(args.uvr_model, config.device, config.is_half, args.uvr_agg)
    vc = VC(config)
    vc.get_vc(model_name)

    with tempfile.TemporaryDirectory(prefix="uvr_tmp_") as tmp:
        tmp_dir = Path(tmp)
        for song in songs:
            key = str(song.relative_to(songs_dir))
            entry = state["songs"].setdefault(key, {"source": str(song)})

            vocal_path, inst_path = resolve_uvr_outputs(
                song.name, vocals_dir, inst_dir, args.uvr_agg, args.uvr_format, args.uvr_model
            )
            sep_done = vocal_path is not None and inst_path is not None

            if args.force or not sep_done:
                if args.dry_run:
                    print(f"[DRY] UVR5 split: {song}")
                else:
                    try:
                        inp_path = reformat_if_needed(song, tmp_dir)
                        if args.uvr_model == "onnx_dereverb_By_FoxJoy":
                            uvr_model._path_audio_(
                                str(inp_path), str(vocals_dir), str(inst_dir), args.uvr_format
                            )
                        else:
                            # Keep positional args to match upstream behavior.
                            uvr_model._path_audio_(
                                str(inp_path), str(inst_dir), str(vocals_dir), args.uvr_format, "HP3" in args.uvr_model
                            )
                    except Exception as exc:
                        entry.update({"status": "error", "error": str(exc), "updated_at": now_iso()})
                        save_state(state_path, state)
                        print(f"UVR5 failed for {song}: {exc}")
                        continue

            vocal_path, inst_path = resolve_uvr_outputs(
                song.name, vocals_dir, inst_dir, args.uvr_agg, args.uvr_format, args.uvr_model
            )
            if vocal_path is None or inst_path is None:
                entry.update({"status": "error", "error": "UVR5 outputs missing", "updated_at": now_iso()})
                save_state(state_path, state)
                print(f"UVR5 outputs missing for {song}")
                continue

            rvc_path = rvc_dir / f"rvc_{vocal_path.name}"
            if args.force or not rvc_path.exists():
                if args.dry_run:
                    print(f"[DRY] RVC convert: {vocal_path}")
                else:
                    try:
                        rvc_convert(
                            vc,
                            vocal_path,
                            rvc_path,
                            args.f0up_key,
                            args.f0method,
                            index_path,
                            args.index_rate,
                            args.filter_radius,
                            args.resample_sr,
                            args.rms_mix_rate,
                            args.protect,
                        )
                    except Exception as exc:
                        entry.update({"status": "error", "error": str(exc), "updated_at": now_iso()})
                        save_state(state_path, state)
                        print(f"RVC failed for {song}: {exc}")
                        continue

            final_path = final_dir / f"{song.stem}_rvc.wav"
            if args.force or not final_path.exists():
                if args.dry_run:
                    print(f"[DRY] Mix: {rvc_path} + {inst_path}")
                else:
                    try:
                        mix_audio(rvc_path, inst_path, final_path)
                    except Exception as exc:
                        entry.update({"status": "error", "error": str(exc), "updated_at": now_iso()})
                        save_state(state_path, state)
                        print(f"Mix failed for {song}: {exc}")
                        continue

            entry.update(
                {
                    "vocal": str(vocal_path),
                    "instrumental": str(inst_path),
                    "rvc": str(rvc_path),
                    "final": str(final_path),
                    "status": "done",
                    "updated_at": now_iso(),
                }
            )
            save_state(state_path, state)
            print(f"Done: {song} -> {final_path}")


if __name__ == "__main__":
    main()
