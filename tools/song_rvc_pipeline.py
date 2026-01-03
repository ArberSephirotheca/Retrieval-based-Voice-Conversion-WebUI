#!/usr/bin/env python3
"""Batch separation + RVC conversion for songs in a folder or a single file.

Skips work if outputs already exist and records progress in a JSON state file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

AUDIO_EXTS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".mp4",
    ".aac",
    ".ogg",
    ".opus",
    ".weba",
    ".webm",
}


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


def convert_to_wav(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def ensure_wav_inputs(songs_dir: Path, output_dir: Path) -> list[Path]:
    songs: list[Path] = []
    seen: set[str] = set()
    for path in sorted(songs_dir.rglob("*")):
        if not path.is_file():
            continue
        if output_dir in path.parents:
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        if path.suffix.lower() == ".wav":
            resolved = str(path.resolve())
            if resolved not in seen:
                songs.append(path)
                seen.add(resolved)
            continue
        wav_path = path.with_suffix(".wav")
        if not wav_path.exists():
            print(f"Converting to wav: {path} -> {wav_path}")
            convert_to_wav(path, wav_path)
        resolved = str(wav_path.resolve())
        if resolved not in seen:
            songs.append(wav_path)
            seen.add(resolved)
    return songs


def ensure_wav_inputs_from_list(files: list[Path], output_dir: Path) -> list[Path]:
    songs: list[Path] = []
    seen: set[str] = set()
    for path in files:
        if not path.is_file():
            continue
        if output_dir in path.parents:
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(f"Unsupported audio type: {path}")
        if path.suffix.lower() == ".wav":
            resolved = str(path.resolve())
            if resolved not in seen:
                songs.append(path)
                seen.add(resolved)
            continue
        wav_path = path.with_suffix(".wav")
        if not wav_path.exists():
            print(f"Converting to wav: {path} -> {wav_path}")
            convert_to_wav(path, wav_path)
        resolved = str(wav_path.resolve())
        if resolved not in seen:
            songs.append(wav_path)
            seen.add(resolved)
    return songs


def find_msst_checkpoint(msst_root: Path, model_type: str) -> Path | None:
    candidates = list(msst_root.rglob("*.ckpt"))
    if not candidates:
        candidates = list(msst_root.rglob("*.pth")) + list(msst_root.rglob("*.pt"))
    if not candidates:
        return None
    filtered = [path for path in candidates if model_type in path.name]
    pool = filtered or candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def msst_device_args(device: str) -> list[str]:
    if device.startswith("cuda"):
        if ":" in device:
            return ["--device_ids", device.split(":", 1)[1]]
        return ["--device_ids", "0"]
    return ["--force_cpu"]


def run_msst_separation(
    songs: list[Path],
    output_dir: Path,
    model_type: str,
    config_path: Path,
    checkpoint_path: Path,
    msst_root: Path,
    python_bin: str,
    device: str,
    force: bool,
) -> None:
    vocals_dir = output_dir / "vocals"
    inst_dir = output_dir / "instrumental"
    pending: list[Path] = []
    for song in songs:
        vocal_path, inst_path = resolve_msst_outputs(song, vocals_dir, inst_dir)
        if force or vocal_path is None or inst_path is None:
            pending.append(song)
    if not pending:
        return
    with tempfile.TemporaryDirectory(prefix="msst_inputs_") as tmp:
        input_dir = Path(tmp)
        for song in pending:
            target = input_dir / song.name
            if target.exists():
                continue
            try:
                os.symlink(song, target)
            except OSError:
                shutil.copy(song, target)
        cmd = [
            python_bin,
            str(msst_root / "inference.py"),
            "--model_type",
            model_type,
            "--config_path",
            str(config_path),
            "--start_check_point",
            str(checkpoint_path),
            "--input_folder",
            str(input_dir),
            "--store_dir",
            str(output_dir),
            "--filename_template",
            "{instr}/{file_name}",
            "--extract_instrumental",
        ]
        cmd.extend(msst_device_args(device))
        print("MSST command:", " ".join(str(item) for item in cmd))
        subprocess.run(cmd, cwd=msst_root, check=True)


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


def resolve_msst_outputs(
    song_path: Path, vocals_dir: Path, inst_dir: Path
) -> tuple[Path | None, Path | None]:
    file_name = song_path.stem
    vocal_path = find_msst_output(vocals_dir, file_name)
    inst_path = find_msst_output(inst_dir, file_name)
    if vocal_path and inst_path:
        return vocal_path, inst_path
    return None, None


def find_msst_output(base_dir: Path, file_name: str) -> Path | None:
    for ext in (".wav", ".flac"):
        candidate = base_dir / f"{file_name}{ext}"
        if candidate.exists():
            return candidate
    return None


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
    info, wav_opt = vc.vc_single(
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
    if not wav_opt or wav_opt[0] is None or wav_opt[1] is None:
        raise RuntimeError(f"RVC conversion failed: {info}")
    wavfile.write(str(out_path), wav_opt[0], wav_opt[1])


def mix_audio(
    vocal_path: Path, inst_path: Path, out_path: Path, vocal_gain: float, inst_gain: float
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(vocal_path),
        "-i",
        str(inst_path),
        "-filter_complex",
        f"[0:a]volume={vocal_gain}[v];[1:a]volume={inst_gain}[i];[v][i]amix=inputs=2:normalize=0",
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
        "--song-file",
        default="",
        help="Process a single song file instead of a folder",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output base directory (defaults to <songs-dir>/uvr)",
    )
    parser.add_argument("--exp", default="wolf_knight_vc", help="Training exp name")
    parser.add_argument("--model-name", default="", help="Model filename in assets/weights")
    parser.add_argument(
        "--index-path",
        default="",
        help="Index path (added_*.index). If empty, auto-pick newest from logs/<exp>.",
    )
    parser.add_argument(
        "--sep-backend",
        default="msst",
        choices=["uvr", "msst"],
        help="Source separation backend to use.",
    )
    parser.add_argument(
        "--msst-root",
        default="/home/zheyuanchen/wolf-knight-bot/Music-Source-Separation-Training",
        help="Music-Source-Separation-Training repo root.",
    )
    parser.add_argument("--msst-model-type", default="mdx23c")
    parser.add_argument(
        "--msst-config",
        default="/home/zheyuanchen/wolf-knight-bot/Music-Source-Separation-Training/configs/config_vocals_mdx23c.yaml",
        help="Config file for MSST inference.",
    )
    parser.add_argument(
        "--msst-checkpoint",
        default="/home/zheyuanchen/wolf-knight-bot/Music-Source-Separation-Training/model_vocals_mdx23c_sdr_10.17.ckpt",
        help="Checkpoint file for MSST inference.",
    )
    parser.add_argument(
        "--msst-python",
        default="/home/zheyuanchen/wolf-knight-bot/Retrieval-based-Voice-Conversion-WebUI/.venv/bin/python",
        help="Python executable for MSST.",
    )
    parser.add_argument("--index-rate", type=float, default=0.45)
    parser.add_argument("--f0method", default="rmvpe")
    parser.add_argument("--f0up-key", type=int, default=0)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=0.7)
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--vocal-gain", type=float, default=1.2)
    parser.add_argument("--inst-gain", type=float, default=0.8)
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
    sys.argv = sys.argv[:1]

    load_dotenv(ROOT / ".env")

    song_file = Path(args.song_file).expanduser().resolve() if args.song_file else None
    if song_file:
        if not song_file.exists() or not song_file.is_file():
            raise FileNotFoundError(f"Song file not found: {song_file}")
        songs_dir = song_file.parent
    else:
        songs_dir = Path(args.songs_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else songs_dir / "uvr"
    vocals_dir = output_dir / "vocals"
    inst_dir = output_dir / "instrumental"
    rvc_dir = output_dir / "vocals_rvc"
    final_dir = output_dir / "final"
    state_path = Path(args.state).expanduser().resolve() if args.state else output_dir / "state.json"

    if song_file:
        songs = ensure_wav_inputs_from_list([song_file], output_dir)
    else:
        songs = ensure_wav_inputs(songs_dir, output_dir)
    if not songs:
        target = song_file or songs_dir
        print(f"No songs found in {target}")
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
    if index_path:
        print(f"Using index: {index_path}")
    else:
        print("Index disabled (index_rate <= 0)")

    # Prepare models
    config = Config()
    config.device = args.device
    if args.fp32:
        config.is_half = False
    elif args.is_half:
        config.is_half = True

    uvr_model = None
    if args.sep_backend == "uvr":
        uvr_model = build_uvr_model(args.uvr_model, config.device, config.is_half, args.uvr_agg)
    else:
        msst_root = Path(args.msst_root).expanduser().resolve()
        if not msst_root.exists():
            raise FileNotFoundError(f"MSST root not found: {msst_root}")
        config_path = (
            Path(args.msst_config).expanduser()
            if args.msst_config
            else msst_root / "configs" / "config_vocals_mdx23c.yaml"
        )
        if not config_path.is_absolute():
            config_path = (msst_root / config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"MSST config not found: {config_path}")
        checkpoint_path = (
            Path(args.msst_checkpoint).expanduser()
            if args.msst_checkpoint
            else find_msst_checkpoint(msst_root, args.msst_model_type)
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                "MSST checkpoint not found. Download one and pass --msst-checkpoint."
            )
        if not checkpoint_path.is_absolute():
            checkpoint_path = (msst_root / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MSST checkpoint not found: {checkpoint_path}")
        python_bin = args.msst_python or sys.executable
        run_msst_separation(
            songs,
            output_dir,
            args.msst_model_type,
            config_path,
            checkpoint_path,
            msst_root,
            python_bin,
            args.device,
            args.force,
        )
    vc = VC(config)
    vc.get_vc(model_name)

    with tempfile.TemporaryDirectory(prefix="uvr_tmp_") as tmp:
        tmp_dir = Path(tmp)
        for song in songs:
            key = str(song.relative_to(songs_dir))
            entry = state["songs"].setdefault(key, {"source": str(song)})

            if args.sep_backend == "uvr":
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
            else:
                vocal_path, inst_path = resolve_msst_outputs(song, vocals_dir, inst_dir)
            if vocal_path is None or inst_path is None:
                entry.update({"status": "error", "error": "Separation outputs missing", "updated_at": now_iso()})
                save_state(state_path, state)
                print(f"Separation outputs missing for {song}")
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
                        mix_audio(rvc_path, inst_path, final_path, args.vocal_gain, args.inst_gain)
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
