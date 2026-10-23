import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import sys

from .config import Settings


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def doctor(settings):
    packages = {}
    for name in ("pyannote.audio", "torch", "torchaudio", "torchcodec", "faster-whisper", "gradio", "silero-vad"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "未安装"
    import torch
    from pyannote.audio import Model, Pipeline
    from silero_vad import load_silero_vad
    load_silero_vad(onnx=True)
    return {"python": sys.version.split()[0], "packages": packages, "cuda_available": torch.cuda.is_available(),
            "hf_token_in_env": bool(os.getenv("HF_TOKEN")), "data_dir": str(settings.data_dir.resolve()),
            "imports_ok": bool(Model and Pipeline), "vad_load_ok": True,
            "note": "此检查不下载 Hugging Face 模型，也不验证 token 权限。"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地声纹录入、验证、分段和 ASR Demo")
    sub = parser.add_subparsers(dest="command", required=True)
    web = sub.add_parser("web", help="启动本地测试界面")
    web.add_argument("--port", type=int, default=7860)
    sub.add_parser("doctor", help="检查依赖和本地 VAD")
    sub.add_parser("profiles", help="列出声纹")
    enrollment = sub.add_parser("enroll", help="录入单人声纹")
    enrollment.add_argument("--name", required=True)
    enrollment.add_argument("audio", nargs="+")
    for command in ("verify", "analyze"):
        action = sub.add_parser(command)
        action.add_argument("--profile", required=True)
        action.add_argument("audio")
        if command == "analyze":
            action.add_argument("--no-asr", action="store_true")
            action.add_argument("--num-speakers", type=int)
    evaluation = sub.add_parser("evaluate", help="固定阈值批量评测（JSONL 清单）")
    evaluation.add_argument("manifest")
    evaluation.add_argument("--output", default="data/evaluation.json")
    deletion = sub.add_parser("delete-profile", help="删除指定声纹")
    deletion.add_argument("profile_id")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    try:
        if args.command == "doctor":
            emit(doctor(settings))
            return
        from .service import VoiceService
        service = VoiceService(settings)
        if args.command == "web":
            from .app import build_app
            build_app(service).launch(server_name="127.0.0.1", server_port=args.port, share=False,
                                      allowed_paths=[str((settings.data_dir / "jobs").resolve())],
                                      max_file_size="200mb", show_error=False)
        elif args.command == "profiles":
            emit([{k: p[k] for k in ("profile_id", "name", "created_at", "usable_seconds")} for p in service.profiles.list()])
        elif args.command == "enroll":
            profile = service.enroll(args.name, args.audio)
            emit({k: v for k, v in profile.items() if k not in ("centroid", "vectors")})
        elif args.command == "verify":
            emit(service.verify(args.profile, args.audio))
        elif args.command == "analyze":
            _, report, archive = service.analyze(args.profile, args.audio, not args.no_asr, args.num_speakers)
            emit({"report": str(report.resolve()), "archive": str(archive.resolve())})
        elif args.command == "evaluate":
            from .evaluation import evaluate
            from .store import write_json
            report = evaluate(service, args.manifest)
            write_json(Path(args.output), report)
            emit({"summary": report["summary"], "output": str(Path(args.output).resolve())})
        elif args.command == "delete-profile":
            service.profiles.delete(args.profile_id)
            emit({"deleted": args.profile_id})
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
