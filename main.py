import argparse
import logging

from tracers.registry import list_tracers, list_tracers_for_framework, get_tracer_by_name
from status.model_download import ensure_models_downloaded
from core.orchestrator import trace_model_orchestrated
from core.config import Config

def _cmd_list_tracers(framework: str | None):
    if framework:
        tracers = list_tracers_for_framework(framework)
        print(f"Available tracers for framework={framework}:")
        for t in tracers:
            print(f"  - {t.name}")
    else:
        tracers = list_tracers()
        print("Available tracers:")
        for t in tracers:
            print(
                f"  - {t.name} "
                f"(frameworks={','.join(sorted(t.supported_frameworks))})"
            )
    return 0


def _cmd_describe_tracer(name: str):
    tracer = get_tracer_by_name(name)

    print(f"Tracer: {tracer.name}")
    if tracer.description:
        print(f"Description: {tracer.description}")

    if tracer.supported_frameworks:
        print(f"Supported frameworks: {', '.join(tracer.supported_frameworks)}")
    else:
        print("Supported frameworks: all")

    print(f"Requires subprocess: {tracer.requires_subprocess}")
    if tracer.exclusive_group:
        print(f"Exclusive group: {tracer.exclusive_group}")

    return 0


def load_models(args) -> list[str]:
    models = []

    if args.models:
        models.extend(
            [m.strip() for m in args.models.split(",") if m.strip()]
        )

    if args.models_file:
        with open(args.models_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    models.append(line)

    if not models:
        if not args.model:
            raise ValueError("No model specified")
        models = [args.model]

    seen = set()
    uniq = []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def load_skip_models(args) -> set[str]:
    skips = set()

    if args.skip_models:
        skips |= {
            m.strip()
            for m in args.skip_models.split(",")
            if m.strip()
        }

    if args.skip_models_file:
        with open(args.skip_models_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    skips.add(line)

    return skips


def main():
    parser = argparse.ArgumentParser("kernel tracer")

    parser.add_argument("--framework")
    parser.add_argument("--model")
    parser.add_argument(
        "--models",
        help="comma separated model names",
    )
    parser.add_argument(
        "--models-file",
        help="file containing model names, one per line",
    )
    parser.add_argument("--skip-models")
    parser.add_argument("--skip-models-file")
    parser.add_argument("--if-exists", default="skip")

    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="eval")

    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    parser.add_argument("--tracers", default="all")

    parser.add_argument(
        "--list-tracers",
        action="store_true",
        help="List available tracers (optionally filtered by --framework)"
    )
    parser.add_argument("--describe-tracer")
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="concurrent model download workers",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config",
    )

    args = parser.parse_args()
    cfg = Config(args.config)
    cfg.apply_bootstrap_env()
    logging.basicConfig(level=logging.INFO)

    # ------------------------------------------------------------
    # CLI-only paths (no orchestrator / no worker)
    # ------------------------------------------------------------
    if args.list_tracers:
        return _cmd_list_tracers(args.framework)

    if args.describe_tracer:
        return _cmd_describe_tracer(args.describe_tracer)

    # ------------------------------------------------------------
    # Normal tracing path
    # ------------------------------------------------------------
    if not args.framework or not args.model or not args.output_dir:
        parser.error("--framework, --model, --output-dir are required")

    if args.tracers == "all":
        tracers = list_tracers_for_framework(args.framework)
    else:
        names = [x.strip() for x in args.tracers.split(",")]
        tracers = []
        for n in names:
            t = get_tracer_by_name(n)
            if not t.supports(args.framework):
                raise ValueError(
                    f"Tracer '{n}' does not support framework '{args.framework}'"
                )
            tracers.append(t)

    models = load_models(args)
    skip_models = load_skip_models(args)

    for model in models:
        if model in skip_models:
            logging.info("[skip-model] %s", model)
            continue
        ensure_models_downloaded(
            models,
            max_workers=args.download_workers,
        )
        trace_model_orchestrated(
            framework=args.framework,
            model=args.model,
            tracers=tracers,
            output_root=args.output_dir,
            device=args.device,
            mode=args.mode,
            tensor_parallel_size=args.tensor_parallel_size,
            cfg=cfg
        )


if __name__ == "__main__":
    main()
