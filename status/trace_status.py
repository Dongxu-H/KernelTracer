import json
import os
import datetime
import time
import getpass
import socket
import shutil
import logging

_TRACE_STATUS = "_TRACE_STATUS.json"

def write_trace_status(
    *,
    output_dir: str,
    tracer: str,
    framework: str,
    model: str,
    device: str,
    mode: str,
    success: bool,
    tracer_version: str | None = None,
):
    data = {
        "tracer": tracer,
        "framework": framework,
        "model": model.__class__.__name__,
        "device": device,
        "mode": mode,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "success": success,
    }

    if tracer_version:
        data["tracer_version"] = tracer_version

    path = os.path.join(output_dir, _TRACE_STATUS)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_trace_status(tracer_dir):
    path = os.path.join(tracer_dir, _TRACE_STATUS)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def success_matches(marker, *, device, mode, framework):
    return (
        marker.get("device") == device
        and marker.get("mode") == mode
        and marker.get("framework") == framework
    )
