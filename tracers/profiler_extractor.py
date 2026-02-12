import json
import csv
import time
import os
import gzip
from collections import defaultdict

def wait_for_trace_file(trace_dir, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        for f in os.listdir(trace_dir):
            if f.endswith(".pt.trace.json"):
                return os.path.join(trace_dir, f)
        time.sleep(0.2)
    raise RuntimeError("torch profiler trace file not found")


def find_vllm_trace_file(output_dir, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        for f in os.listdir(output_dir):
            if f.endswith(".json"):
                return os.path.join(output_dir, f)
            elif f.endswith(".pt.trace.json.gz"):
                with gzip.open(os.path.join(output_dir, f), "rt") as file:
                    trace = json.load(file)
                    output_file = os.path.join(output_dir, f.replace(".pt.trace.json.gz", ".json"))
                    with open(output_file, "w") as f:
                        json.dump(trace, f, indent=2)
                return output_file
        time.sleep(0.2)
    raise RuntimeError("vLLM profiler trace file not found")


def format_shapes(input_dims, input_types):
    shapes = []
    if input_dims and input_types and len(input_dims) == len(input_types):
        for d, t in zip(input_dims, input_types):
            shapes.append(f"({t}, {tuple(d) if isinstance(d, list) else d})")
    return shapes


def write_cpu_kernel_csv(json_output, csv_path):
    """
    Write CSV with:
    CPU_OP_NAME | CPU_OP_TOTAL_COUNT | KERNEL_NAME | KERNEL_TOTAL_COUNT
    """

    # cpu_op_name -> { total_count, kernel_counts }
    cpu_summary = {}

    for entry in json_output:
        cpu = entry["cpu_op_name"]
        if cpu not in cpu_summary:
            cpu_summary[cpu] = {
                "total_count": 0,
                "kernel_counts": defaultdict(int)
            }

        cpu_summary[cpu]["total_count"] += 1

        for k, cnt in entry.get("kernel_call_count", {}).items():
            cpu_summary[cpu]["kernel_counts"][k] += cnt

    # write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cpu_op_name",
            "cpu_op_total_count",
            "gpu_kernel_name",
            "gpu_kernel_total_count"
        ])

        for cpu_op, data in cpu_summary.items():
            kernels = data["kernel_counts"]

            if not kernels:
                # CPU op with no kernels
                writer.writerow([cpu_op, data["total_count"], "", ""])
                continue

            first = True
            for kernel_name, kernel_cnt in kernels.items():
                if first:
                    writer.writerow([
                        cpu_op,
                        data["total_count"],
                        kernel_name,
                        kernel_cnt
                    ])
                    first = False
                else:
                    writer.writerow([
                        "",
                        "",
                        kernel_name,
                        kernel_cnt
                    ])


def extract_kernel_calls_from_profile(input_file, output_file):
    with open(input_file, "r") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])

    # ---------------- storage ----------------
    op_metrics = defaultdict(lambda: {
        "sequence_number": None,
        "cpu_op_name": None,
        "cpu_op_cat": None,
        "cpu_external_id": None,
        "cpu_time_us": 0.0,
        "kernel_call_count": defaultdict(int),
        "kernel_call_count_by_shape": defaultdict(lambda: defaultdict(int)),
        "kernel_call_count_by_stream": defaultdict(lambda: defaultdict(int)),
        "kernel_durations": [],
        "associated_kernels": [],
        "total_kernel_time_us": 0.0,
        "input_dims_count": defaultdict(int),
        "call_stack": None,
    })

    extid_to_seq = {}
    seq_to_corr = defaultdict(set)
    corr_to_kernel = defaultdict(list)

    # ---------------- pass 1 ----------------
    for e in events:
        ph = e.get("ph")
        cat = e.get("cat", "")
        name = e.get("name", "")
        args = e.get("args", {})

        seq = args.get("Sequence number")
        ext = args.get("External id")
        corr = args.get("correlation")

        # ----- CPU op -----
        if ph == "X" and cat in ("cpu_op", "operator"):

            key = seq if seq is not None else ext
            m = op_metrics[key]

            m["sequence_number"] = seq
            m["cpu_op_name"] = name
            m["cpu_op_cat"] = cat
            m["cpu_external_id"] = ext
            m["cpu_time_us"] += e.get("dur", 0.0)
            cs = args.get("Call stack")
            if cs and m["call_stack"] is None:
                m["call_stack"] = cs.split(";")

            if ext is not None:
                extid_to_seq[ext] = key

            # input shapes
            shapes = format_shapes(
                args.get("Input Dims"),
                args.get("Input type")
            )
            shape_key = ";".join(shapes) if shapes else "N/A"
            m["__shape_key"] = shape_key
            for s in shapes:
                m["input_dims_count"][s] += 1

        # ----- CUDA runtime -----
        elif ph == "X" and cat == "cuda_runtime":
            if ext in extid_to_seq and corr is not None:
                seq_to_corr[extid_to_seq[ext]].add(corr)

        # ----- GPU kernel -----
        elif ph == "X" and ("kernel" in cat or "gpu" in cat):
            if corr is not None:
                corr_to_kernel[corr].append({
                    "kernel_name": name,
                    "kernel_cat": cat,
                    "kernel_external_id": corr,
                    "kernel_duration_us": round(e.get("dur", 0.0), 3),
                    "stream_id": e.get("tid"),
                })

    # ---------------- pass 2 ----------------
    for seq, m in op_metrics.items():
        shape_key = m.get("__shape_key", "N/A")
        for corr in seq_to_corr.get(seq, []):
            for k in corr_to_kernel.get(corr, []):
                m["associated_kernels"].append(k)
                m["total_kernel_time_us"] += k["kernel_duration_us"]

                name = k["kernel_name"]
                stream = str(k.get("stream_id", "unknown"))
                dur = k["kernel_duration_us"]
                m["kernel_call_count"][name] += 1
                m["kernel_call_count_by_shape"][shape_key][name] += 1
                m["kernel_call_count_by_stream"][stream][name] += 1
                m["kernel_durations"].append(dur)

    # ---------------- output ----------------
    output = []
    for m in op_metrics.values():
        if m["cpu_op_name"] is None:
            continue

        durations = m["kernel_durations"]
        if durations:
            stats = {
                "min": round(min(durations), 3),
                "max": round(max(durations), 3),
                "avg": round(sum(durations) / len(durations), 3),
            }
        else:
            stats = {"min": 0.0, "max": 0.0, "avg": 0.0}

        output.append({
            "sequence_number": m["sequence_number"],
            "cpu_op_name": m["cpu_op_name"],
            "cpu_op_cat": m["cpu_op_cat"],
            "cpu_external_id": m["cpu_external_id"],
            "cpu_time_us": round(m["cpu_time_us"], 3),
            "total_kernel_time_us": round(m["total_kernel_time_us"], 3),
            "kernel_call_count": dict(m["kernel_call_count"]),
            "associated_kernels": m["associated_kernels"],
            "kernel_call_count_by_shape": {
                k: dict(v) for k, v in m["kernel_call_count_by_shape"].items()
            },
            "kernel_call_count_by_stream": {
                k: dict(v) for k, v in m["kernel_call_count_by_stream"].items()
            },
            "kernel_duration_stats_us": stats,
            "input_dims_count": dict(m["input_dims_count"]),
            "call_stack": m["call_stack"],
        })

    output.sort(key=lambda x: (x["sequence_number"] is None, x["sequence_number"]))

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    return output