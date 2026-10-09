"""tools/build_trt_engine.py -- build a TensorRT engine from an ONNX file with the Python API.

    python tools/build_trt_engine.py <model.onnx> <out.engine> [workspace_GiB]

Same job as `trtexec --onnx=... --saveEngine=...`, for machines where trtexec is not installed
(the pip TensorRT wheels ship the libraries and the Python bindings but no trtexec binary;
JetPack ships trtexec at /usr/src/tensorrt/bin/trtexec). Engines are tied to the GPU and the
TensorRT version, so this always runs on the machine that will serve.

The SEANet window decoder used here has fully static shapes, so no optimisation profile is
needed; precision stays fp32 to match weights/seanet_w12_fp32.engine on the Orin.
"""
import os
import sys

import tensorrt as trt


def build(onnx_path: str, engine_path: str, workspace_gib: float = 4.0) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"onnx parse error: {parser.get_error(i)}", file=sys.stderr)
            raise SystemExit(f"could not parse {onnx_path}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))
    print(f"[build_trt_engine] TensorRT {trt.__version__}: {onnx_path} -> {engine_path} "
          f"(fp32, workspace {workspace_gib:g} GiB)", flush=True)
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"    input  {t.name} {tuple(t.shape)} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"    output {t.name} {tuple(t.shape)} {t.dtype}")

    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise SystemExit("TensorRT build_serialized_network returned None")
    tmp = engine_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(plan)
    os.replace(tmp, engine_path)
    print(f"[build_trt_engine] wrote {engine_path} ({os.path.getsize(engine_path)/2**20:.1f} MiB)",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    build(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 4.0)
