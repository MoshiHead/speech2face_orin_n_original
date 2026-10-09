"""tools/gpu_probe.py -- report the GPU and decide which fast paths this GPU can actually run.

    python tools/gpu_probe.py [out.env]

The Orin configuration quantises the 7B trunk to int4 with ATen's tinygemm kernels
(torch.ops.aten._weight_int4pack_mm). That kernel is not compiled for every CUDA
architecture, so on some cards it raises instead of running. This script tries it with the
exact code path the server uses (src/niloys/int4_linear.py) and writes a shell-sourceable file:

    TRY19_INT4_G=32          # int4 trunk works -> use it (needs ~14 GB of VRAM)
    TRY19_INT4_G=0           # int4 unavailable  -> bf16 trunk (needs ~26 GB of VRAM)
    TRY19_DEP_GATING_INT4=1/0

run_x86.sh sources that file automatically if it is next to the package as .gpu_probe.env.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "niloys"))

import torch  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, ".gpu_probe.env")


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: torch cannot see a CUDA GPU", file=sys.stderr)
        return 1
    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    vram = props.total_memory / 2 ** 30
    print(f"GPU            : {props.name}  sm_{cc[0]}{cc[1]}  {vram:.1f} GiB")
    print(f"torch          : {torch.__version__} (CUDA {torch.version.cuda})")
    try:
        import tensorrt
        print(f"TensorRT       : {tensorrt.__version__}")
    except Exception as e:
        print(f"TensorRT       : NOT AVAILABLE ({type(e).__name__}) -- the SEANet decoder "
              "stays in PyTorch (a few ms slower, same audio)")

    # bf16 is required: the trunk weights are bf16 and the int4 kernels take bf16 activations.
    if cc[0] < 8:
        print("bf16           : NOT SUPPORTED on sm_%d%d -- this pipeline needs Ampere or newer"
              % cc, file=sys.stderr)
        return 1
    print("bf16           : OK")

    int4_ok, int4_err = False, ""
    try:
        from int4_linear import Int4Linear, group_quantize
        lin = torch.nn.Linear(512, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        packed, sz = group_quantize(lin.weight.data, 32)
        q = Int4Linear(packed, sz, None, 256, 32)
        x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
        ref, got = lin(x).float(), q(x).float()
        cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), 0).item()
        if not torch.isfinite(got).all():
            raise RuntimeError("int4 matmul produced non-finite values")
        if cos < 0.9:
            raise RuntimeError(f"int4 matmul cosine {cos:.3f} vs bf16 -- kernel misbehaving")
        int4_ok = True
        print(f"int4 tinygemm  : OK (cosine {cos:.4f} vs bf16)")
    except Exception as e:                                   # noqa: BLE001
        int4_err = f"{type(e).__name__}: {e}"
        print(f"int4 tinygemm  : UNAVAILABLE -- {int4_err}")

    try:
        f = torch.compile(lambda t: torch.sin(t) * 2 + 1)
        t = torch.randn(1024, device="cuda")
        assert torch.allclose(f(t), torch.sin(t) * 2 + 1, atol=1e-5)
        print("torch.compile  : OK")
    except Exception as e:                                   # noqa: BLE001
        print(f"torch.compile  : FAILED ({type(e).__name__}: {e})")

    need = 14 if int4_ok else 26
    print(f"\nplan           : {'int4 G=32 trunk' if int4_ok else 'bf16 trunk (int4 kernels unusable)'}"
          f", needs ~{need} GB of VRAM, this GPU has {vram:.0f} GB")
    if vram < need:
        print(f"WARNING        : only {vram:.0f} GB of VRAM -- expect CUDA OOM. Use a bigger GPU "
              f"(>= {need} GB).", file=sys.stderr)

    with open(OUT, "w") as fh:
        fh.write("# written by tools/gpu_probe.py -- sourced by run_x86.sh\n")
        fh.write(f"TRY19_INT4_G={32 if int4_ok else 0}\n")
        fh.write(f"TRY19_DEP_GATING_INT4={1 if int4_ok else 0}\n")
    print(f"wrote          : {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
