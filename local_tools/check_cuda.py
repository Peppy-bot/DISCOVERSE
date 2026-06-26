#!/usr/bin/env python3
"""
DISCOVERSE CUDA / GPU 验证脚本 (CUDA / GPU verification)

Verifies that the GPU stack this pixi environment is built around actually works:

  1. PyTorch sees a CUDA device  (torch.cuda.is_available + device name)
  2. This GPU's arch is in torch's build list  (sm_120 / Blackwell for the RTX 5090)
  3. A real GPU matmul runs and the result matches the CPU  (kernels actually execute)
  4. gsplat's CUDA kernels load and run  (the 3DGS renderer backend compiles & works)

These mirror the four things the pixi.toml CUDA stack is pinned for. Each check is
independent: a failure is reported but the remaining checks still run, so one run
tells you everything that is wrong.

Usage:
    pixi run check-cuda
    python local_tools/check_cuda.py
"""

import sys

# Arch this environment is pinned to (see TORCH_CUDA_ARCH_LIST in pixi.toml).
# RTX 5090 / Blackwell == compute capability 12.0 == sm_120.
EXPECTED_ARCH = "sm_120"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"


def _section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def check_torch_import():
    """torch must import before anything else can be checked."""
    _section("PyTorch import")
    try:
        import torch
        print(f"  {PASS} torch {torch.__version__}")
        print(f"    built with CUDA: {torch.version.cuda}")
        return torch
    except Exception as e:
        print(f"  {FAIL} could not import torch: {e}")
        return None


def check_device(torch) -> bool:
    """A usable CUDA device must be visible to torch."""
    _section("CUDA device")
    if not torch.cuda.is_available():
        print(f"  {FAIL} torch.cuda.is_available() is False")
        print("    No CUDA device visible. Check the NVIDIA driver and that this")
        print("    is the GPU pytorch build (cuda129_*), not the CPU build.")
        return False

    n = torch.cuda.device_count()
    print(f"  {PASS} {n} CUDA device(s) visible")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"    [{i}] {name}  (sm_{major}{minor}, {total_gb:.1f} GiB)")
    return True


def check_arch(torch) -> bool:
    """This GPU's sm_ arch must be in torch's compiled arch list."""
    _section(f"Arch support ({EXPECTED_ARCH})")
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception as e:
        print(f"  {FAIL} could not read torch arch list: {e}")
        return False

    print(f"    torch arch list: {', '.join(arch_list) or '(empty)'}")

    if EXPECTED_ARCH in arch_list:
        print(f"  {PASS} {EXPECTED_ARCH} present — kernels are built for this GPU")
        return True

    # PTX for an older arch can JIT forward-compat onto sm_120, so this is a warning,
    # not a hard fail — but it means no native sm_120 kernels were compiled.
    print(f"  {WARN} {EXPECTED_ARCH} not in torch's arch list")
    print("    The GPU may still run via PTX JIT, but native Blackwell kernels are")
    print("    missing. Expected the cuda129_* pytorch build (see pixi.toml).")
    return False


def check_matmul(torch) -> bool:
    """Run a real matmul on the GPU and confirm it matches the CPU result."""
    _section("GPU matmul")
    try:
        torch.manual_seed(0)
        a = torch.randn(512, 512)
        b = torch.randn(512, 512)
        expected = a @ b

        a_gpu, b_gpu = a.cuda(), b.cuda()
        got = (a_gpu @ b_gpu).cpu()
        torch.cuda.synchronize()

        if torch.allclose(got, expected, atol=1e-3, rtol=1e-3):
            print(f"  {PASS} 512x512 matmul on GPU matches CPU (atol=1e-3)")
            return True

        max_err = (got - expected).abs().max().item()
        print(f"  {FAIL} GPU result diverges from CPU (max abs err {max_err:.2e})")
        return False
    except Exception as e:
        print(f"  {FAIL} GPU matmul failed: {e}")
        return False


def check_gsplat(torch) -> bool:
    """gsplat's CUDA kernels (the 3DGS renderer backend) must load and run."""
    _section("gsplat CUDA kernels")
    try:
        import gsplat
    except Exception as e:
        print(f"  {FAIL} could not import gsplat: {e}")
        return False

    print(f"  {PASS} gsplat {getattr(gsplat, '__version__', '(unknown version)')}")

    # Project a single trivial Gaussian. This forces gsplat to JIT-compile/load and
    # actually launch its CUDA kernels — the real test of the gsplat backend.
    try:
        means = torch.zeros(1, 3, device="cuda")
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
        scales = torch.ones(1, 3, device="cuda") * 0.1
        opacities = torch.ones(1, device="cuda")
        viewmats = torch.eye(4, device="cuda").unsqueeze(0)
        Ks = torch.tensor(
            [[[300.0, 0.0, 150.0], [0.0, 300.0, 150.0], [0.0, 0.0, 1.0]]],
            device="cuda",
        )

        radii, *_ = gsplat.fully_fused_projection(
            means, None, quats, scales, viewmats, Ks, width=300, height=300
        )
        torch.cuda.synchronize()
        print(f"  {PASS} gsplat CUDA kernels ran (projected radii shape {tuple(radii.shape)})")
        return True
    except Exception as e:
        print(f"  {FAIL} gsplat CUDA kernel launch failed: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("DISCOVERSE CUDA / GPU check")
    print("=" * 60)

    torch = check_torch_import()
    if torch is None:
        print(f"\n{FAIL} torch is unavailable — cannot run any GPU checks.")
        return 1

    # Run device first; if there's no device the rest can't pass, but keep going so
    # the report shows every failure at once.
    have_device = check_device(torch)

    results = {
        "device": have_device,
        "arch": check_arch(torch),
    }
    if have_device:
        results["matmul"] = check_matmul(torch)
        results["gsplat"] = check_gsplat(torch)
    else:
        print("\n  (skipping matmul + gsplat: no CUDA device)")
        results["matmul"] = False
        results["gsplat"] = False

    _section("Summary")
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL} {name}")

    all_ok = all(results.values())
    print()
    if all_ok:
        print(f"{PASS} All CUDA checks passed — GPU stack is good to go.")
        return 0
    print(f"{FAIL} Some CUDA checks failed — see details above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
