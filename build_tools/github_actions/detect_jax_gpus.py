#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Detects ROCm GPUs visible to JAX after wheel installation.

This is meant to run *after* jax, jax_rocm7_plugin, and jax_rocm7_pjrt have been
installed in the test workflow. It asks JAX itself (via ``jax.devices()``)
whether any ROCm GPUs are usable. If none are found, the JAX test steps should
be skipped: running them without a GPU just burns CI time and reports
misleading failures.

The script never fails the job on a "no GPU" outcome. Instead it reports the
result through GitHub Actions step outputs so the caller can gate later steps:

    gpu_count=<int>          number of ROCm devices JAX can see
    gpus_detected=true|false whether at least one ROCm device was found

It also writes a one-line job summary and prints a human-readable message.

Example usage:

    python detect_jax_gpus.py

Then gate the test step with:

    if: steps.detect_gpus.outputs.gpus_detected == 'true'
"""

import os
import sys

from github_actions_api import gha_set_output, gha_append_step_summary


def count_jax_rocm_devices() -> tuple[int, str]:
    """Returns (gpu_count, detail) for ROCm devices JAX can see.

    Never raises: any import/runtime failure is treated as "0 GPUs" with the
    error captured in the detail string, because the goal is to gate tests, not
    to crash the job.
    """
    try:
        import jax
    except Exception as e:  # noqa: BLE001 - report any import failure as 0 GPUs
        return 0, f"failed to import jax: {e}"

    try:
        devices = jax.devices()
    except Exception as e:  # noqa: BLE001 - no backend / no GPU raises here
        return 0, f"jax.devices() raised: {e}"

    rocm_devices = [
        d for d in devices if getattr(d, "platform", "").lower() in ("rocm", "gpu")
    ]
    if not rocm_devices:
        return 0, f"jax.devices() returned no ROCm devices (saw: {devices})"

    detail = ", ".join(
        f"{getattr(d, 'device_kind', '?')}#{getattr(d, 'id', '?')}"
        for d in rocm_devices
    )
    return len(rocm_devices), detail


def main() -> int:
    gpu_count, detail = count_jax_rocm_devices()
    gpus_detected = gpu_count > 0

    if gpus_detected:
        print(f"Detected {gpu_count} ROCm GPU(s) visible to JAX: {detail}")
        summary = f"GPU detection: found {gpu_count} ROCm GPU(s) — running JAX tests."
    else:
        print(f"No ROCm GPUs visible to JAX ({detail}). JAX tests will be skipped.")
        summary = (
            f"GPU detection: no ROCm GPUs visible to JAX ({detail}). "
            "Skipping JAX tests to avoid wasting CI time."
        )

    gha_set_output(
        {
            "gpu_count": gpu_count,
            "gpus_detected": "true" if gpus_detected else "false",
        }
    )
    gha_append_step_summary(summary)

    # Always exit 0: a missing GPU is a "skip", not a build failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
