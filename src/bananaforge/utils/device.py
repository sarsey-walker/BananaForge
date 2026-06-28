"""Device selection helpers for PyTorch execution."""

from dataclasses import dataclass, field
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class DeviceResolution:
    """Result of resolving a requested compute device."""

    requested: str
    selected: str
    fallback: bool = False
    reason: str = ""
    failed_devices: List[Tuple[str, str]] = field(default_factory=list)


def device_is_usable(device: str) -> tuple[bool, str]:
    """Return whether PyTorch can execute a basic operation on the device."""
    if device == "cpu":
        return True, ""

    if device == "cuda" and not torch.cuda.is_available():
        return False, "CUDA is not available to PyTorch"

    if device == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        return False, "MPS is not available to PyTorch"

    try:
        test_tensor = torch.ones(1, device=device)
        result = (test_tensor + 1).sum().item()
        if result != 2.0:
            return False, "device smoke test returned an unexpected result"

        if device == "cuda":
            torch.cuda.synchronize()

        return True, ""
    except Exception as exc:
        reason = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return False, reason


def resolve_device(requested_device: str) -> DeviceResolution:
    """Resolve auto devices and fall back when an accelerator cannot run kernels."""
    if requested_device == "auto":
        failed_devices = []
        for candidate in ("cuda", "mps", "cpu"):
            usable, reason = device_is_usable(candidate)
            if usable:
                return DeviceResolution(
                    requested=requested_device,
                    selected=candidate,
                    fallback=candidate != "cuda" and bool(failed_devices),
                    failed_devices=failed_devices,
                )
            failed_devices.append((candidate, reason))

    usable, reason = device_is_usable(requested_device)
    if usable:
        return DeviceResolution(requested=requested_device, selected=requested_device)

    return DeviceResolution(
        requested=requested_device,
        selected="cpu",
        fallback=True,
        reason=reason,
        failed_devices=[(requested_device, reason)],
    )
