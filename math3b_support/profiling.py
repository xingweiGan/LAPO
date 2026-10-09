"""Opt-in timing and GPU-memory profiling helpers.

The profiler intentionally accepts only a fixed set of phase names.  This keeps
NVTX labels and exported metric keys free of prompts, responses, token IDs, or
other training data.  When profiling is disabled, its context managers are
no-ops and no output file is created.
"""

from __future__ import annotations

import atexit
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import threading
from timeit import default_timer
from typing import Any

import torch


# Keep these names static: they are also used as NVTX range labels and W&B keys.
PROFILE_PHASES: tuple[str, ...] = (
    "reference_precompute",
    "rollout",
    "reward",
    "tokenization",
    "old_logprob",
    "forward",
    "loss",
    "backward",
    "grad_metrics",
    "param_diagnostics",
    "allocator_cleanup",
    "optimizer",
    "zero_grad",
    "evaluation",
    "controller_validation",
    "controller_update",
    "logging",
)
_PROFILE_PHASE_SET = frozenset(PROFILE_PHASES)
_CORE_PHASES = ("rollout", "forward", "loss", "backward", "optimizer")
# These phases can involve vLLM child processes, whose allocations are invisible
# to PyTorch's parent-process allocator counters.  NVML still polls one outer
# window continuously; limiting phase-level NVML windows avoids a process-tree
# query for every training microbatch.
_NVML_PHASES = frozenset(
    ("reference_precompute", "rollout", "evaluation", "controller_validation")
)
_WANDB_MEMORY_METRICS = frozenset(
    (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_allocated_increment_bytes",
        "peak_reserved_increment_bytes",
        "peak_device_used_bytes",
        "peak_device_used_bytes_increment",
        "peak_process_tree_used_bytes",
        "peak_process_tree_used_bytes_increment",
    )
)


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _distribution(values: Sequence[float | int]) -> dict[str, float | int]:
    """Return JSON- and W&B-friendly population statistics."""
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return {"count": 0}
    ordered = sorted(finite_values)
    return {
        "count": len(ordered),
        "mean": float(statistics.fmean(ordered)),
        "std": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
        "min": float(ordered[0]),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": float(ordered[-1]),
    }


@dataclass(frozen=True)
class _CudaDevice:
    index: int

    @property
    def torch_device(self) -> torch.device:
        return torch.device("cuda", self.index)

    @property
    def label(self) -> str:
        return f"cuda:{self.index}"


def _normalise_cuda_devices(devices: Sequence[str | int | torch.device]) -> list[_CudaDevice]:
    if not torch.cuda.is_available():
        return []

    result: list[_CudaDevice] = []
    seen: set[int] = set()
    count = torch.cuda.device_count()
    for value in devices:
        try:
            if isinstance(value, int):
                index = value
            else:
                device = torch.device(value)
                if device.type != "cuda":
                    continue
                index = torch.cuda.current_device() if device.index is None else device.index
            if index < 0 or index >= count or index in seen:
                continue
            # This also rejects devices that are visible but cannot be initialised.
            torch.cuda.get_device_properties(index)
        except (AssertionError, RuntimeError, TypeError, ValueError):
            continue
        seen.add(index)
        result.append(_CudaDevice(index=index))
    return result


def _synchronise(devices: Sequence[_CudaDevice]) -> None:
    for device in devices:
        try:
            torch.cuda.synchronize(device.torch_device)
        except AssertionError:
            # A build without CUDA support may expose a stub CUDA namespace.
            # RuntimeError is deliberately not swallowed: synchronize is where
            # asynchronous CUDA kernel failures surface.
            continue


def _prepare_torch_memory(devices: Sequence[_CudaDevice]) -> dict[str, dict[str, int]]:
    snapshots: dict[str, dict[str, int]] = {}
    for device in devices:
        try:
            torch.cuda.reset_peak_memory_stats(device.torch_device)
            snapshots[device.label] = {
                "start_allocated_bytes": int(torch.cuda.memory_allocated(device.torch_device)),
                "start_reserved_bytes": int(torch.cuda.memory_reserved(device.torch_device)),
            }
        except (AssertionError, RuntimeError):
            continue
    return snapshots


def _finish_torch_memory(
    devices: Sequence[_CudaDevice],
    starts: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    snapshots: dict[str, dict[str, int]] = {}
    for device in devices:
        start = starts.get(device.label)
        if start is None:
            continue
        try:
            end_allocated = int(torch.cuda.memory_allocated(device.torch_device))
            end_reserved = int(torch.cuda.memory_reserved(device.torch_device))
            peak_allocated = max(
                start["start_allocated_bytes"],
                int(torch.cuda.max_memory_allocated(device.torch_device)),
            )
            peak_reserved = max(
                start["start_reserved_bytes"],
                int(torch.cuda.max_memory_reserved(device.torch_device)),
            )
        except (AssertionError, RuntimeError):
            continue
        snapshots[device.label] = {
            **start,
            "end_allocated_bytes": end_allocated,
            "end_reserved_bytes": end_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_increment_bytes": max(
                0, peak_allocated - start["start_allocated_bytes"]
            ),
            "peak_reserved_increment_bytes": max(
                0, peak_reserved - start["start_reserved_bytes"]
            ),
        }
    return snapshots


class _NVMLSampler:
    """Poll device and current-process-tree memory for overlapping windows."""

    def __init__(self, devices: Sequence[_CudaDevice], interval_s: float) -> None:
        self.available = False
        self.interval_s = interval_s
        self._pynvml: Any | None = None
        self._bindings: list[tuple[str, Any]] = []
        self._windows: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil: Any | None = None

        if not devices:
            return
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._pynvml = pynvml
            try:
                import psutil  # type: ignore[import-not-found]

                self._psutil = psutil
            except ImportError:
                self._psutil = None

            for device in devices:
                handle = self._handle_for_cuda_index(device.index)
                if handle is not None:
                    self._bindings.append((device.label, handle))
            self.available = bool(self._bindings)
        except Exception:  # Import, driver, and NVML version failures are optional.
            self._pynvml = None
            self._bindings = []

    def _handle_for_cuda_index(self, cuda_index: int) -> Any | None:
        pynvml = self._pynvml
        if pynvml is None:
            return None

        # CUDA indices are logical.  Resolve common CUDA_VISIBLE_DEVICES forms
        # without ever persisting the environment value in profiling output.
        token: str | None = None
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            tokens = [item.strip() for item in visible_devices.split(",")]
            if cuda_index < len(tokens):
                token = tokens[cuda_index]
        try:
            if token and token.isdigit():
                return pynvml.nvmlDeviceGetHandleByIndex(int(token))
            if token and (token.startswith("GPU-") or token.startswith("MIG-")):
                try:
                    return pynvml.nvmlDeviceGetHandleByUUID(token)
                except TypeError:
                    return pynvml.nvmlDeviceGetHandleByUUID(token.encode("utf-8"))
            return pynvml.nvmlDeviceGetHandleByIndex(cuda_index)
        except Exception:  # NVML exposes version-dependent exception classes.
            return None

    def _process_tree_pids(self) -> set[int]:
        root_pid = os.getpid()
        result = {root_pid}
        if self._psutil is not None:
            try:
                root = self._psutil.Process(root_pid)
                result.update(child.pid for child in root.children(recursive=True))
            except Exception:
                pass
            return result

        # psutil is optional.  Linux is the deployment target for Runpod, so a
        # small /proc fallback still includes spawned vLLM workers there.
        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return result
        parent_by_pid: dict[int, int] = {}
        try:
            for entry in proc_root.iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    status = (entry / "status").read_text(
                        encoding="utf-8", errors="replace"
                    )
                    parent_line = next(
                        line for line in status.splitlines() if line.startswith("PPid:")
                    )
                    parent_by_pid[int(entry.name)] = int(parent_line.split()[1])
                except (OSError, StopIteration, ValueError):
                    continue
        except OSError:
            return result
        changed = True
        while changed:
            changed = False
            for pid, parent_pid in parent_by_pid.items():
                if pid not in result and parent_pid in result:
                    result.add(pid)
                    changed = True
        return result

    def _running_processes(self, handle: Any) -> list[Any] | None:
        pynvml = self._pynvml
        if pynvml is None:
            return None
        processes: dict[int, Any] = {}
        queried = False
        for base_name in (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            function = None
            for suffix in ("_v3", "_v2", ""):
                candidate = getattr(pynvml, f"{base_name}{suffix}", None)
                if candidate is not None:
                    function = candidate
                    break
            if function is None:
                continue
            try:
                running_processes = function(handle)
                queried = True
                for process in running_processes:
                    processes[int(process.pid)] = process
            except Exception:
                continue
        return list(processes.values()) if queried else None

    def _read_sample(self) -> dict[str, dict[str, int | None]]:
        pynvml = self._pynvml
        if pynvml is None:
            return {}
        tree_pids = self._process_tree_pids()
        sample: dict[str, dict[str, int | None]] = {}
        for label, handle in self._bindings:
            device_used: int | None = None
            process_tree_used: int | None = None
            try:
                device_used = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            except Exception:
                pass

            processes = self._running_processes(handle)
            if processes is not None:
                process_tree_used = 0
                for process in processes:
                    if int(process.pid) not in tree_pids:
                        continue
                    value = getattr(process, "usedGpuMemory", None)
                    # NVML_VALUE_NOT_AVAILABLE is normally an unsigned sentinel.
                    if value is None:
                        continue
                    value = int(value)
                    if value < 0 or value >= (1 << 63):
                        continue
                    process_tree_used += value
            sample[label] = {
                "device_used_bytes": device_used,
                "process_tree_used_bytes": process_tree_used,
            }
        return sample

    @staticmethod
    def _initial_window(sample: dict[str, dict[str, int | None]]) -> dict[str, Any]:
        return {
            "start": {
                label: dict(values)
                for label, values in sample.items()
            },
            "peak": {
                label: dict(values)
                for label, values in sample.items()
            },
        }

    def _update_windows(self, sample: dict[str, dict[str, int | None]]) -> None:
        for window in self._windows.values():
            for label, metrics in sample.items():
                peaks = window["peak"].setdefault(label, {})
                window["start"].setdefault(label, dict(metrics))
                for metric, value in metrics.items():
                    if value is None:
                        peaks.setdefault(metric, None)
                        continue
                    previous = peaks.get(metric)
                    if previous is None or value > previous:
                        peaks[metric] = value

    def _ensure_thread(self) -> None:
        if self._thread is not None or not self.available:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="lapo-nvml-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            with self._lock:
                if not self._windows:
                    continue
                try:
                    self._update_windows(self._read_sample())
                except Exception:
                    # NVML polling is auxiliary; a transient driver error must
                    # never interrupt training.
                    continue

    def begin_window(self, key: str) -> None:
        if not self.available:
            return
        with self._lock:
            sample = self._read_sample()
            self._windows[key] = self._initial_window(sample)
            self._ensure_thread()

    def end_window(self, key: str) -> dict[str, dict[str, int | None]]:
        if not self.available:
            return {}
        with self._lock:
            self._update_windows(self._read_sample())
            window = self._windows.pop(key, None)
        if window is None:
            return {}

        result: dict[str, dict[str, int | None]] = {}
        for label, starts in window["start"].items():
            peaks = window["peak"].get(label, {})
            metrics: dict[str, int | None] = {}
            for base_name in ("device_used_bytes", "process_tree_used_bytes"):
                start = starts.get(base_name)
                peak = peaks.get(base_name)
                metrics[f"start_{base_name}"] = start
                metrics[f"peak_{base_name}"] = peak
                metrics[f"peak_{base_name}_increment"] = (
                    max(0, int(peak) - int(start))
                    if start is not None and peak is not None
                    else None
                )
            result[label] = metrics
        return result

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2.0))
            self._thread = None
        pynvml = self._pynvml
        self.available = False
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _aggregate_stage_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_phase[event["phase"]].append(event)

    result: dict[str, Any] = {}
    for phase in PROFILE_PHASES:
        phase_events = by_phase.get(phase)
        if not phase_events:
            continue
        phase_summary: dict[str, Any] = {
            "count": len(phase_events),
            "completed_count": sum(bool(event["completed"]) for event in phase_events),
            "duration_s": _distribution(
                [event["duration_s"] for event in phase_events]
            ),
        }

        torch_values: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        nvml_values: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event in phase_events:
            for label, metrics in event["torch_memory"].items():
                for metric, value in metrics.items():
                    torch_values[label][metric].append(value)
            for label, metrics in event["nvml_memory"].items():
                for metric, value in metrics.items():
                    if value is not None:
                        nvml_values[label][metric].append(value)

        if torch_values:
            phase_summary["torch_memory"] = {
                label: {
                    metric: _distribution(values)
                    for metric, values in metrics.items()
                }
                for label, metrics in torch_values.items()
            }
        if nvml_values:
            phase_summary["nvml_memory"] = {
                label: {
                    metric: _distribution(values)
                    for metric, values in metrics.items()
                }
                for label, metrics in nvml_values.items()
            }
        result[phase] = phase_summary
    return result


class StageProfiler:
    """Profile fixed training phases without recording model or dataset content.

    Args:
        enabled: If false, all context managers are no-ops.
        devices: CUDA devices whose allocator counters should be measured.
            Duplicate devices and CPU devices are ignored.
        output_path: Optional JSON path written by :meth:`close`.
        enable_nvml: Poll physical device and current process-tree GPU memory.
        nvml_interval_s: NVML polling interval.  Values below 10 ms are rejected.

    A phase cannot be nested inside another phase because resetting PyTorch peak
    allocator counters for the inner phase would corrupt the outer measurement.
    An ``outer_window`` is not a phase and is intended to contain many phases.
    """

    def __init__(
        self,
        enabled: bool,
        devices: Sequence[str | int | torch.device] = (),
        *,
        output_path: str | os.PathLike[str] | None = None,
        enable_nvml: bool = True,
        nvml_interval_s: float = 0.05,
    ) -> None:
        self.enabled = bool(enabled)
        if self.enabled and enable_nvml and (
            not math.isfinite(nvml_interval_s) or nvml_interval_s < 0.01
        ):
            raise ValueError(
                "nvml_interval_s must be finite and at least 0.01 seconds"
            )
        self.output_path = Path(output_path) if output_path is not None else None
        self._devices = _normalise_cuda_devices(devices) if self.enabled else []
        self._events: list[dict[str, Any]] = []
        self._outer_records: list[dict[str, Any]] = []
        self._active_phase = False
        self._active_outer = False
        self._manual_outer_context: Any | None = None
        self._next_window_id = 0
        self._closed = False
        self._nvml = (
            _NVMLSampler(self._devices, nvml_interval_s)
            if self.enabled and enable_nvml
            else None
        )
        self._nvml_available_at_start = bool(
            self._nvml is not None and self._nvml.available
        )
        if self.enabled:
            atexit.register(self._close_at_exit)

    @property
    def cuda_devices(self) -> tuple[str, ...]:
        return tuple(device.label for device in self._devices)

    @property
    def nvml_enabled(self) -> bool:
        return self._nvml_available_at_start

    def _window_key(self, kind: str) -> str:
        key = f"{kind}:{self._next_window_id}"
        self._next_window_id += 1
        return key

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure one fixed phase; safe as a no-op when disabled."""
        if name not in _PROFILE_PHASE_SET:
            allowed = ", ".join(PROFILE_PHASES)
            raise ValueError(f"unknown profiling phase {name!r}; expected one of: {allowed}")
        if not self.enabled:
            yield
            return
        if self._closed:
            raise RuntimeError("cannot start a phase after the profiler is closed")
        if self._active_phase:
            raise RuntimeError("profiling phases cannot be nested")

        self._active_phase = True
        try:
            _synchronise(self._devices)
        except BaseException:
            self._active_phase = False
            raise
        torch_starts = _prepare_torch_memory(self._devices)
        nvml_key = self._window_key("phase")
        capture_phase_nvml = self._nvml is not None and name in _NVML_PHASES
        if capture_phase_nvml:
            self._nvml.begin_window(nvml_key)

        nvtx_pushed = False
        if self._devices:
            try:
                torch.cuda.nvtx.range_push(name)
                nvtx_pushed = True
            except (AssertionError, RuntimeError):
                pass
        started_at = default_timer()
        completed = False
        try:
            yield
            completed = True
        finally:
            sync_error: RuntimeError | None = None
            try:
                _synchronise(self._devices)
            except RuntimeError as exc:
                sync_error = exc
            duration_s = default_timer() - started_at
            torch_memory = _finish_torch_memory(self._devices, torch_starts)
            if nvtx_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except (AssertionError, RuntimeError):
                    pass
            nvml_memory = (
                self._nvml.end_window(nvml_key) if capture_phase_nvml else {}
            )
            self._events.append(
                {
                    "phase": name,
                    "duration_s": float(duration_s),
                    "completed": completed and sync_error is None,
                    "torch_memory": torch_memory,
                    "nvml_memory": nvml_memory,
                }
            )
            self._active_phase = False
            if completed and sync_error is not None:
                raise sync_error

    # ``stage`` reads naturally at call sites and remains an exact alias.
    stage = phase

    @contextmanager
    def outer_window(
        self,
        step: int,
        *,
        optimizer_updates: int | None = None,
    ) -> Iterator[None]:
        """Measure one outer GRPO iteration and derive its aggregate metrics."""
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if optimizer_updates is not None and optimizer_updates < 0:
            raise ValueError("optimizer_updates must be non-negative")
        if not self.enabled:
            yield
            return
        if self._closed:
            raise RuntimeError("cannot start an outer window after the profiler is closed")
        if self._active_outer:
            raise RuntimeError("outer profiling windows cannot be nested")

        self._active_outer = True
        event_start = len(self._events)
        try:
            _synchronise(self._devices)
        except BaseException:
            self._active_outer = False
            raise
        nvml_key = self._window_key("outer")
        if self._nvml is not None:
            self._nvml.begin_window(nvml_key)
        started_at = default_timer()
        completed = False
        try:
            yield
            completed = True
        finally:
            sync_error: RuntimeError | None = None
            try:
                _synchronise(self._devices)
            except RuntimeError as exc:
                sync_error = exc
            window_completed = completed and sync_error is None
            outer_wall_s = float(default_timer() - started_at)
            outer_nvml_memory = (
                self._nvml.end_window(nvml_key) if self._nvml is not None else {}
            )
            outer_events = self._events[event_start:]
            stage_summary = _aggregate_stage_events(outer_events)
            phase_totals = {
                phase: sum(
                    event["duration_s"]
                    for event in outer_events
                    if event["phase"] == phase
                )
                for phase in PROFILE_PHASES
            }
            measured_optimizer_updates = sum(
                event["phase"] == "optimizer" for event in outer_events
            )
            expected_optimizer_updates = (
                measured_optimizer_updates
                if optimizer_updates is None
                else int(optimizer_updates)
            )
            updates = (
                expected_optimizer_updates
                if window_completed
                else measured_optimizer_updates
            )
            core_total_s = float(sum(phase_totals[name] for name in _CORE_PHASES))
            rollout_total_s = float(phase_totals["rollout"])
            self._outer_records.append(
                {
                    "step": step,
                    "completed": window_completed,
                    "outer_wall_s": outer_wall_s,
                    "core_total_s": core_total_s,
                    "other_s": float(outer_wall_s - core_total_s),
                    "rollout_amortized_per_optimizer_s": (
                        rollout_total_s / updates if updates > 0 else None
                    ),
                    "optimizer_updates": updates,
                    "expected_optimizer_updates": expected_optimizer_updates,
                    "measured_optimizer_phase_count": measured_optimizer_updates,
                    "stages": stage_summary,
                    "nvml_memory": outer_nvml_memory,
                }
            )
            self._active_outer = False
            if completed and sync_error is not None:
                raise sync_error

    def last_outer_scalars(self, prefix: str = "profile") -> dict[str, float]:
        """Return compact aggregate scalars for the most recent outer window."""
        if not self._outer_records:
            return {}
        return self._outer_record_scalars(self._outer_records[-1], prefix)

    def begin_outer_step(
        self,
        step: int,
        *,
        optimizer_updates: int | None = None,
    ) -> None:
        """Enter an outer window when indenting a large training loop is impractical."""
        if not self.enabled:
            return
        if self._manual_outer_context is not None:
            raise RuntimeError("an outer step is already active")
        outer_context = self.outer_window(
            step,
            optimizer_updates=optimizer_updates,
        )
        outer_context.__enter__()
        self._manual_outer_context = outer_context

    def end_outer_step(self) -> None:
        """Finish the outer window opened by :meth:`begin_outer_step`."""
        if not self.enabled:
            return
        outer_context = self._manual_outer_context
        if outer_context is None:
            raise RuntimeError("no outer step is active")
        self._manual_outer_context = None
        outer_context.__exit__(None, None, None)

    def _close_at_exit(self) -> None:
        """Best-effort partial summary if training exits before normal cleanup."""
        if self._closed:
            return
        try:
            outer_context = self._manual_outer_context
            if outer_context is not None:
                self._manual_outer_context = None
                outer_context.__exit__(RuntimeError, RuntimeError(), None)
            self.close()
        except Exception:
            # Profiling cleanup must never replace the original training error.
            pass

    @staticmethod
    def _outer_record_scalars(record: dict[str, Any], prefix: str) -> dict[str, float]:
        result = {
            f"{prefix}/time/outer_wall_s": float(record["outer_wall_s"]),
            f"{prefix}/time/core_total_s": float(record["core_total_s"]),
            f"{prefix}/time/other_s": float(record["other_s"]),
            f"{prefix}/count/optimizer_updates": float(record["optimizer_updates"]),
            f"{prefix}/count/expected_optimizer_updates": float(
                record["expected_optimizer_updates"]
            ),
        }
        amortized = record["rollout_amortized_per_optimizer_s"]
        if amortized is not None:
            result[f"{prefix}/time/rollout_amortized_per_optimizer_s"] = float(
                amortized
            )

        for phase, summary in record["stages"].items():
            duration = summary["duration_s"]
            result[f"{prefix}/time/{phase}_total_s"] = float(
                duration["mean"] * duration["count"]
            )
            result[f"{prefix}/time/{phase}_mean_s"] = float(duration["mean"])
            result[f"{prefix}/time/{phase}_p95_s"] = float(duration["p95"])
            result[f"{prefix}/count/{phase}"] = float(summary["count"])

            for source in ("torch_memory", "nvml_memory"):
                for label, metrics in summary.get(source, {}).items():
                    for metric, distribution in metrics.items():
                        if metric not in _WANDB_MEMORY_METRICS:
                            continue
                        result[
                            f"{prefix}/memory/{source}/{label}/{phase}/{metric}"
                        ] = float(distribution["max"])

        for label, metrics in record.get("nvml_memory", {}).items():
            for metric, value in metrics.items():
                if metric in _WANDB_MEMORY_METRICS and value is not None:
                    result[
                        f"{prefix}/memory/nvml/{label}/outer/{metric}"
                    ] = float(value)
        return result

    def summary(self) -> dict[str, Any]:
        """Return the complete JSON-safe aggregate summary."""
        outer_distributions: dict[str, dict[str, float | int]] = {}
        for metric in (
            "outer_wall_s",
            "core_total_s",
            "other_s",
            "rollout_amortized_per_optimizer_s",
        ):
            values = [
                record[metric]
                for record in self._outer_records
                if record[metric] is not None
            ]
            outer_distributions[metric] = _distribution(values)

        outer_stage_totals: dict[str, dict[str, float | int]] = {}
        for phase in PROFILE_PHASES:
            values = []
            for record in self._outer_records:
                phase_summary = record["stages"].get(phase)
                if phase_summary is None:
                    continue
                duration = phase_summary["duration_s"]
                values.append(float(duration["mean"] * duration["count"]))
            if values:
                outer_stage_totals[phase] = _distribution(values)

        outer_nvml_values: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in self._outer_records:
            for label, metrics in record.get("nvml_memory", {}).items():
                for metric, value in metrics.items():
                    if value is not None:
                        outer_nvml_values[label][metric].append(int(value))
        outer_nvml_summary = {
            label: {
                metric: _distribution(values)
                for metric, values in metrics.items()
            }
            for label, metrics in outer_nvml_values.items()
        }

        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "cuda_devices": list(self.cuda_devices),
            "nvml_enabled": self.nvml_enabled,
            "definitions": {
                "core_total_s": "rollout + forward + loss + backward + optimizer",
                "other_s": "outer_wall_s - core_total_s",
                "nvml_device_used_bytes": "physical device usage, including other processes",
                "nvml_process_tree_used_bytes": (
                    "current Python process plus discoverable child processes"
                ),
                "torch_phase_peaks": (
                    "allocator peak counters are reset per phase; no outer-window "
                    "PyTorch allocator peak is reported"
                ),
            },
            "outer": outer_distributions,
            "outer_stage_total_s": outer_stage_totals,
            "outer_nvml_memory": outer_nvml_summary,
            "stages": _aggregate_stage_events(self._events),
            "outer_steps": list(self._outer_records),
        }

    def summary_scalars(self, prefix: str = "profile_summary") -> dict[str, float]:
        """Return global mean/std/p50/p95 scalars suitable for one W&B log."""
        summary = self.summary()
        result: dict[str, float] = {}
        for metric, distribution in summary["outer"].items():
            for statistic in ("mean", "std", "p50", "p95"):
                value = distribution.get(statistic)
                if value is not None:
                    result[f"{prefix}/time/{metric}/{statistic}"] = float(value)

        for phase, distribution in summary["outer_stage_total_s"].items():
            for statistic in ("mean", "std", "p50", "p95"):
                result[
                    f"{prefix}/time/{phase}_outer_total_s/{statistic}"
                ] = float(distribution[statistic])

        for label, metrics in summary["outer_nvml_memory"].items():
            for metric, distribution in metrics.items():
                if metric not in _WANDB_MEMORY_METRICS:
                    continue
                for statistic in ("mean", "std", "p50", "p95", "max"):
                    result[
                        f"{prefix}/memory/nvml/{label}/outer/{metric}/{statistic}"
                    ] = float(distribution[statistic])

        for phase, phase_summary in summary["stages"].items():
            duration = phase_summary["duration_s"]
            for statistic in ("mean", "std", "p50", "p95"):
                result[f"{prefix}/time/{phase}/{statistic}_s"] = float(
                    duration[statistic]
                )
            result[f"{prefix}/count/{phase}"] = float(phase_summary["count"])

            for source in ("torch_memory", "nvml_memory"):
                for label, metrics in phase_summary.get(source, {}).items():
                    for metric, distribution in metrics.items():
                        if metric not in _WANDB_MEMORY_METRICS:
                            continue
                        for statistic in ("mean", "std", "p50", "p95", "max"):
                            result[
                                f"{prefix}/memory/{source}/{label}/{phase}/"
                                f"{metric}/{statistic}"
                            ] = float(distribution[statistic])
        return result

    def write_json(
        self,
        path: str | os.PathLike[str] | None = None,
    ) -> Path | None:
        """Atomically write the safe profiling summary to JSON."""
        destination = Path(path) if path is not None else self.output_path
        if not self.enabled or destination is None:
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
                output.write("\n")
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
        return destination

    def close(self) -> Path | None:
        """Stop NVML polling and write the configured JSON summary once."""
        if self._closed:
            return None
        if self._active_phase or self._active_outer:
            raise RuntimeError("cannot close the profiler while a window is active")
        try:
            return self.write_json()
        finally:
            if self._nvml is not None:
                self._nvml.close()
            self._closed = True

    def __enter__(self) -> StageProfiler:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
            return
        # Never replace a training exception with an auxiliary profiler-output
        # failure during stack unwinding.
        try:
            self.close()
        except Exception:
            pass


__all__ = ["PROFILE_PHASES", "StageProfiler"]
