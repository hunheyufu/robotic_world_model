# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime helpers shared by the RWM package on Isaac Lab 3.x / Isaac Sim 6.2.

Only one runtime is supported: Isaac Lab 3.x (``isaaclab_tasks.core`` layout), Isaac Sim 6.2 and
Python >= 3.12.  The helpers here isolate the parts of the Isaac Lab API that are not stable across
its point releases, and the data-access rules that the RWM models depend on:

* Task modules are imported from the 3.x layout (``isaaclab_tasks.core``).
* Quaternions are never converted: the 3.x tensors are already ``xyzw`` and are only moved between
  the simulator and the models, never re-interpreted.
* Never mutate simulator-owned proxy arrays in place.  Values that outlive a step are read through
  :func:`to_tensor` and explicitly cloned by the caller when they are written to.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch


def lab_generation() -> str:
    """Return the runtime tag used in logs and manifests."""
    return "lab3"


def package_versions() -> dict[str, str]:
    """Return the versions of the packages that determine the runtime contract."""
    import importlib.metadata as metadata

    versions: dict[str, str] = {}
    for name in (
        "isaaclab",
        "isaaclab_tasks",
        "isaaclab_rl",
        "isaacsim",
        "rsl-rl-lib",
        "torch",
        "tensordict",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "<not installed>"
    versions["layout"] = lab_generation()
    return versions


def import_velocity_env_cfg() -> Any:
    """Import the Isaac Lab 3.x locomotion velocity env-cfg module."""
    return importlib.import_module("isaaclab_tasks.core.velocity.velocity_env_cfg")


def import_velocity_mdp() -> Any:
    """Import the Isaac Lab 3.x locomotion velocity mdp module."""
    return importlib.import_module("isaaclab_tasks.core.velocity.mdp")


def import_anymal_rough_env_cfg() -> Any:
    """Import the Isaac Lab 3.x ANYmal-D rough env config class."""
    module = importlib.import_module("isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg")
    return module.AnymalDRoughEnvCfg


def to_tensor(value: Any) -> torch.Tensor:
    """Return the torch tensor backing *value*.

    Isaac Lab 3.x exposes articulation and contact data as Warp-backed ``ProxyArray`` objects whose
    ``.torch`` attribute is a zero-copy view.  This helper accepts both proxy arrays and plain
    tensors, and leaves already-torch values untouched.
    """
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "torch", None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    return value


def snapshot(value: Any) -> torch.Tensor:
    """Return a private copy of *value* that can outlive a simulator step."""
    return to_tensor(value).clone()


def as_env_ids(env: Any, env_ids: Any) -> torch.Tensor:
    """Normalize ``env_ids`` to a 1-D ``long`` tensor on the environment device.

    Isaac Lab 3.x may pass ``slice`` objects (e.g. ``slice(None)`` for a full reset) to event and
    command terms; this normalizes them to an index tensor.
    """
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def contact_normal_force_history(contact_sensor: Any) -> torch.Tensor:
    """Return the contact-force history shaped ``(num_envs, history, num_bodies, 3)``.

    The RWM labels are defined on the *normal* force.  Isaac Sim 6.2 moved the *total* (normal +
    tangential) force to ``net_forces_w_history``, so the normal history is selected explicitly
    instead of relying on the old name.
    """
    normal = getattr(contact_sensor.data, "net_normal_forces_w_history", None)
    if normal is None:
        raise RuntimeError(
            "The contact sensor does not expose 'net_normal_forces_w_history'. The RWM system-dynamics "
            "labels are defined on the normal force; refusing to substitute the total force."
        )
    return to_tensor(normal)


def total_force_history(contact_sensor: Any) -> torch.Tensor:
    """Return the total force history (used only for diagnostics/parity checks)."""
    return to_tensor(contact_sensor.data.net_forces_w_history)


def add_launcher_args(parser: Any) -> None:
    """Add the Isaac Lab 3.x launcher arguments to *parser*."""
    from isaaclab.app import add_launcher_args as _add_launcher_args

    _add_launcher_args(parser)


def launch(env_cfg: Any, args_cli: Any):
    """Return a context manager that starts the simulator runtime for *env_cfg*.

    This is :func:`isaaclab.app.launch_simulation`, which starts Kit only when the resolved config
    needs it.
    """
    from isaaclab.app import launch_simulation

    return launch_simulation(env_cfg, args_cli)


def resolve_task_config(task: str, entry_point: str, play_mode: bool = False, overrides: Any = None):
    """Resolve the env and agent configs, applying config overrides."""
    from isaaclab_tasks.utils.hydra import resolve_task_config as _resolve

    return _resolve(task, entry_point, play_mode=play_mode, overrides=overrides)
