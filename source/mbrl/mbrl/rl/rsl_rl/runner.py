# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner construction and checkpoint handling for the pinned RWM ``rsl_rl`` fork.

The fork stores everything in one file::

    {
        "model_state_dict": ...,                    # actor-critic
        "system_dynamics_state_dict": ...,           # world model
        "optimizer_state_dict": ...,
        "system_dynamics_optimizer_state_dict": ...,
        "iter": int,
        "infos": dict | None,
    }

Finetune needs the world model *without* restoring the policy (it starts the policy from scratch)
while ``play``/``visualize`` need the policy with the world model loaded from a separate run.  The
fork's own ``load`` couples the two, so loading is done here with every stage spelled out.

Two loading intents are supported and kept distinct:

* **resume** - the same run continues: policy, world model and iteration counter come from one
  checkpoint, so a pretrain resume restores the world-model weights instead of silently starting
  from a fresh ensemble;
* **warm start** - the weights come from another run: the policy is optional, the world model is
  read from the artifact the caller names, and the iteration counter stays at zero.

Each optimizer follows the checkpoint that supplied its weights.  Optimizer state is not restored
by default (``--resume_optimizer``): policy/optimizer state saved by the Isaac Lab 2.2.1 runtime is
accepted for weights, but resuming an Adam state across a different torch build is not part of this
migration.
"""

from __future__ import annotations

import os
from typing import Any

import torch


RUNNER_CLASSES = ("OnPolicyRunner", "MBPOOnPolicyRunner")


def create_runner(env, agent_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    """Instantiate the runner named by ``agent_cfg['class_name']`` using the vendored fork.

    Args:
        env: Environment already wrapped as a fork-compatible ``VecEnv``.
        agent_cfg: Runner configuration dictionary (``agent_cfg.to_dict()``).
        log_dir: Directory for checkpoints, tensorboard events and git state.
        device: Torch device string.

    Returns:
        A runner instance whose ``learn``/``load``/``save`` API matches the fork.
    """
    from mbrl.rl.rsl_rl_fork.runners import MBPOOnPolicyRunner, OnPolicyRunner

    class_name = agent_cfg.get("class_name")
    if class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=device)
    if class_name == "MBPOOnPolicyRunner":
        return MBPOOnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=device)
    raise ValueError(f"Unsupported runner class {class_name!r}; expected one of {RUNNER_CLASSES}.")


def describe_checkpoint(path: str) -> dict[str, Any]:
    """Return a small, printable description of a checkpoint file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not a mapping produced by the RWM runner.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a dictionary; got {type(state).__name__}.")
    description: dict[str, Any] = {"path": os.path.abspath(path), "keys": sorted(state.keys())}
    if "iter" in state:
        description["iter"] = state["iter"]
    for key in ("model_state_dict", "system_dynamics_state_dict"):
        if key in state:
            description[key] = {
                name: tuple(tensor.shape) for name, tensor in state[key].items() if hasattr(tensor, "shape")
            }
    return description


def _load_state_dict(module: torch.nn.Module, state_dict: dict, *, what: str, path: str, allow_partial: bool) -> None:
    """Load *state_dict* into *module*, reporting the offending entry on mismatch."""
    own = module.state_dict()
    missing = sorted(set(own) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(own))
    mismatched = sorted(
        name
        for name in set(own) & set(state_dict)
        if hasattr(state_dict[name], "shape") and tuple(own[name].shape) != tuple(state_dict[name].shape)
    )
    if mismatched:
        details = ", ".join(f"{name}: checkpoint {tuple(state_dict[name].shape)} != model {tuple(own[name].shape)}" for name in mismatched)
        raise ValueError(
            f"Cannot load the {what} from '{path}': shape mismatch ({details}). "
            "The checkpoint was produced by a different observation/action layout or ensemble size."
        )
    if (missing or unexpected) and not allow_partial:
        raise ValueError(
            f"Cannot load the {what} from '{path}': missing keys {missing}, unexpected keys {unexpected}. "
            "Pass allow_partial=True only for a checkpoint you have verified by hand."
        )
    if allow_partial and (missing or unexpected):
        print(f"[WARN] Loading the {what} from '{path}' partially: missing {missing}, unexpected {unexpected}.")
    try:
        module.load_state_dict(state_dict, strict=not allow_partial)
    except RuntimeError as exc:  # pragma: no cover - defensive, details already reported above
        raise ValueError(f"Failed to load the {what} from '{path}': {exc}") from exc
    print(f"[INFO] Loaded {what} from '{path}' ({len(own)} tensors).")


def load_checkpoint(
    runner,
    path: str,
    *,
    load_policy: bool = True,
    load_system_dynamics: bool | None = None,
    dynamics_path: str | None = None,
    load_optimizer: bool = False,
    restore_iteration: bool = True,
    map_location: str | None = None,
    allow_partial: bool = False,
) -> dict | None:
    """Load the requested parts of a checkpoint into *runner*.

    Each stage is restored from the file that actually holds it: the policy (and its iteration
    counter) from *path*, the world model (and its optimizer) from *dynamics_path* when given,
    otherwise from the runner configuration's ``system_dynamics_load_path`` or from *path* itself.
    Optimizer state is restored per stage and a missing entry raises instead of being skipped, so a
    resume never continues with a silently re-initialized optimizer.

    Args:
        runner: Runner created by :func:`create_runner`.
        path: Checkpoint holding the policy (and possibly the world model).
        load_policy: Restore ``model_state_dict``.  Set ``False`` to start the policy from scratch.
        load_system_dynamics: Restore ``system_dynamics_state_dict``.  ``None`` keeps the runner
            configuration's ``load_system_dynamics`` flag.
        dynamics_path: Explicit file for the world-model weights.  Used by a resume, where the
            world model must come from the same run as the policy.  When ``None``, the runner
            configuration's ``system_dynamics_load_path`` is used, falling back to *path*.
        load_optimizer: Restore the optimizer state of every restored stage.  Off by default;
            cross-runtime resume is not promised.
        restore_iteration: Restore ``iter`` into ``runner.current_learning_iteration``.  A warm
            start from a different run must leave the iteration counter at zero.
        map_location: ``torch.load`` device mapping, typically ``agent_cfg.device``.
        allow_partial: Load with ``strict=False`` for hand-verified legacy artifacts.

    Returns:
        The ``infos`` entry of the policy checkpoint, or ``None``.
    """
    loaded = torch.load(path, weights_only=False, map_location=map_location)
    if not isinstance(loaded, dict):
        raise ValueError(f"Checkpoint {path} does not contain a dictionary.")
    if load_policy and "model_state_dict" not in loaded:
        raise KeyError(f"Checkpoint {path} has no 'model_state_dict'; keys: {sorted(loaded.keys())}")

    infos = loaded.get("infos")

    if load_policy:
        _load_state_dict(
            runner.alg.policy, loaded["model_state_dict"], what="policy", path=path, allow_partial=allow_partial
        )
        if restore_iteration and "iter" in loaded:
            runner.current_learning_iteration = loaded["iter"]
        elif not restore_iteration:
            print(f"[INFO] Warm start: keeping iteration counter at {runner.current_learning_iteration}.")

    dynamics_state = None
    if load_system_dynamics is None:
        load_system_dynamics = bool(runner.cfg.get("load_system_dynamics", False))
    if load_system_dynamics:
        source_path = dynamics_path or runner.cfg.get("system_dynamics_load_path") or path
        if os.path.abspath(source_path) == os.path.abspath(path):
            dynamics_state = loaded
        else:
            dynamics_state = torch.load(source_path, weights_only=False, map_location=map_location)
        if not isinstance(dynamics_state, dict) or "system_dynamics_state_dict" not in dynamics_state:
            raise KeyError(
                f"Dynamics checkpoint '{source_path}' has no 'system_dynamics_state_dict'; "
                f"keys: {sorted(dynamics_state.keys()) if isinstance(dynamics_state, dict) else type(dynamics_state)}"
            )
        _load_state_dict(
            runner.alg.system_dynamics,
            dynamics_state["system_dynamics_state_dict"],
            what="system dynamics",
            path=source_path,
            allow_partial=allow_partial,
        )
        dynamics_source_path = source_path

    if load_optimizer:
        # Each optimizer follows the checkpoint its weights came from.  A missing entry is an error:
        # continuing with a fresh Adam state silently discards the resume.
        if load_policy:
            if "optimizer_state_dict" not in loaded:
                raise KeyError(
                    f"Cannot restore the policy optimizer: checkpoint '{path}' has no "
                    f"'optimizer_state_dict'; keys: {sorted(loaded.keys())}"
                )
            try:
                runner.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"Failed to restore the policy optimizer from '{path}': {exc}") from exc
            print(f"[INFO] Restored policy optimizer state from '{path}'.")
        if load_system_dynamics:
            if "system_dynamics_optimizer_state_dict" not in dynamics_state:
                raise KeyError(
                    f"Cannot restore the world-model optimizer: checkpoint '{dynamics_source_path}' has no "
                    f"'system_dynamics_optimizer_state_dict'; keys: {sorted(dynamics_state.keys())}"
                )
            try:
                runner.alg.system_dynamics_optimizer.load_state_dict(
                    dynamics_state["system_dynamics_optimizer_state_dict"]
                )
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to restore the world-model optimizer from '{dynamics_source_path}': {exc}"
                ) from exc
            print(f"[INFO] Restored world-model optimizer state from '{dynamics_source_path}'.")

    return infos
