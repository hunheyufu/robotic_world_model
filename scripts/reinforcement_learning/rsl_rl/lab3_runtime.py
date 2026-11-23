# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared runtime for the RWM entry points on Isaac Lab 3.x / Isaac Sim 6.2.

The entry points use ``add_launcher_args`` plus a ``launch_simulation`` context that starts only
the runtimes the config needs.  This module owns that lifecycle plus the bookkeeping an acceptance
run needs: a runtime banner (interpreter, Lab/Sim packages, resolved physics backend), a per-run
log directory, config dumps, the runner, the checkpoint stages, and a reverse-order teardown that
fails loudly.
"""

from __future__ import annotations

import contextlib
import os
from datetime import datetime

import torch

from mbrl import compat
from mbrl.rl.rsl_rl import MbrlVecEnvWrapper, create_runner, load_checkpoint


def report_runtime() -> dict[str, str]:
    """Print and return the runtime identity of the current process."""
    import sys

    versions = compat.package_versions()
    print("[INFO] Runtime identity")
    print(f"[INFO]   python      : {sys.executable} ({sys.version.split()[0]})")
    for key, value in versions.items():
        print(f"[INFO]   {key:<12}: {value}")
    return versions


def report_backend(env_cfg) -> None:
    """Print the physics backend and Kit availability actually used by the run."""
    physics = getattr(getattr(env_cfg, "sim", None), "physics", None)
    physics_name = type(physics).__name__ if physics is not None else "<none>"
    has_kit = False
    try:
        from isaaclab.utils import has_kit

        has_kit = bool(has_kit())
    except Exception:  # noqa: BLE001 - diagnostics only
        has_kit = False
    print(f"[INFO] physics backend: {physics_name} | Kit loaded: {has_kit}")


def apply_env_overrides(args_cli, env_cfg) -> None:
    """Apply the generic environment overrides from the command line."""
    if getattr(args_cli, "num_envs", None) is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
    if getattr(args_cli, "seed", None) is not None:
        env_cfg.seed = args_cli.seed


def make_log_dir(agent_cfg, args_cli) -> str:
    """Create and return the log directory for a new run."""
    root = args_cli.log_root or os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    root = os.path.abspath(root)
    run_name = args_cli.run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Ray Tune extracts the experiment name from this line; keep the format stable.
    print(f"Exact experiment name requested from command line: {run_name}")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    return log_dir


def dump_configs(log_dir: str, env_cfg, agent_cfg) -> None:
    """Write the resolved environment and agent configuration into ``<log_dir>/params``."""
    from isaaclab.utils.io import dump_yaml

    params_dir = os.path.join(log_dir, "params")
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)


def make_env(task: str, env_cfg):
    """Create the Gymnasium environment for *task*."""
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  (registers the host tasks)
    import mbrl.tasks  # noqa: F401  (registers the RWM tasks)

    return gym.make(task, cfg=env_cfg)


def wrap_env(env, agent_cfg):
    """Wrap *env* in the fork-compatible ``VecEnv`` adapter."""
    return MbrlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def resolve_checkpoint(args_cli, agent_cfg, log_root: str, *, required: bool) -> str | None:
    """Resolve the policy checkpoint to load for training.

    Only ``--checkpoint`` and the ``resume``/``load_run`` pair of the *training* entry point are
    considered: a warm start passes ``--checkpoint`` explicitly, a resume uses the run selected by
    ``--load_run``.  The checkpoint is resolved and validated *before* the simulator starts, so a
    typo fails fast instead of after a multi-minute Kit startup.
    """
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_tasks.utils import get_checkpoint_path

    if args_cli.checkpoint:
        path = retrieve_file_path(args_cli.checkpoint)
    elif agent_cfg.resume:
        path = _latest_checkpoint(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        path = None
    if path is None and required:
        raise FileNotFoundError(
            "No policy checkpoint given. Pass --checkpoint <model.pt> or set resume/load_run in the agent config."
        )
    if path is not None:
        _require_checkpoint(path)
        print(f"[INFO] Policy checkpoint: {os.path.abspath(path)}")
    return path


def resolve_play_checkpoint(args_cli, agent_cfg, log_root: str) -> str | None:
    """Resolve the checkpoint to play, following the pre-migration ``play.py`` semantics.

    ``--use_pretrained_checkpoint`` selects the published checkpoint of the task; ``--checkpoint``
    selects an explicit file; otherwise the newest checkpoint of ``--load_run`` (default: the most
    recent run under ``logs/rsl_rl/<experiment_name>``) is used.  ``None`` means "nothing to play".
    """
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    if getattr(args_cli, "use_pretrained_checkpoint", False):
        task_name = args_cli.task.split(":")[-1]
        path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return None
        return path
    if args_cli.checkpoint:
        path = retrieve_file_path(args_cli.checkpoint)
    else:
        path = _latest_checkpoint(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    _require_checkpoint(path)
    print(f"[INFO] Policy checkpoint: {os.path.abspath(path)}")
    return path


def _latest_checkpoint(log_root: str, load_run: str, load_checkpoint: str) -> str:
    """Return the newest checkpoint of *load_run* under *log_root*, with an actionable error."""
    from isaaclab_tasks.utils import get_checkpoint_path

    try:
        return get_checkpoint_path(log_root, load_run, load_checkpoint)
    except ValueError as exc:  # raised when the run directory or the checkpoint file is missing
        raise FileNotFoundError(
            f"No checkpoint found under '{log_root}' (load_run={load_run!r}, load_checkpoint={load_checkpoint!r}). "
            "Pass --checkpoint <model.pt> or --load_run <run-directory>."
        ) from exc


def _require_checkpoint(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Policy checkpoint not found: {path}")


def resolve_dynamics_path(args_cli, agent_cfg) -> str | None:
    """Resolve the world-model checkpoint, failing before the simulator starts."""
    path = args_cli.system_dynamics_load_path or agent_cfg.system_dynamics_load_path
    if path is not None and not os.path.isfile(path):
        raise FileNotFoundError(f"System-dynamics checkpoint not found: {path}")
    if path is not None:
        print(f"[INFO] System-dynamics checkpoint: {os.path.abspath(path)}")
        agent_cfg.system_dynamics_load_path = path
    return path


def algorithm_config(agent_cfg):
    """Return the runner's ``algorithm`` sub-configuration.

    The fork consumes ``train_cfg["algorithm"]`` (``OnPolicyRunner.__init__`` stores it as
    ``self.alg_cfg`` and passes it to the algorithm constructor), so world-model hyperparameters
    must be written there.  Setting them on the runner config silently created unused attributes.
    """
    algorithm = getattr(agent_cfg, "algorithm", None)
    if algorithm is None:
        raise AttributeError(
            f"{type(agent_cfg).__name__} has no 'algorithm' sub-configuration; cannot apply world-model overrides."
        )
    return algorithm


def _set_existing(obj, name: str, value):
    """Set *name* on *obj*, refusing to create an attribute the consumer would never read."""
    if not hasattr(obj, name):
        raise AttributeError(
            f"{type(obj).__name__} has no attribute '{name}'; the configuration layout changed. "
            "Update apply_smoke_overrides instead of writing a top-level attribute."
        )
    setattr(obj, name, value)


def apply_smoke_overrides(agent_cfg, smoke: bool, *, kind: str) -> None:
    """Shrink a run to an acceptance-sized workload.

    The values are chosen so the smoke run still exercises the real code path: a pretrain run
    collects enough transitions for at least one ``history_horizon + forecast`` window, and a
    finetune run performs a non-empty imagination rollout and a PPO update with no warmup.  The
    world-model values live in ``agent_cfg.algorithm`` because that is the dictionary the fork
    hands to ``MBPOPPO``; the rollout/imagination values stay on the runner config.
    """
    if not smoke:
        return
    if agent_cfg.max_iterations > 1:
        agent_cfg.max_iterations = 1
    if kind == "pretrain":
        # history_horizon (32) + forecast horizon (8) transitions are needed for one window, and the
        # evaluation needs a longer contiguous trajectory than the history horizon.
        agent_cfg.num_steps_per_env = max(agent_cfg.num_steps_per_env, 64)
        algorithm = algorithm_config(agent_cfg)
        _set_existing(algorithm, "system_dynamics_num_mini_batches", 2)
        _set_existing(algorithm, "system_dynamics_mini_batch_size", 64)
        _set_existing(algorithm, "system_dynamics_replay_buffer_size", 512)
        _set_existing(algorithm, "system_dynamics_num_eval_trajectories", 2)
        _set_existing(algorithm, "system_dynamics_len_eval_trajectory", 40)
    elif kind in ("finetune", "visualize"):
        # The imagination history is seeded from a real, contiguous window of ``history_horizon``
        # transitions, so the smoke rollout must collect at least that many steps before the first
        # (warmup-free) imagination rollout.
        agent_cfg.num_steps_per_env = max(agent_cfg.num_steps_per_env, 40)
        _set_existing(agent_cfg, "system_dynamics_warmup_iterations", 0)
        _set_existing(agent_cfg.imagination, "num_envs", 64)
        _set_existing(agent_cfg.imagination, "num_steps_per_env", 8)
        _set_existing(agent_cfg.imagination, "max_episode_length", 32)
        _set_existing(agent_cfg.imagination, "command_resample_interval_range", None)
        algorithm = algorithm_config(agent_cfg)
        _set_existing(algorithm, "system_dynamics_num_mini_batches", 2)
        _set_existing(algorithm, "system_dynamics_mini_batch_size", 64)
        _set_existing(algorithm, "system_dynamics_replay_buffer_size", 512)
        _set_existing(algorithm, "system_dynamics_num_eval_trajectories", 1)
        _set_existing(algorithm, "system_dynamics_len_eval_trajectory", 40)


@contextlib.contextmanager
def launched_runtime(env_cfg, args_cli):
    """Launch the simulator runtime for *env_cfg* and yield the resolved physics config."""
    with compat.launch(env_cfg, args_cli) as physics_cfg:
        report_backend(env_cfg)
        yield physics_cfg


@contextlib.contextmanager
def managed_env(task: str, env_cfg, agent_cfg):
    """Create the environment and wrap it, closing it before the runtime shuts down."""
    env = make_env(task, env_cfg)
    try:
        yield env, wrap_env(env, agent_cfg)
    finally:
        env.close()


def build_runner(vec_env, agent_cfg, log_dir: str | None):
    """Build the runner for *agent_cfg* using the vendored fork."""
    return create_runner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)


def load_training_checkpoint(runner, resume_path: str | None, agent_cfg, args_cli, *, resume: bool) -> None:
    """Load the training checkpoint for a *resume* or a *warm start*.

    Two cases are deliberately distinct:

    * **Resume** (``--resume True``): the run continues.  The policy, the world model and the
      iteration counter all come from the checkpoint of *this* run, so a pretrain resume restores
      the world-model weights instead of silently re-initializing them.
    * **Warm start** (``--checkpoint`` / ``--system_dynamics_load_path``): the weights come from
      another run.  The iteration counter restarts at zero and the world model is read from the
      artifact the configuration names.

    Optimizer state follows the checkpoint that supplied each weight and is only restored with
    ``--resume_optimizer``; a missing entry raises instead of being skipped.
    """
    from mbrl.rl.rsl_rl import load_checkpoint

    allow_partial = bool(getattr(args_cli, "allow_partial_checkpoint", False))
    restore_optimizer = bool(getattr(args_cli, "resume_optimizer", False))
    has_world_model = getattr(agent_cfg, "class_name", "") == "MBPOOnPolicyRunner"
    load_system_dynamics = has_world_model and (bool(agent_cfg.load_system_dynamics) or resume)

    if resume_path is None:
        if load_system_dynamics:
            dynamics_path = agent_cfg.system_dynamics_load_path
            if dynamics_path is None:
                raise FileNotFoundError(
                    "--system_dynamics_load_path is required when the task loads a world model "
                    "and no policy checkpoint is given."
                )
            load_checkpoint(
                runner,
                dynamics_path,
                load_policy=False,
                load_system_dynamics=True,
                load_optimizer=restore_optimizer,
                restore_iteration=False,
                map_location=runner.device,
                allow_partial=allow_partial,
            )
        return

    if resume:
        # Continuing this run: the world model must come from the same checkpoint as the policy.
        dynamics_source = resume_path if load_system_dynamics else None
        restored = "policy + world model + iteration counter" if load_system_dynamics else "policy + iteration counter"
        print(f"[INFO] Resuming the run from '{resume_path}' ({restored}).")
    else:
        # Warm start: the world model comes from the configured artifact, falling back to the given
        # checkpoint when it carries both heads (e.g. a pretrain ``model_1.pt``).
        dynamics_source = (agent_cfg.system_dynamics_load_path or resume_path) if load_system_dynamics else None
        print(f"[INFO] Warm start from '{resume_path}' (iteration counter restarts at 0).")
    load_checkpoint(
        runner,
        resume_path,
        load_policy=True,
        load_system_dynamics=load_system_dynamics,
        dynamics_path=dynamics_source,
        load_optimizer=restore_optimizer,
        restore_iteration=resume,
        map_location=runner.device,
        allow_partial=allow_partial,
    )


def bounded_loop(env, policy, args_cli, *, video_length: int | None = None) -> int:
    """Run the policy until ``--max_steps``/``--video_length`` is reached or the app closes."""
    import time

    app = _kit_app()
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    progress_every = 10
    while app is None or app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        timestep += 1
        if timestep % progress_every == 0:
            print(f"[INFO] step {timestep}/{args_cli.max_steps} | sim time {timestep * dt:.2f}s", flush=True)
        if video_length is not None and timestep >= video_length:
            break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        # keep the Kit viewport responsive in GUI runs
        if app is not None:
            app.update()
    print(f"[INFO] Reached step {timestep} (max_steps={args_cli.max_steps}).", flush=True)
    return timestep


def _kit_app():
    """Return the running Kit app, or ``None`` when the run does not use Kit."""
    try:
        import omni.kit.app

        return omni.kit.app.get_app()
    except (ImportError, ModuleNotFoundError):
        return None


def attach_video_recorder(env_cfg, output_dir: str, video_length: int) -> None:
    """Enable a Kit-viewport video recorder on the environment config."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    os.makedirs(output_dir, exist_ok=True)
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:kit",
            output_dir=output_dir,
            video_length=video_length,
            output_filename_prefix="clip",
        )
    ]
    print(f"[INFO] Recording Kit viewport video to: {output_dir}")
