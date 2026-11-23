# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Run from a checkout without installing the extension into the host environment: the source tree
# is put on ``sys.path`` before the first ``mbrl`` import. The host venv is never modified.
_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "source", "mbrl"))
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from isaaclab.app import add_launcher_args

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint with an RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
cli_args.add_mbrl_args(parser)
add_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video; recordings come from the Kit viewport
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.visualizer = "kit"

# clear out sys.argv for the config resolver
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""

import torch

import isaaclab_tasks  # noqa: F401  (registers the host tasks)

# PLACEHOLDER: Extension template (do not remove this comment)
import mbrl.tasks  # noqa: F401  (registers the RWM tasks)
from mbrl.rl.rsl_rl import RslRlRwmRunnerCfg, create_runner, load_checkpoint
from mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg import (
    AnymalDFlatPPOInitRunnerCfg,
)

import lab3_runtime as runtime


DEFAULT_TASK = "Isaac-Velocity-Flat-AnymalD"


def _play_agent_config(agent_cfg):
    """Return the runner configuration that matches the checkpoint's policy ABI.

    Project tasks register a ``RslRlRwmRunnerCfg`` and it is used unchanged, so ``--load_run``,
    ``obs_groups`` and the policy layout come from the task.  Official Isaac Lab 3.x tasks register
    the host runner config (separate ``actor``/``critic`` models), which the pre-migration
    checkpoints cannot be loaded into; only then is the project's legacy policy configuration used.
    """
    if isinstance(agent_cfg, RslRlRwmRunnerCfg):
        return agent_cfg
    print(
        f"[WARN] Task '{args_cli.task}' registers {type(agent_cfg).__name__}, which uses Isaac Lab 3.x's "
        "actor/critic policy ABI. Playing a pre-migration checkpoint with the project's ActorCritic "
        "runner configuration instead."
    )
    return AnymalDFlatPPOInitRunnerCfg()


def _run() -> int:
    """Play a checkpoint on Isaac Lab 3.x / Isaac Sim 6.2 with a bounded loop."""
    from mbrl import compat

    runtime.report_runtime()
    task = args_cli.task or DEFAULT_TASK
    args_cli.task = task
    # ``play_mode`` applies the official play overrides (no push events, smaller terrain grid)
    env_cfg, agent_cfg = compat.resolve_task_config(
        task, "rsl_rl_cfg_entry_point", play_mode=True, overrides=hydra_args
    )
    agent_cfg = _play_agent_config(agent_cfg)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    runtime.apply_env_overrides(args_cli, env_cfg)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = runtime.resolve_play_checkpoint(args_cli, agent_cfg, log_root)
    if resume_path is None:
        return 0
    log_dir = os.path.dirname(resume_path)
    if args_cli.video:
        runtime.attach_video_recorder(env_cfg, os.path.join(log_dir, "videos", "play"), args_cli.video_length)

    steps = 0
    with runtime.launched_runtime(env_cfg, args_cli):
        env = runtime.make_env(task, env_cfg)
        try:
            vec_env = runtime.wrap_env(env, agent_cfg)
            runner = create_runner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            load_checkpoint(
                runner,
                resume_path,
                load_policy=True,
                load_system_dynamics=False,
                map_location=agent_cfg.device,
                allow_partial=args_cli.allow_partial_checkpoint,
            )
            policy = runner.get_inference_policy(device=vec_env.device)
            steps = runtime.bounded_loop(
                vec_env,
                policy,
                args_cli,
                video_length=args_cli.video_length if args_cli.video else None,
            )
        finally:
            env.close()
    print(f"[INFO] Play finished after {steps} environment steps.")
    return 0


def main() -> int:
    """Play on Isaac Lab 3.x."""
    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
