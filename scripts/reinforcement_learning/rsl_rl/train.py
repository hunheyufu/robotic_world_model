# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL and project cli arguments
cli_args.add_rsl_rl_args(parser)
cli_args.add_mbrl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video; recordings come from the Kit viewport
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.visualizer = "kit"

# clear out sys.argv for the config resolver
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""

import torch

import lab3_runtime as runtime

import isaaclab_tasks  # noqa: F401  (registers the host tasks)
# PLACEHOLDER: Extension template (do not remove this comment)
import mbrl.tasks  # noqa: F401  (registers the RWM tasks)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _run() -> int:
    """Train with the RWM fork on Isaac Lab 3.x / Isaac Sim 6.2."""
    from mbrl import compat

    runtime.report_runtime()
    env_cfg, agent_cfg = compat.resolve_task_config(
        args_cli.task, "rsl_rl_cfg_entry_point", overrides=hydra_args
    )
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    runtime.apply_env_overrides(args_cli, env_cfg)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    task_name = args_cli.task.split(":")[-1]
    if "Pretrain" in task_name:
        smoke_kind = "pretrain"
    elif "Finetune" in task_name:
        smoke_kind = "finetune"
    else:
        smoke_kind = "init"
    runtime.apply_smoke_overrides(agent_cfg, args_cli.smoke, kind=smoke_kind)
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    # validate the dynamics artifact before the simulator starts
    runtime.resolve_dynamics_path(args_cli, agent_cfg)

    log_dir = runtime.make_log_dir(agent_cfg, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume = bool(agent_cfg.resume)
    resume_path = runtime.resolve_checkpoint(args_cli, agent_cfg, log_root, required=resume)
    if args_cli.video:
        runtime.attach_video_recorder(env_cfg, os.path.join(log_dir, "videos", "train"), args_cli.video_length)

    with runtime.launched_runtime(env_cfg, args_cli):
        env = runtime.make_env(args_cli.task, env_cfg)
        try:
            vec_env = runtime.wrap_env(env, agent_cfg)
            runner = runtime.build_runner(vec_env, agent_cfg, log_dir)
            runner.add_git_repo_to_log(__file__)
            runtime.load_training_checkpoint(runner, resume_path, agent_cfg, args_cli, resume=resume)
            runtime.dump_configs(log_dir, env_cfg, agent_cfg)
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
            )
        finally:
            env.close()

    print(f"[INFO] Run directory: {log_dir}")
    return 0


def main() -> int:
    """Train on Isaac Lab 3.x."""
    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
