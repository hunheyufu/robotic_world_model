# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize real and imagined rollouts of an RSL-RL agent."""

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
parser = argparse.ArgumentParser(description="Visualize real and imagined ANYmal-D instances side by side.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Total number of instances (real + imagined). Must be even; the 3.x runtime rejects odd values.",
)
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

import isaaclab_tasks  # noqa: F401  (registers the host tasks)

# PLACEHOLDER: Extension template (do not remove this comment)
import mbrl.tasks  # noqa: F401  (registers the RWM tasks)
from mbrl.rl.rsl_rl import create_runner, load_checkpoint

import lab3_runtime as runtime


DEFAULT_TASK = "Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0"


def _run() -> int:
    """Visualize paired real/imagined instances on Isaac Lab 3.x / Isaac Sim 6.2."""
    from mbrl import compat

    runtime.report_runtime()
    task = args_cli.task or DEFAULT_TASK
    args_cli.task = task
    if args_cli.num_envs is not None and args_cli.num_envs % 2 != 0:
        raise ValueError(
            f"--num_envs is the *total* number of instances and must be even (got {args_cli.num_envs}): "
            "even indices are real instances, odd indices are their imagination partners."
        )

    env_cfg, agent_cfg = compat.resolve_task_config(
        task, "rsl_rl_cfg_entry_point", overrides=hydra_args
    )
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    runtime.apply_env_overrides(args_cli, env_cfg)
    if env_cfg.scene.num_envs % 2 != 0:
        raise ValueError(f"The visualize scene needs an even instance count, got {env_cfg.scene.num_envs}.")
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    runtime.apply_smoke_overrides(agent_cfg, args_cli.smoke, kind="visualize")
    # validate both artifacts before the simulator starts
    runtime.resolve_dynamics_path(args_cli, agent_cfg)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = runtime.resolve_checkpoint(args_cli, agent_cfg, log_root, required=True)
    log_dir = os.path.dirname(resume_path)
    if args_cli.video:
        runtime.attach_video_recorder(env_cfg, os.path.join(log_dir, "videos", "visualize"), args_cli.video_length)

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
                load_system_dynamics=True,
                map_location=agent_cfg.device,
                allow_partial=args_cli.allow_partial_checkpoint,
            )
            policy = runner.get_inference_policy(device=vec_env.device)
            vec_env.unwrapped.init_imagination_history(agent_cfg.system_dynamics.history_horizon)
            steps = runtime.bounded_loop(
                vec_env,
                policy,
                args_cli,
                video_length=args_cli.video_length if args_cli.video else None,
            )
        finally:
            env.close()
    print(f"[INFO] Visualize finished after {steps} environment steps.")
    return 0


def main() -> int:
    """Visualize on Isaac Lab 3.x."""
    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
