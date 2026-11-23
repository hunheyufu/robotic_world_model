from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_mbrl_args(parser: argparse.ArgumentParser):
    """Add the project arguments shared by the train/play/visualize entry points.

    Args:
        parser: The parser to add the arguments to.
    """
    arg_group = parser.add_argument_group("world_model", description="Arguments for the RWM entries.")
    arg_group.add_argument(
        "--system_dynamics_load_path", type=str, default=None, help="Dynamics model load path."
    )
    arg_group.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop the play/visualize loop after this many environment steps (bounded runs).",
    )
    arg_group.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Small-scale acceptance run: shorter rollouts/imagination and no long warmup.",
    )
    arg_group.add_argument(
        "--log_root",
        type=str,
        default=None,
        help="Root directory for new runs (defaults to logs/rsl_rl/<experiment_name>).",
    )
    arg_group.add_argument(
        "--run_timestamp",
        type=str,
        default=None,
        help="Run directory name (defaults to the current timestamp).",
    )
    arg_group.add_argument(
        "--allow_partial_checkpoint",
        action="store_true",
        default=False,
        help="Load checkpoints with strict=False. Only for artifacts verified by hand.",
    )
    arg_group.add_argument(
        "--resume_optimizer",
        action="store_true",
        default=False,
        help="Restore optimizer state as well. Cross-runtime resume is not guaranteed.",
    )
    arg_group.add_argument(
        "--physics",
        type=str,
        default="isaacsim_physx",
        choices=["physx", "isaacsim_physx", "newton_mjwarp", "newton_vbd", "ovphysx"],
        help=(
            "Physics backend for the Isaac Lab 3.x launcher. The default keeps the Isaac Sim "
            "PhysX + Kit backend."
        ),
    )
    arg_group.add_argument("--headless", action="store_true", default=False, help="Run without a Kit viewport.")


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
