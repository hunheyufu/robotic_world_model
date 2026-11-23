# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlActorCriticCfg:
    """Actor-critic configuration in the layout expected by the RWM ``rsl_rl`` fork.

    The pinned fork (``rsl_rl`` 3.1.0, commit ``18eebcdd``) instantiates
    ``ActorCritic(obs, obs_groups, num_actions, **cfg)`` from this dictionary.  Isaac Lab 3.x ships
    a different runner configuration (separate ``actor``/``critic`` models), so the RWM runner
    config owns this class instead of inheriting the host's.
    """

    class_name: str = "ActorCritic"
    """Policy class name resolved inside the runner module."""

    init_noise_std: float = 1.0
    """Initial action noise standard deviation."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """Whether the noise standard deviation is a scalar or its logarithm."""

    state_dependent_std: bool = False
    """Whether the standard deviation is conditioned on the observation."""

    actor_hidden_dims: list[int] = MISSING
    """Actor hidden layer sizes."""

    critic_hidden_dims: list[int] = MISSING
    """Critic hidden layer sizes."""

    activation: str = "elu"
    """Activation function name."""

    actor_obs_normalization: bool = False
    """Whether the actor normalizes its observations."""

    critic_obs_normalization: bool = False
    """Whether the critic normalizes its observations."""


@configclass
class RslRlRwmPpoAlgorithmCfg:
    """PPO configuration in the layout expected by the RWM ``rsl_rl`` fork."""

    class_name: str = "PPO"
    """Algorithm class name resolved inside the runner module."""

    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.005
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False
    symmetry_cfg: dict | None = None
    rnd_cfg: dict | None = None


@configclass
class RslRlSystemDynamicsCfg:
    """Configuration for the system dynamics networks."""
    
    ensemble_size: int = MISSING
    """The ensemble size of the system dynamics network."""

    history_horizon: int = MISSING
    """The prediction horizon of the system dynamics network."""

    architecture_config: dict = MISSING
    """The architecture configuration of the system dynamics network."""
    
    freeze_auxiliary: bool = MISSING
    """Whether to freeze the auxiliary networks."""


@configclass
class RslRlNormalizerCfg:
    """Configuration for the normalizer."""

    mean: list[float] = MISSING
    """The mean of the normalizer."""

    std: list[float] = MISSING
    """The std of the normalizer."""


@configclass
class RslRlMbrlImaginationCfg:
    """Configuration for the imagination."""
    
    num_envs: int = MISSING
    """The number of environments for the imagination."""
    
    num_steps_per_env: int = MISSING
    """The number of steps for the imagination."""
    
    max_episode_length: int = MISSING
    """The maximum episode length for the imagination."""

    command_resample_interval_range: list[float] | None = MISSING
    """The resample interval range for the command."""
    
    uncertainty_penalty_weight: float = MISSING
    """The weight for the uncertainty penalty."""
    
    state_normalizer: RslRlNormalizerCfg = MISSING
    """The normalizer for the state."""
    
    action_normalizer: RslRlNormalizerCfg = MISSING
    """The normalizer for the action."""


@configclass
class RslRlMbrlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "MBPOPPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    policy_learning_rate: float = MISSING
    """The learning rate for the policy."""

    system_dynamics_learning_rate: float = MISSING
    """The learning rate for the system dynamics."""
    
    system_dynamics_weight_decay: float = MISSING
    """The weight decay for the system dynamics."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    
    system_dynamics_forecast_horizon: int = MISSING
    """The forecast horizon for the system dynamics."""
    
    system_dynamics_loss_weights: dict[str, float] = MISSING
    """The loss weights for the system dynamics."""
    
    system_dynamics_num_mini_batches: int = MISSING
    """The number of mini-batches for the system dynamics."""
    
    system_dynamics_mini_batch_size: int = MISSING
    """The mini-batch size for the system dynamics."""
    
    system_dynamics_replay_buffer_size: int = MISSING
    """The replay buffer size for the system dynamics."""
    
    system_dynamics_num_eval_trajectories: int = MISSING
    """The number of evaluation trajectories for the system dynamics."""
    
    system_dynamics_len_eval_trajectory: int = MISSING
    """The length of the evaluation trajectory for the system dynamics."""
    
    system_dynamics_eval_traj_noise_scale: list[float] = MISSING
    """The noise scale for the evaluation trajectory for the system dynamics."""


@configclass
class RslRlRwmRunnerCfg:
    """Runner configuration for the pinned RWM ``rsl_rl`` fork.

    This is an explicit copy of the pre-migration runner configuration.  It is intentionally *not*
    derived from Isaac Lab's runner config: Isaac Lab 3.x replaced the single ``policy`` entry with
    separate ``actor``/``critic`` model configs, renamed the observation sets to
    ``actor``/``critic`` and added algorithm keys the fork does not accept.  The values below match
    the checkpoints produced by the Isaac Lab 2.2.1 baseline
    (``logs/rsl_rl/anymal_d_flat/2026-11-23_19-29-43``).
    """

    class_name: str = "OnPolicyRunner"
    """Runner class name resolved by the project scripts."""

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 300
    save_interval: int = 50
    experiment_name: str = "anymal_d_flat"
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = None
    empirical_normalization: bool = False
    init_at_random_ep_len: bool = True

    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["policy"]}
    """Observation sets used by the fork: ``policy`` feeds the actor, ``critic`` the critic."""

    policy: RslRlActorCriticCfg = MISSING
    """Actor-critic configuration."""

    algorithm: RslRlRwmPpoAlgorithmCfg = MISSING
    """PPO algorithm configuration."""

    # -- RWM specific extensions (only used by ``MBPOOnPolicyRunner``)
    system_dynamics: RslRlSystemDynamicsCfg = MISSING
    imagination: RslRlMbrlImaginationCfg = MISSING
    load_system_dynamics: bool = False
    system_dynamics_load_path: str | None = None
    system_dynamics_warmup_iterations: int = 0
    system_dynamics_num_visualizations: int = 4
    system_dynamics_state_idx_dict: dict = MISSING
    pca_obs_buf_size: int = 10000
