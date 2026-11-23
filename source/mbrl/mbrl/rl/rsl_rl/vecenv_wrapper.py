# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter between Isaac Lab 3.x vectorized environments and the RWM ``rsl_rl`` fork.

:class:`MbrlVecEnvWrapper` is a thin adapter written against the fork's own ``VecEnv`` contract.  It
deliberately does not inherit Isaac Lab's wrapper, which targets the host's (different) ``rsl_rl``
release.  The original semantics are preserved: action clipping, ``dones = terminated | truncated``
and ``extras["time_outs"]`` for infinite-horizon tasks.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mbrl.rl.rsl_rl_fork.env import VecEnv


class MbrlVecEnvWrapper(VecEnv):
    """Expose an Isaac Lab 3.x manager-based environment as a fork-compatible ``VecEnv``."""

    def __init__(self, env, clip_actions: float | None = None):
        """Wrap *env*.

        Args:
            env: Isaac Lab manager-based environment (gymnasium ``Env``).
            clip_actions: Optional symmetric action clip applied before stepping.
        """
        self.env = env
        self.clip_actions = clip_actions

        unwrapped = env.unwrapped
        self.num_envs = unwrapped.num_envs
        self.device = unwrapped.device
        self.max_episode_length = unwrapped.max_episode_length
        self.num_actions = unwrapped.action_manager.total_action_dim

        # the runner never calls ``reset`` before the first rollout
        env.reset()

    def __str__(self) -> str:  # noqa: D105
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self) -> str:  # noqa: D105
        return str(self)

    @property
    def cfg(self):
        """Environment configuration object."""
        return self.unwrapped.cfg

    @property
    def unwrapped(self):
        """The bare Isaac Lab environment underneath all layers of wrappers."""
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Current episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    def _to_tensordict(self, obs_dict) -> TensorDict:
        """Wrap a mapping of observation groups into a batched ``TensorDict`` without copying."""
        if isinstance(obs_dict, TensorDict):
            return obs_dict
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        obs_dict, extras = self.env.reset()
        return self._to_tensordict(obs_dict), extras

    def get_observations(self) -> TensorDict:  # noqa: D102
        return self._to_tensordict(self.unwrapped.obs_buf)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # infinite-horizon tasks rely on ``time_outs`` to bootstrap through the time limit
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self._to_tensordict(obs_dict), rewards, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()
