# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import torch

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils import index_fill_

from .manager_based_mbrl_env import ManagerBasedMBRLEnv, ManagerBasedRLEnvCfg


class ManagerBasedVisualizeEnv(ManagerBasedMBRLEnv):
    """Visualize environment with paired real/imagination instances.

    Even indices are real instances, odd indices are their imagination partners.  The class keeps
    the shared imagination-history helpers and mirrors the Isaac Lab 3.x ``ManagerBasedRLEnv.step``
    lifecycle, adding exactly three things:

    * the imagination write is performed before the last physics sub-step,
    * resets are paired (a real instance and its imagination partner reset together),
    * the imagination histories are re-synchronised from the real instance after the observation
      update.

    Everything else (recorder lifecycle, physics-owned decimation, render enabling, UI reset
    requests) follows the parent class verbatim so the 3.x lifecycle is preserved.
    """
    
    
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.env_ids_real = torch.arange(0, self.num_envs, 2, device=self.device)
        self.env_ids_imagination = torch.arange(1, self.num_envs, 2, device=self.device)

        
    def init_imagination_history(self, history_horizon):
        self.imagination_state_history = torch.zeros(self.num_envs // 2, history_horizon, self.observation_manager.group_obs_dim["system_state"][0], device=self.device)
        self.imagination_action_history = torch.zeros(self.num_envs // 2, history_horizon, self.observation_manager.group_obs_dim["system_action"][0], device=self.device)
        
    
    def _sync_imagination_history(self, env_ids_real):
        self.imagination_state_history[env_ids_real // 2] = 0.0
        self.imagination_action_history[env_ids_real // 2] = 0.0
        self.imagination_state_history[env_ids_real // 2, -1] = self.imagination_state_normalizer(self.observation_manager.compute()["system_state"])[env_ids_real]


    def _update_imagination_envs(self, action):
        self.num_imagination_envs = len(self.env_ids_imagination)
        rollout_action = action[self.env_ids_imagination]
        self.imagination_action_history = torch.cat([self.imagination_action_history[:, 1:].clone(), self.imagination_action_normalizer(rollout_action).unsqueeze(1)], dim=1)
        if self.system_dynamics.architecture_config["type"] in ["rnn", "rssm"]:
            self.imagination_state_history = self.imagination_state_history[:, -1].unsqueeze(1)
            self.imagination_action_history = self.imagination_action_history[:, -1].unsqueeze(1)
        imagination_states, *_ = self.system_dynamics.forward(self.imagination_state_history, self.imagination_action_history)
        imagination_states_denormalized = self.imagination_state_normalizer.inverse(imagination_states)
        parsed_imagination_states = self._parse_imagination_states(imagination_states_denormalized)
        self._reset_imagination_sim(parsed_imagination_states)
        self.imagination_state_history = torch.cat([self.imagination_state_history[:, 1:].clone(), imagination_states.unsqueeze(1)], dim=1)


    def _reset_imagination_sim(self, parsed_imagination_states):
        raise NotImplementedError


    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the paired real/imagination environment.

        Mirrors the Isaac Lab 3.x ``ManagerBasedRLEnv.step`` lifecycle and inserts the imagination
        write before the last physics sub-step, the paired reset and the history synchronisation.
        """
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.is_rendering
        steps_per_call = self.cfg.decimation if self._physics_handles_decimation else 1
        num_calls = self.cfg.decimation // steps_per_call
        for call_index in range(num_calls):
            self._sim_step_counter += steps_per_call
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            if call_index == num_calls - 1:
                # write the predicted state of the odd instances before the last physics step
                self._update_imagination_envs(action)
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt * steps_per_call)

        # post-step: counters, terminations, rewards
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # paired reset: a real instance resets together with the imagination instance next to it
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        env_ids_real = torch.empty(0, dtype=torch.long, device=self.device)
        if len(reset_env_ids) > 0:
            uniques, counts = torch.cat([reset_env_ids, self.env_ids_real]).unique(return_counts=True)
            env_ids_real = uniques[counts > 1]
            env_ids = torch.vstack([env_ids_real, env_ids_real + 1]).T.flatten()
            if len(env_ids) > 0:
                self.recorder_manager.record_pre_reset(env_ids)
                self._reset_idx(env_ids)
                self.recorder_manager.record_post_reset(env_ids)
            if (
                self.render_enabled
                and is_rendering
                and self.has_rtx_sensors
                and self.cfg.num_rerenders_on_reset > 0
            ):
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # handle episode reset requested from visualizer UI controls
        if self.sim.consume_reset_request():
            not_yet_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            index_fill_(not_yet_reset, reset_env_ids, False)
            manual_reset_ids = not_yet_reset.nonzero(as_tuple=False).squeeze(-1).int()
            if len(manual_reset_ids) > 0:
                index_fill_(self.reset_terminated, manual_reset_ids, True)
                self.recorder_manager.record_pre_reset(manual_reset_ids)
                self._reset_idx(manual_reset_ids)
                self.recorder_manager.record_post_reset(manual_reset_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.active_terms:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        for recorder in self.video_recorders:
            recorder.step()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if len(env_ids_real) > 0:
            self._sync_imagination_history(env_ids_real)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
