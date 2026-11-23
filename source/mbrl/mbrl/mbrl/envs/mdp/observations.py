# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from mbrl.compat import contact_normal_force_history

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def body_contact(env: ManagerBasedEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact status of the body of the asset, based on the *normal* contact force.

    The RWM system-dynamics labels are defined on ``max_history(norm(force)) > threshold`` for the
    normal force.  Isaac Lab 3.x / Sim 6.2 changed ``net_forces_w_history`` to the total force
    (normal + tangential), so the normal history is selected explicitly; Lab 2.x is unchanged.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_normal_forces_history = contact_normal_force_history(contact_sensor)
    is_contact = (
        torch.max(torch.norm(net_normal_forces_history[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    return is_contact
