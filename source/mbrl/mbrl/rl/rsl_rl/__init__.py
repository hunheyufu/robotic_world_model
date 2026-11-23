# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner configuration, environment adapter and checkpoint handling for the RWM ``rsl_rl`` fork.

The following example shows how to wrap an Isaac Lab 3.x environment for the fork:

.. code-block:: python

    from mbrl.rl.rsl_rl import MbrlVecEnvWrapper

    env = MbrlVecEnvWrapper(env)

"""

from .rl_cfg import *
from .vecenv_wrapper import MbrlVecEnvWrapper
from .runner import create_runner, load_checkpoint, describe_checkpoint
