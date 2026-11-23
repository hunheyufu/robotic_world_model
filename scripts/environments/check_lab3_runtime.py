"""Runtime pre-flight check for the Isaac Lab 3.x / Isaac Sim 6.2 migration.

Without ``--launch`` this only prints the interpreter, package versions, the detected task layout
and the resolved task configs; it does not start the simulator.  With ``--launch`` it also creates
the official ANYmal-D flat environment once, resets it, applies one zero action and reports the
observation/action dimensions and the active physics backend.

Reproducible usage (from the repository root)::

    C:\\IsaacLab\\.venv\\Scripts\\python.exe scripts/environments/check_lab3_runtime.py
    C:\\IsaacLab\\.venv\\Scripts\\python.exe scripts/environments/check_lab3_runtime.py --launch --viz kit

The host virtual environment is used read-only: ``source/mbrl`` is added to ``sys.path`` instead of
being installed into it.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "mbrl"))
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

parser = argparse.ArgumentParser(description="Check the Lab 3.x runtime for the RWM migration.")
parser.add_argument("--launch", action="store_true", default=False, help="Also launch Kit and step an env once.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-AnymalD")
parser.add_argument("--physics", type=str, default="isaacsim_physx")
args_cli, _ = parser.parse_known_args()

from mbrl import compat  # noqa: E402

print("[check] interpreter :", sys.executable)
print("[check] version     :", sys.version.split()[0])
print("[check] task layout :", compat.lab_generation())
for key, value in compat.package_versions().items():
    print(f"[check] {key:<12}: {value}")

try:
    import warp as wp

    print("[check] warp        :", wp.__version__, "| CUDA devices:", wp.get_cuda_device_count())
except Exception as exc:  # noqa: BLE001 - diagnostics only
    print("[check] warp        : unavailable", exc)

os.environ.setdefault("WARP_LOG_LEVEL", "WARN")

import mbrl.tasks  # noqa: E402,F401  (registers the RWM tasks)

for task in (
    "Template-Isaac-Velocity-Flat-Anymal-D-Init-v0",
    "Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0",
    "Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0",
    "Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0",
):
    env_cfg, agent_cfg = compat.resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    groups = [name for name in vars(env_cfg.observations) if not name.startswith("_")]
    print(f"[check] {task.split(':')[-1]:<52} runner={agent_cfg.class_name:<20} obs={groups}")

if not args_cli.launch:
    print("[check] configuration resolved; pass --launch to start Kit")
    raise SystemExit(0)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser2 = argparse.ArgumentParser()
add_launcher_args(parser2)
launcher_args, _ = parser2.parse_known_args()
launcher_args.physics = args_cli.physics
launcher_args.headless = getattr(launcher_args, "headless", False) or not getattr(launcher_args, "visualizer", None)

env_cfg, _ = compat.resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point", overrides=[])
env_cfg.scene.num_envs = 1
with launch_simulation(env_cfg, launcher_args):
    physics = type(env_cfg.sim.physics).__name__
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        unwrapped = env.unwrapped
        for name, value in obs.items():
            print(f"[check] obs group {name:<16}: {tuple(value.shape)} {value.dtype}")
        actions = torch.zeros(unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device)
        obs, rewards, terminated, truncated, extras = env.step(actions)
        print(f"[check] step ok: reward={rewards.tolist()} terminated={terminated.tolist()} truncated={truncated.tolist()}")
        print(f"[check] physics: {physics} | step_dt={unwrapped.step_dt} | num_actions={unwrapped.action_manager.total_action_dim}")
        print(f"[check] joints : {unwrapped.scene['robot'].joint_names}")
        print("[check] launch OK")
    finally:
        env.close()
