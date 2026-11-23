"""Print the environments registered by this extension.

The script iterates over the Gym registry and prints the task name, entry point and config for each
of the project's tasks.  It does **not** launch Kit: task registration only imports the config
modules, and the environment classes are imported lazily when an environment is created.

A missing dependency (``gymnasium``, ``prettytable`` or Isaac Lab) is reported with an actionable
error instead of a traceback.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "mbrl"))
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--keyword",
        type=str,
        default="Template-",
        help="Only list task ids containing this substring (default: the project's 'Template-' tasks).",
    )
    parser.add_argument("--all", action="store_true", help="List every registered task, not only --keyword matches.")
    return parser.parse_args()


def main() -> int:
    """Print all environments registered by the extension."""
    args = _parse_args()
    try:
        import gymnasium as gym
        from prettytable import PrettyTable
    except ImportError as exc:
        raise SystemExit(
            f"[ERROR] list_envs needs gymnasium and prettytable ({exc}). "
            "Run it with the Python environment where Isaac Lab is installed."
        ) from exc

    try:
        import isaaclab_tasks  # noqa: F401  (registers the host tasks)
        import mbrl.tasks  # noqa: F401  (registers the RWM tasks)
    except ImportError as exc:
        raise SystemExit(
            f"[ERROR] list_envs needs Isaac Lab and the mbrl sources ({exc}). "
            "Install Isaac Lab 3.x or run from a checkout with the extension on sys.path."
        ) from exc

    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in the RWM Extension"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "env_cfg_entry_point" not in task_spec.kwargs:
            continue
        if not args.all and args.keyword not in task_spec.id:
            continue
        table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
        index += 1

    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
