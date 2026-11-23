"""
Python module serving as a project/extension template.

Importing this package must stay free of side effects: the Isaac Lab 3.x entry points import it
*before* the simulator is launched, and any Kit/USD module imported at that point prevents
``SimulationApp`` from initialising correctly.  Tasks are registered by importing ``mbrl.tasks``
(which every entry point does explicitly), and the Kit UI example is registered as its own
extension module (see ``config/extension.toml``).
"""


def register_tasks() -> None:
    """Register the project's Gym environments (equivalent to ``import mbrl.tasks``)."""
    import mbrl.tasks  # noqa: F401
