"""Package containing task implementations for various robotic environments."""

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages.
# ``envs`` holds the custom environment classes, which pull in USD/Kit modules; they are referenced
# by Gym entry points as strings and imported when the environment is created.
_BLACKLIST_PKGS = ["utils", "envs"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
