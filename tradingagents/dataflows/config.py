from copy import deepcopy
from typing import Dict, Optional

import tradingagents.default_config as default_config

# Use default config but allow it to be overridden
_config: Optional[Dict] = None


def initialize_config():
    """Initialize the configuration with default values.

    The stored config is a deep copy of ``DEFAULT_CONFIG``. Nested maps such
    as ``data_vendors`` and ``tool_vendors`` are independent objects, so
    editing ``DEFAULT_CONFIG`` or the stored config leaves the other as it was.
    """
    global _config
    if _config is None:
        _config = deepcopy(default_config.DEFAULT_CONFIG)


def set_config(config: Dict):
    """Update the configuration with custom values.

    Nested maps on ``config`` are copied before they are stored. Editing the
    passed-in dict, ``DEFAULT_CONFIG``, or the stored config leaves the others
    as they were. Keys left out of ``config`` keep their current values.
    """
    global _config
    if _config is None:
        _config = deepcopy(default_config.DEFAULT_CONFIG)
    _config.update(deepcopy(config))


def get_config() -> Dict:
    """Get the current configuration.

    The result is a deep copy. Nested maps such as ``data_vendors`` and
    ``tool_vendors`` are not the live objects, so editing them does not
    change later lookups.
    """
    if _config is None:
        initialize_config()
    return deepcopy(_config)


# Initialize with default config
initialize_config()
