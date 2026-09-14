"""Process identity metadata, independent of power-sensor permissions.

Root identity does not establish that a sensor exists or can be read. Likewise,
administrators can grant a non-root process access to individual sensors. Record
this metadata alongside the monitor's actual capability probes.
"""

import os
import platform


def _process_id(name):
    """Read one Unix process ID when the operating system supports it."""
    getter = getattr(os, name, None)
    if not callable(getter):
        return None
    try:
        return getter()
    except (OSError, NotImplementedError):
        return None


def runtime_identity():
    """Return JSON-serializable process IDs and platform metadata.

    ``uid`` and ``gid`` are real IDs; ``euid`` and ``egid`` are effective IDs.
    ``is_root`` refers to effective UID 0 in the process's own environment, and
    is ``None`` when effective IDs are unsupported. It never implies host-level
    sensor access, including when the process runs in a container.
    """
    identity = {name: _process_id(f"get{name}") for name in ("uid", "euid", "gid", "egid")}
    identity["is_root"] = None if identity["euid"] is None else identity["euid"] == 0
    identity["platform"] = platform.system()
    identity["machine_architecture"] = platform.machine()
    return identity
