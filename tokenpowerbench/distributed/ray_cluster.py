"""Resolve a Ray connection without importing Ray or starting cluster processes.

Resolution order is explicit options, RAY_HEAD_ADDRESS/RAY_ADDRESS, the first
SLURM allocation hostname, then Ray's existing-cluster discovery ("auto").
An explicit port overrides RAY_HEAD_PORT and an address's embedded port.
"""

from __future__ import annotations

import ipaddress
import os
import subprocess
from dataclasses import dataclass
from urllib.parse import urlsplit
import warnings


def _port(value):
    if isinstance(value, bool):
        raise ValueError("Ray port must be an integer between 1 and 65535")
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("Ray port must be an integer between 1 and 65535")
    return value


def _address_parts(address, default_port):
    if not isinstance(address, str) or not address.strip():
        raise ValueError("Ray address must be a nonempty string")
    address = address.strip()
    if address in ("auto", "local"):
        return "", address, default_port
    if any(character.isspace() for character in address):
        raise ValueError("Ray address must contain one hostname or IP address")
    if "://" in address:
        parsed = urlsplit(address)
        if (parsed.scheme != "ray" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment):
            raise ValueError("Ray Client address must have the form ray://host:port")
        return "ray", parsed.hostname, _port(10001 if parsed.port is None else parsed.port)
    if address.startswith("["):
        closing = address.find("]")
        if closing < 0:
            raise ValueError("Invalid bracketed IPv6 Ray address")
        host, suffix = address[1:closing], address[closing + 1:]
        ipaddress.IPv6Address(host)
        if suffix and not suffix.startswith(":"):
            raise ValueError("Invalid suffix after IPv6 Ray address")
        return "", host, _port(suffix[1:]) if suffix else default_port
    if address.count(":") > 1:
        ipaddress.IPv6Address(address)
        return "", address, default_port
    if ":" in address:
        host, port = address.rsplit(":", 1)
        if not host or any(character in host for character in "/?#@"):
            raise ValueError("Ray hostname must not be empty")
        return "", host, _port(port)
    if any(character in address for character in "/?#@"):
        raise ValueError("Invalid Ray hostname")
    return "", address, default_port


def _endpoint(scheme, host, port):
    if host in ("auto", "local"):
        return host
    host = f"[{host}]" if ":" in host else host
    return (scheme + "://" if scheme else "") + f"{host}:{port}"


@dataclass
class RayClusterConfig:
    """Connection settings; CPU/GPU/memory hints do not start or modify a cluster.

    Native host:port and ray:// client addresses retain their embedded ports.
    head_port supplies the default for bare hostnames and IP addresses.
    Use resolve() to apply explicit options and environment precedence.
    """

    head_address: str = "auto"
    head_port: int = 6379
    num_cpus: int | None = None
    num_gpus: int | None = None
    object_store_memory: int = 3_000_000_000

    def __post_init__(self):
        self.head_port = _port(self.head_port)
        scheme, host, port = _address_parts(self.head_address, self.head_port)
        self.head_address = _endpoint(scheme, host, port)
        self.head_port = port
        for name in ("num_cpus", "num_gpus"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        if (isinstance(self.object_store_memory, bool) or not isinstance(self.object_store_memory, int)
                or self.object_store_memory <= 0):
            raise ValueError("object_store_memory must be a positive integer")

    @property
    def ray_init_address(self):
        """Canonical address passed to ray.init(address=...)."""
        return self.head_address

    @property
    def ray_start_address(self):
        """Concrete GCS address for ray start --address on worker nodes."""
        if self.head_address in ("auto", "local") or self.head_address.startswith("ray://"):
            raise ValueError("Worker startup requires a concrete native Ray head host:port")
        return self.head_address

    @classmethod
    def resolve(cls, head_address=None, head_port=None):
        """Apply explicit > environment > SLURM > existing-cluster precedence."""
        address = head_address
        if address is None:
            address = (os.environ.get("RAY_HEAD_ADDRESS", "").strip()
                       or os.environ.get("RAY_ADDRESS", "").strip() or None)
        if address is None:
            address = cls._slurm_head() or "auto"
        scheme, host, embedded = _address_parts(address, None)
        if head_port is not None:
            port = _port(head_port)
        elif embedded is not None:
            port = embedded
        elif host in ("auto", "local"):
            port = 6379
        else:
            port = _port(os.environ.get("RAY_HEAD_PORT", "6379"))
        return cls(head_address=_endpoint(scheme, host, port), head_port=port)

    @classmethod
    def from_env(cls):
        """Read the Ray address and port environment without probing SLURM."""
        address = (os.environ.get("RAY_HEAD_ADDRESS", "").strip()
                   or os.environ.get("RAY_ADDRESS", "").strip() or "auto")
        return cls.resolve(head_address=address)

    @classmethod
    def from_slurm(cls):
        """Resolve environment overrides first, then the first SLURM hostname."""
        return cls.resolve()

    @staticmethod
    def _slurm_head():
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "").strip()
        if not nodelist:
            return None
        try:
            nodes = subprocess.check_output(
                ["scontrol", "show", "hostnames", nodelist], text=True,
                stderr=subprocess.DEVNULL, timeout=5,
            ).splitlines()
            if not nodes or not nodes[0].strip():
                raise ValueError("SLURM returned no hostnames")
            # Ray accepts hostnames. DNS resolution stays with Ray rather than
            # forcing getent's first (possibly IPv6 or unrelated-interface) IP.
            return nodes[0].strip()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            warnings.warn(f"Cannot resolve the SLURM Ray head ({exc}); using Ray discovery", RuntimeWarning)
            return None
