"""
Ray cluster configuration — no hardcoded IPs.

The head node address is resolved via (in priority order):
  1. Explicit constructor argument
  2. RAY_HEAD_ADDRESS environment variable
  3. SLURM_JOB_NODELIST — first node in the list is the head (auto-detected
     from SLURM environment when running inside a SLURM job)
  4. "auto" — Ray's built-in service-discovery (works when ray start --head
     was already called on the same machine or cluster)

SLURM scripts should export RAY_HEAD_ADDRESS before calling Python:
    export RAY_HEAD_ADDRESS=$(hostname --ip-address)
    # or let RayClusterConfig.from_slurm() do it automatically.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RayClusterConfig:
    """
    Parameters for connecting to (or starting) a Ray cluster.

    Attributes
    ----------
    head_address : str
        IP or hostname of the Ray head node, or "auto" for service discovery.
    head_port : int
        Port Ray is listening on (default 6379).
    num_cpus : int, optional
        Number of CPUs to advertise when starting the head. None = detect automatically.
    num_gpus : int, optional
        Number of GPUs to advertise when starting the head. None = detect automatically.
    object_store_memory : int
        Object store memory in bytes (default 3 GB).
    """

    head_address: str = "auto"
    head_port: int = 6379
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    object_store_memory: int = 3_000_000_000  # 3 GB

    @property
    def ray_init_address(self) -> str:
        """Address string for ray.init(address=...)."""
        if self.head_address == "auto":
            return "auto"
        return f"{self.head_address}:{self.head_port}"

    @property
    def ray_start_address(self) -> str:
        """Address string for ``ray start --address=...`` on worker nodes."""
        return f"{self.head_address}:{self.head_port}"

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> RayClusterConfig:
        """
        Read cluster config from environment variables.

        Variables
        ---------
        RAY_HEAD_ADDRESS   IP or hostname of the head node  (default: "auto")
        RAY_HEAD_PORT      Port                             (default: 6379)
        """
        addr = os.environ.get("RAY_HEAD_ADDRESS", "auto")
        port = int(os.environ.get("RAY_HEAD_PORT", "6379"))
        return cls(head_address=addr, head_port=port)

    @classmethod
    def from_slurm(cls) -> RayClusterConfig:
        """
        Auto-detect the head node IP from the SLURM environment.

        The first node in SLURM_JOB_NODELIST is treated as the Ray head.
        Falls back to from_env() if SLURM variables are not set or
        scontrol / getent are not available.
        """
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
        if not nodelist:
            return cls.from_env()
        try:
            nodes = subprocess.check_output(
                ["scontrol", "show", "hostnames", nodelist],
                text=True, stderr=subprocess.DEVNULL,
            ).strip().splitlines()
            head_node = nodes[0]
            # Resolve hostname → IP (getent is available on most Linux systems)
            ip = subprocess.check_output(
                ["getent", "hosts", head_node],
                text=True, stderr=subprocess.DEVNULL,
            ).strip().split()[0]
            port = int(os.environ.get("RAY_HEAD_PORT", "6379"))
            return cls(head_address=ip, head_port=port)
        except Exception:
            return cls.from_env()
