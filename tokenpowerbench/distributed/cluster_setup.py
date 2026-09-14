"""Create and resolve explicit cluster profiles without starting cluster services."""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from .ray_cluster import RayClusterConfig

_FIELDS = {"schema_version", "mode", "head_address", "head_port", "worker_addresses", "network_interface"}
_INTERFACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,14}\Z")
_HOSTNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _ipv4(value):
    if not isinstance(value, str):
        raise ValueError("Node addresses must be IPv4 address strings")
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as error:
        raise ValueError(f"Invalid IPv4 node address: {value!r}") from error
    if (address.is_unspecified or address.is_loopback or address.is_link_local
            or address.is_multicast or address.is_reserved):
        raise ValueError(f"Node address must be a usable IPv4 peer address: {value!r}")
    return str(address)


def _port(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("head_port must be an integer between 1 and 65535")
    return value


def _interface(value):
    if value is not None and (not isinstance(value, str) or not _INTERFACE.fullmatch(value)):
        raise ValueError("network_interface must be null or a safe interface name of 1 to 15 characters")
    return value


def validate_profile(profile: dict) -> dict:
    """Return a validated copy; reject unknown, missing, or contradictory fields."""
    if not isinstance(profile, dict):
        raise ValueError("Cluster profile must be a JSON object")
    if set(profile) != _FIELDS:
        missing = sorted(_FIELDS - set(profile))
        unknown = sorted((repr(key) for key in set(profile) - _FIELDS))
        raise ValueError(f"Invalid cluster profile fields; missing={missing}, unknown={unknown}")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    mode = profile["mode"]
    if mode not in ("manual", "slurm"):
        raise ValueError("mode must be 'manual' or 'slurm'")
    port = _port(profile["head_port"])
    interface = _interface(profile["network_interface"])
    if not isinstance(profile["worker_addresses"], list):
        raise ValueError("worker_addresses must be a list of IPv4 address strings")
    workers = [_ipv4(value) for value in profile["worker_addresses"]]
    if len(set(workers)) != len(workers):
        raise ValueError("worker_addresses must not contain duplicates")
    if mode == "manual":
        head = _ipv4(profile["head_address"])
        if head in workers:
            raise ValueError("The head address must not also appear in worker_addresses")
        if interface is not None:
            raise ValueError("network_interface is available only in slurm mode")
    else:
        head = profile["head_address"]
        if head is not None or workers:
            raise ValueError("Slurm profiles must use head_address=null and worker_addresses=[]; nodes are discovered per allocation")
    return {"schema_version": 1, "mode": mode, "head_address": head, "head_port": port,
            "worker_addresses": workers, "network_interface": interface}


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON profile field: {key!r}")
        value[key] = item
    return value


def load_profile(path) -> dict:
    """Read strict JSON and validate its complete profile schema."""
    with Path(path).open(encoding="utf-8") as stream:
        return validate_profile(json.load(stream, object_pairs_hook=_unique_object))


def _slurm_nodes():
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "").strip()
    if not job_id or not nodelist:
        raise RuntimeError("Slurm profile resolution requires a current allocation with SLURM_JOB_ID and SLURM_JOB_NODELIST")
    try:
        output = subprocess.check_output(["scontrol", "show", "hostnames", nodelist],
                                         text=True, stderr=subprocess.PIPE, timeout=5)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Cannot discover the current Slurm allocation with scontrol show hostnames") from error
    nodes = [line.strip() for line in output.splitlines() if line.strip()]
    if not nodes or any(not _HOSTNAME.fullmatch(node) for node in nodes) or len(set(nodes)) != len(nodes):
        raise RuntimeError("scontrol returned an empty or invalid allocation hostname list")
    return nodes


def _slurm_interface_address(head, interface):
    command = ["srun", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--gres=none",
               "--nodelist", head, "ip", "-j", "-4", "address", "show", "dev", interface]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Cannot query interface {interface!r} on the allocated Slurm head with srun") from error
    try:
        devices = json.loads(output)
        if not isinstance(devices, list):
            raise ValueError("Expected an interface list")
        addresses = set()
        for device in devices:
            if not isinstance(device, dict) or not isinstance(device.get("addr_info"), list):
                raise ValueError("Expected interface address information")
            for entry in device["addr_info"]:
                if not isinstance(entry, dict):
                    raise ValueError("Expected an address object")
                if entry.get("family") != "inet" or entry.get("scope") != "global":
                    continue
                address = ipaddress.IPv4Address(_ipv4(entry.get("local", "")))
                if not (address.is_unspecified or address.is_loopback or address.is_link_local
                        or address.is_multicast or address.is_reserved):
                    addresses.add(str(address))
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"Invalid IPv4 interface data returned for {interface!r} on the Slurm head") from error
    if len(addresses) != 1:
        raise RuntimeError(f"Interface {interface!r} on the Slurm head must have exactly one global usable IPv4 address; found {len(addresses)}")
    return addresses.pop()


def resolve_profile(profile: dict) -> RayClusterConfig:
    """Resolve this allocation explicitly, ignoring Ray address environment hints.

    Slurm discovery is never persisted. An interface selection queries the
    allocated head through srun so its address matches the launcher interface.
    """
    profile = validate_profile(profile)
    head = profile["head_address"]
    if profile["mode"] == "slurm":
        head = _slurm_nodes()[0]
        if profile["network_interface"] is not None:
            head = _slurm_interface_address(head, profile["network_interface"])
    return RayClusterConfig(head_address=head, head_port=profile["head_port"])


def _ask(prompt, convert):
    while True:
        try:
            return convert(input(prompt).strip())
        except ValueError as error:
            print(f"Invalid input: {error}")


def _mode(value):
    modes = {"": "manual", "1": "manual", "manual": "manual", "2": "slurm", "slurm": "slurm"}
    if value.lower() not in modes:
        raise ValueError("Choose 1/manual or 2/slurm")
    return modes[value.lower()]


def _workers(value, head):
    workers = [_ipv4(item.strip()) for item in value.split(",")] if value else []
    if len(set(workers)) != len(workers) or head in workers:
        raise ValueError("Worker addresses must be unique and different from the head address")
    return workers


def _instructions(profile):
    port = profile["head_port"]
    if profile["mode"] == "manual":
        head = profile["head_address"]
        lines = ["Run each command on its corresponding node:", f"Head node ({head}):",
                 f"  VLLM_HOST_IP={head} ray start --head --node-ip-address={head} --port={port}"]
        for worker in profile["worker_addresses"]:
            lines += [f"Worker node ({worker}):",
                      f"  VLLM_HOST_IP={worker} ray start --node-ip-address={worker} --address={head}:{port}"]
        if not profile["worker_addresses"]:
            lines.append("No worker startup commands requested; existing registered workers may still be used by Ray.")
        lines.append("Participating GPU nodes need the same environment and access to the selected model.")
        return lines
    lines = ["Discover the current allocation with:", '  scontrol show hostnames "$SLURM_JOB_NODELIST"',
             "The first hostname is the Ray head; all remaining hostnames are workers.",
             f"Ray head port: {port}."]
    if profile["network_interface"]:
        lines.append(f"Use network interface {profile['network_interface']} consistently on every allocated node.")
    if os.environ.get("SLURM_JOB_ID", "").strip() and os.environ.get("SLURM_JOB_NODELIST", "").strip():
        try:
            nodes = _slurm_nodes()
        except RuntimeError as error:
            lines.append(f"Allocation discovery unavailable: {error}. It will be retried when resolving the profile.")
        else:
            lines += [f"Current head: {nodes[0]}", f"Current workers: {', '.join(nodes[1:]) or '(none)'}"]
    else:
        lines.append("No current Slurm allocation; hostname discovery is deferred until this profile is used inside an allocation.")
    lines.append("The profile stores no allocation hostnames or node IPs.")
    lines.append("Submit a new allocation with scripts/submit_jobs.sh, or run scripts/slurm_head.sh on the first node inside an existing allocation.")
    return lines


def _save_new_profile(path, profile):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(profile, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Publishing the complete file with link() is atomic and refuses an
        # existing destination, including one created while answers were read.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def configure(output: Path) -> int:
    """Run the terminal wizard; save once, without starting or contacting nodes.

    Return 0 on success, 1 for EOF or file errors, and 130 for Ctrl-C. Invalid
    answers are retried. Existing profiles are never overwritten.
    """
    output = Path(output)
    if os.path.lexists(output):
        print(f"Cannot create profile: {output} already exists. Choose a different output path.")
        return 1
    try:
        print("TokenPowerBench cluster setup")
        mode = _ask("Cluster mode [1=manual, 2=slurm] (1): ", _mode)
        profile = {"schema_version": 1, "mode": mode, "head_address": None, "head_port": 6379,
                   "worker_addresses": [], "network_interface": None}
        if mode == "manual":
            profile["head_address"] = _ask("Head node IPv4 address: ", _ipv4)
            profile["worker_addresses"] = _ask("Worker node IPv4 addresses (comma-separated, optional): ",
                                                lambda value: _workers(value, profile["head_address"]))
        profile["head_port"] = _ask("Ray head port (6379): ", lambda value: _port(int(value or "6379")))
        if mode == "slurm":
            profile["network_interface"] = _ask("Network interface (optional, e.g. ib0): ",
                                                 lambda value: _interface(value or None))
        profile = validate_profile(profile)
        instructions = _instructions(profile)
        _save_new_profile(output, profile)
    except EOFError:
        print("Cluster setup cancelled: no profile was saved.")
        return 1
    except KeyboardInterrupt:
        print("\nCluster setup cancelled: no partial profile was saved.")
        return 130
    except (OSError, ValueError) as error:
        print(f"Cannot create cluster profile: {error}")
        return 1
    print(f"Saved cluster profile: {output}")
    for line in instructions:
        print(line)
    print(f"Benchmark command after Ray is ready: tpbench-multi --cluster-config {shlex.quote(str(output))} --models MODEL")
    print("Setup does not SSH to nodes, start Ray, or submit Slurm jobs.")
    return 0
