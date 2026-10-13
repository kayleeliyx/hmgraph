#!/usr/bin/env python3
"""Start the independent, local human review workbench."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))


def state_paths(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = Path(config.get("state_dir", "state"))
    if not state.is_absolute():
        state = config_path.parent / state
    state = state.resolve()
    return state / "server.json", state / "server.log"


def _metadata(path):
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or type(value.get("pid")) is not int or value["pid"] <= 0:
        raise ValueError(f"Invalid server metadata: {path}. Nothing was stopped or replaced.")
    return value


def _process_state(pid):
    """Linux start time protects against a recycled PID; zombies are not live."""
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        return {"state": fields[0], "start_ticks": fields[19]}
    except (OSError, IndexError):
        return None


def _alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    state = _process_state(pid)
    if state and state["state"] == "Z":
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _owned(metadata, config_path):
    """Never signal a PID without checking its exact expected foreground CLI."""
    if not metadata or metadata.get("config") != str(Path(config_path).resolve()):
        return False
    if not isinstance(metadata.get("host"), str) or type(metadata.get("port")) is not int:
        return False
    script = str(Path(__file__).resolve())
    if metadata.get("script") != script or not _alive(metadata.get("pid")):
        return False
    state = _process_state(metadata["pid"])
    if metadata.get("proc_start_ticks") and (not state or state["start_ticks"] != metadata["proc_start_ticks"]):
        return False
    try:
        argv = (Path("/proc") / str(metadata["pid"]) / "cmdline").read_bytes().split(b"\0")
        argv = [os.fsdecode(item) for item in argv if item]
    except OSError:
        return False
    expected = ["-u", script, "--config", metadata["config"], "--host", metadata["host"],
                "--port", str(metadata["port"])]
    return len(argv) == len(expected) + 1 and argv[1:] == expected


def _url(host, port):
    # Listen-all addresses, when explicitly requested, are checked via loopback.
    host = "127.0.0.1" if host == "0.0.0.0" else "::1" if host == "::" else host
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


def _health(host, port, timeout=0.5):
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(Request(_url(host, port) + "/api/bootstrap"), timeout=timeout) as response:
            if not response.headers.get("Server", "").startswith("ActGroundReview/"):
                return False
            result = json.loads(response.read())
            return (response.status == 200 and isinstance(result, dict) and
                    isinstance(result.get("models"), list) and type(result.get("example_count")) is int)
    except (OSError, URLError, ValueError, TimeoutError):
        return False


def _port_occupied(host, port):
    try:
        with socket.create_connection(("127.0.0.1" if host == "0.0.0.0" else host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _save_metadata(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".server-", suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def status(config_path, host="127.0.0.1", port=8765):
    metadata_path, log_path = state_paths(config_path)
    metadata = _metadata(metadata_path)
    if metadata and _alive(metadata["pid"]):
        if not _owned(metadata, config_path):
            return {"state": "foreign", "healthy": False,
                    "message": "Stored PID is not this review server; it will not be stopped or replaced."}
        healthy = _health(metadata["host"], metadata["port"])
        return {"state": "running" if healthy else "unhealthy", "healthy": healthy,
                "pid": metadata["pid"], "url": _url(metadata["host"], metadata["port"]), "log": str(log_path)}
    healthy = _health(host, port)
    if healthy or _port_occupied(host, port):
        return {"state": "unmanaged", "healthy": healthy, "url": _url(host, port),
                "message": "A service occupies this port without owned server metadata."}
    return {"state": "stopped", "healthy": False, "log": str(log_path)}


def stop(config_path):
    metadata_path, _ = state_paths(config_path)
    metadata = _metadata(metadata_path)
    if not metadata or not _alive(metadata["pid"]):
        if metadata_path.exists():
            metadata_path.unlink()
        return {"state": "stopped", "message": "Review server is stopped."}
    if not _owned(metadata, config_path):
        raise ValueError("Stored PID is not this review server. Nothing was stopped.")
    os.kill(metadata["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 3
    while _alive(metadata["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _alive(metadata["pid"]):
        return {"state": "stopping", "message": "Stop requested; process is still exiting."}
    if _metadata(metadata_path) == metadata:
        metadata_path.unlink()
    return {"state": "stopped", "message": "Review server stopped. Saved labels are retained."}


def background(config_path, host="127.0.0.1", port=8765, readiness_timeout=10.0):
    config_path = Path(config_path).resolve()
    metadata_path, log_path = state_paths(config_path)
    previous = _metadata(metadata_path)
    if previous and _alive(previous["pid"]):
        if not _owned(previous, config_path):
            raise ValueError("Stored PID belongs to another process. Nothing was stopped or replaced.")
        if previous["host"] != host or previous["port"] != port:
            raise ValueError(f"Review server already runs at {_url(previous['host'], previous['port'])}; use --stop before changing its address.")
        if _health(host, port):
            return {"state": "running", "reused": True, "pid": previous["pid"],
                    "url": _url(host, port), "log": str(log_path)}
        raise ValueError(f"Owned server is running but not responding. See {log_path}; use --stop before restarting.")
    if _port_occupied(host, port):
        raise ValueError(f"Port {port} is occupied without owned server metadata. Nothing was stopped or replaced.")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(config_path),
               "--host", host, "--port", str(port)]
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True, close_fds=True)
    state = _process_state(process.pid)
    metadata = {"pid": process.pid, "host": host, "port": port, "config": str(config_path),
                "script": str(Path(__file__).resolve()), "python": sys.executable,
                "proc_start_ticks": state["start_ticks"] if state else None,
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "url": _url(host, port)}
    _save_metadata(metadata_path, metadata)
    deadline = time.monotonic() + readiness_timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Review server exited during startup. See {log_path}.")
            if _owned(metadata, config_path) and _health(host, port, timeout=min(0.5, max(0.01, deadline - time.monotonic()))):
                return {"state": "running", "reused": False, "pid": process.pid,
                        "url": _url(host, port), "log": str(log_path)}
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        raise RuntimeError(f"Review server did not become ready within {readiness_timeout:g}s. See {log_path}.")
    except Exception:
        if process.poll() is None:
            process.terminate()
        if _metadata(metadata_path) == metadata:
            metadata_path.unlink()
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--background", action="store_true", help="Start a durable local server; reuse an owned running instance.")
    modes.add_argument("--status", action="store_true", help="Check the owned server's health.")
    modes.add_argument("--stop", action="store_true", help="Stop only this config's verified review server; retain labels.")
    modes.add_argument("--test", action="store_true", help="Run CPU-only unit and HTTP tests.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or args.port == 0 and (args.background or args.status):
        parser.error("port must be 1–65535 for lifecycle commands; foreground also accepts 0")
    if args.test:
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"), pattern="test_*.py"))
        return 0 if result.wasSuccessful() else 1
    if args.background or args.status or args.stop:
        try:
            result = (background(args.config, args.host, args.port) if args.background else
                      status(args.config, args.host, args.port) if args.status else stop(args.config))
        except (OSError, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["state"] in {"running", "stopped", "stopping"} else 1
    from server import serve
    serve(args.config.resolve(), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
