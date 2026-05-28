"""``jarvis start|stop|restart|status`` — daemon management commands."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time

import click
from rich.console import Console

from openjarvis.core.config import DEFAULT_CONFIG_DIR, load_config

_PID_FILE = DEFAULT_CONFIG_DIR / "server.pid"
_LOG_FILE = DEFAULT_CONFIG_DIR / "server.log"


def _read_pidfile() -> dict | None:
    """Read daemon record. Returns dict with pid/host/port if alive, else None.

    Accepts both the legacy plain-integer format (older daemons that only
    wrote the pid) and the structured JSON format. For legacy files we fall
    back to config defaults for host/port so ``status`` doesn't break for
    anyone with a running daemon when this ships.
    """
    if not _PID_FILE.exists():
        return None
    raw = _PID_FILE.read_text().strip()
    try:
        record = json.loads(raw)
        pid = int(record["pid"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        try:
            pid = int(raw)
            record = {"pid": pid}
        except ValueError:
            _PID_FILE.unlink(missing_ok=True)
            return None
    try:
        os.kill(pid, 0)
    except OSError:
        _PID_FILE.unlink(missing_ok=True)
        return None
    return record


def _read_pid() -> int | None:
    record = _read_pidfile()
    return record["pid"] if record else None


def _write_pidfile(pid: int, host: str, port: int) -> None:
    """Persist the daemon's pid + bound address so ``status`` reports truth."""
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(json.dumps({"pid": pid, "host": host, "port": port}))


def _port_in_use(host: str, port: int) -> bool:
    """Best-effort check: is anything already listening on ``host:port``?

    Done before forking the daemon so the user gets a clean error instead
    of a pidfile pointing at a process that died on startup. We use
    ``connect_ex`` instead of trial-binding because ``bind`` with
    SO_REUSEADDR is too permissive on macOS' dual-stack — a Docker
    container holding ``*:8000`` on IPv6 doesn't block an IPv4 bind, but
    uvicorn's actual bind still fails later. Connecting catches it.

    Wildcard hosts (``0.0.0.0``, ``::``) probe via loopback since you
    can't ``connect()`` to a wildcard.
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "*") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((probe_host, port)) == 0:
            return True
    return False


@click.group()
def daemon() -> None:
    """Manage the OpenJarvis server daemon."""


@daemon.command()
@click.option("--host", default=None, help="Bind address.")
@click.option("--port", default=None, type=int, help="Port number.")
@click.option("-e", "--engine", "engine_key", default=None, help="Engine backend.")
@click.option("-m", "--model", "model_name", default=None, help="Default model.")
@click.option("-a", "--agent", "agent_name", default=None, help="Agent type.")
def start(
    host: str | None,
    port: int | None,
    engine_key: str | None,
    model_name: str | None,
    agent_name: str | None,
) -> None:
    """Start the OpenJarvis server as a background daemon."""
    console = Console(stderr=True)

    existing = _read_pid()
    if existing is not None:
        console.print(f"[yellow]Server already running (PID {existing}).[/yellow]")
        console.print("Use 'jarvis stop' to stop it first, or 'jarvis restart'.")
        sys.exit(1)

    config = load_config()
    bind_host = host or config.server.host
    bind_port = port or config.server.port

    if _port_in_use(bind_host, bind_port):
        console.print(
            f"[red]Port {bind_port} is already in use on {bind_host}.[/red]\n"
            f"  Something else is bound there (often Docker, another jarvis,\n"
            f"  or a leftover dev server). Free it or pick a different port:\n"
            f"    jarvis start --port <other>"
        )
        sys.exit(1)

    # Build command to run jarvis serve
    cmd = [sys.executable, "-m", "openjarvis.cli", "serve"]
    if host:
        cmd.extend(["--host", host])
    if port:
        cmd.extend(["--port", str(port)])
    if engine_key:
        cmd.extend(["--engine", engine_key])
    if model_name:
        cmd.extend(["--model", model_name])
    if agent_name:
        cmd.extend(["--agent", agent_name])

    # Start as background process
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(_LOG_FILE, "a")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )
    _write_pidfile(proc.pid, bind_host, bind_port)

    console.print(
        f"[green]OpenJarvis server started[/green] (PID {proc.pid})\n"
        f"  URL: http://{bind_host}:{bind_port}\n"
        f"  Log: {_LOG_FILE}"
    )


@daemon.command()
def stop() -> None:
    """Stop the running OpenJarvis server daemon."""
    console = Console(stderr=True)
    pid = _read_pid()
    if pid is None:
        console.print("[yellow]No running server found.[/yellow]")
        sys.exit(1)

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 10 seconds for graceful shutdown
        for _ in range(20):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except OSError:
                break
        else:
            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    except OSError:
        pass

    _PID_FILE.unlink(missing_ok=True)
    console.print(f"[green]Server stopped[/green] (PID {pid}).")


@daemon.command()
@click.pass_context
def restart(ctx: click.Context) -> None:
    """Restart the OpenJarvis server daemon."""
    console = Console(stderr=True)
    pid = _read_pid()
    if pid is not None:
        console.print(f"Stopping server (PID {pid})...")
        ctx.invoke(stop)
    ctx.invoke(start)


@daemon.command()
def status() -> None:
    """Show status of the OpenJarvis server daemon."""
    console = Console(stderr=True)
    record = _read_pidfile()
    if record is None:
        console.print("[yellow]Server is not running.[/yellow]")
        return

    pid = record["pid"]

    # Get process info
    uptime_info = ""
    try:
        import psutil

        proc = psutil.Process(pid)
        uptime = time.time() - proc.create_time()
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_info = f"\n  Uptime: {hours}h {minutes}m {seconds}s"
    except (ImportError, Exception):
        pass

    # Prefer the address the daemon was actually started on. Legacy pidfiles
    # (pre-structured-record) won't have these — fall back to config defaults
    # so we don't lie about the URL if we genuinely don't know it.
    if "host" in record and "port" in record:
        bind_host = record["host"]
        bind_port = record["port"]
    else:
        config = load_config()
        bind_host = config.server.host
        bind_port = config.server.port
    console.print(
        f"[green]Server is running[/green] (PID {pid}){uptime_info}\n"
        f"  URL: http://{bind_host}:{bind_port}\n"
        f"  Log: {_LOG_FILE}"
    )


__all__ = ["daemon", "start", "stop", "restart", "status"]
