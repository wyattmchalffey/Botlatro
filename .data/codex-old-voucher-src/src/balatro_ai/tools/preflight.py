"""Local preflight checks for running Botlatro against BalatroBot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENDPOINT = "http://127.0.0.1:12346"
BALATRO_APP_ID = "2379780"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str

    @property
    def status(self) -> str:
        return "OK" if self.ok else "MISSING"


def run_checks(endpoint: str = DEFAULT_ENDPOINT) -> tuple[Check, ...]:
    balatro_install = _find_balatro_install()
    return (
        _command_check("python"),
        _command_check("uv"),
        _command_check("uvx"),
        _balatro_install_check(balatro_install),
        _lovely_check(balatro_install),
        _mods_folder_check(),
        _mod_folder_check("smods"),
        _mod_folder_check("balatrobot"),
        _bridge_health_check(endpoint),
    )


def checks_to_text(checks: tuple[Check, ...]) -> str:
    width = max(len(check.name) for check in checks)
    lines = []
    for check in checks:
        lines.append(f"{check.status:<8} {check.name:<{width}}  {check.detail}")
    return "\n".join(lines)


def _command_check(command: str) -> Check:
    path = shutil.which(command)
    if path is None and command in {"uv", "uvx"}:
        path = _python_user_script(command)
    return Check(
        name=f"command:{command}",
        ok=path is not None,
        detail=path or "not found on PATH",
    )


def _python_user_script(command: str) -> str | None:
    script = Path(os.environ.get("APPDATA", "")) / "Python" / "Python312" / "Scripts" / f"{command}.exe"
    return str(script) if script.exists() else None


def _balatro_install_check(install: Path | None) -> Check:
    return Check(
        name="balatro install",
        ok=install is not None,
        detail=str(install) if install else "Balatro install was not found in Steam libraries",
    )


def _lovely_check(install: Path | None) -> Check:
    if install is None:
        return Check("lovely injector", False, "Balatro install was not found")
    lovely = install / "version.dll"
    return Check(
        name="lovely injector",
        ok=lovely.exists(),
        detail=str(lovely) if lovely.exists() else f"{lovely} does not exist",
    )


def _mods_folder_check() -> Check:
    mods = _mods_path()
    if mods is None:
        return Check("mods folder", False, "APPDATA is not set")
    if not mods.exists():
        return Check("mods folder", False, f"{mods} does not exist")

    installed = [path.name for path in mods.iterdir() if path.is_dir()]
    if not installed:
        return Check("mods folder", False, f"{mods} exists but contains no mod folders")

    return Check("mods folder", True, f"{mods} contains: {', '.join(installed)}")


def _mod_folder_check(mod_name: str) -> Check:
    mods = _mods_path()
    if mods is None:
        return Check(f"mod:{mod_name}", False, "APPDATA is not set")
    mod_path = mods / mod_name
    return Check(
        name=f"mod:{mod_name}",
        ok=mod_path.exists(),
        detail=str(mod_path) if mod_path.exists() else f"{mod_path} does not exist",
    )


def _mods_path() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Balatro" / "Mods"


def _bridge_health_check(endpoint: str) -> Check:
    payload = {"jsonrpc": "2.0", "method": "health", "id": 1}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return Check("balatrobot health", False, f"{endpoint} unavailable: {exc.reason}")
    except TimeoutError:
        return Check("balatrobot health", False, f"{endpoint} timed out")

    if data.get("result", {}).get("status") == "ok":
        return Check("balatrobot health", True, f"{endpoint} returned ok")
    return Check("balatrobot health", False, f"{endpoint} returned unexpected response: {data}")


def _find_balatro_install() -> Path | None:
    candidates = [
        Path("C:/Program Files (x86)/Steam/steamapps/common/Balatro"),
        Path("C:/Program Files/Steam/steamapps/common/Balatro"),
    ]
    for library in _steam_libraries():
        candidates.append(library / "steamapps" / "common" / "Balatro")

    for candidate in candidates:
        if (candidate / "Balatro.exe").exists():
            return candidate
    return None


def _steam_libraries() -> tuple[Path, ...]:
    vdf = Path("C:/Program Files (x86)/Steam/steamapps/libraryfolders.vdf")
    if not vdf.exists():
        return ()

    libraries: list[Path] = []
    for line in vdf.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith('"path"'):
            parts = stripped.split('"')
            if len(parts) >= 4:
                libraries.append(Path(parts[3].replace("\\\\", "\\")))
    return tuple(libraries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local Botlatro/BalatroBot setup.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="BalatroBot JSON-RPC endpoint.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = run_checks(endpoint=args.endpoint)
    print(checks_to_text(checks))
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
