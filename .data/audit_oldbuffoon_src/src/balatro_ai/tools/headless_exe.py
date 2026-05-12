"""Helpers for creating a tiny-startup Balatro executable copy."""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path


TINY_STARTUP_CONF_LUA = """_RELEASE_MODE = true
_DEMO = false

function love.conf(t)
\tt.console = not _RELEASE_MODE
\tt.title = 'Balatro'
\tt.window.width = 100
\tt.window.height = 100
\tt.window.minwidth = 1
\tt.window.minheight = 1
\tt.window.fullscreen = false
\tt.window.borderless = false
\tt.window.resizable = false
\tt.window.vsync = 0
end
"""


def ensure_tiny_startup_copy(source_exe: Path, output_dir: Path) -> tuple[Path, Path | None]:
    """Create/update a Balatro copy whose embedded conf.lua starts tiny.

    BalatroBot's headless mode runs after the game has already created its first
    window. A patched copy avoids the fullscreen-sized startup flash without
    modifying the user's Steam install.
    """

    source_exe = source_exe.resolve()
    output_dir = output_dir.resolve()
    target_exe = output_dir / source_exe.name
    target_lovely = output_dir / "version.dll"

    if source_exe == target_exe and target_exe.exists():
        return target_exe, target_lovely if target_lovely.exists() else None

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_runtime_files(source_exe.parent, output_dir, skip_name=source_exe.name)

    if _needs_rebuild(source_exe, target_exe):
        build_tiny_startup_exe(source_exe, target_exe)

    return target_exe, target_lovely if target_lovely.exists() else None


def build_tiny_startup_exe(source_exe: Path, target_exe: Path) -> None:
    """Write a fused LOVE executable with only conf.lua replaced."""

    target_exe.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = source_exe.read_bytes()
    with zipfile.ZipFile(source_exe) as source_zip:
        archive_offset = min(info.header_offset for info in source_zip.infolist())
        fused_prefix = source_bytes[:archive_offset]
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as target_zip:
            for source_info in source_zip.infolist():
                data = (
                    TINY_STARTUP_CONF_LUA.encode("utf-8")
                    if source_info.filename == "conf.lua"
                    else source_zip.read(source_info.filename)
                )
                target_info = zipfile.ZipInfo(source_info.filename, source_info.date_time)
                target_info.compress_type = source_info.compress_type
                target_info.external_attr = source_info.external_attr
                target_zip.writestr(target_info, data)

    target_exe.write_bytes(fused_prefix + archive_bytes.getvalue())


def _copy_runtime_files(source_dir: Path, output_dir: Path, *, skip_name: str) -> None:
    for source_file in source_dir.iterdir():
        if not source_file.is_file() or source_file.name == skip_name:
            continue
        target_file = output_dir / source_file.name
        if _needs_copy(source_file, target_file):
            shutil.copy2(source_file, target_file)


def _needs_copy(source_file: Path, target_file: Path) -> bool:
    if not target_file.exists():
        return True
    source_stat = source_file.stat()
    target_stat = target_file.stat()
    return source_stat.st_size != target_stat.st_size or source_stat.st_mtime > target_stat.st_mtime


def _needs_rebuild(source_exe: Path, target_exe: Path) -> bool:
    if not target_exe.exists():
        return True
    return source_exe.stat().st_mtime > target_exe.stat().st_mtime
