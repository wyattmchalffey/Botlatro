from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import context  # noqa: F401
from balatro_ai.tools.headless_exe import TINY_STARTUP_CONF_LUA, build_tiny_startup_exe, ensure_tiny_startup_copy


class HeadlessExeTests(unittest.TestCase):
    def test_build_tiny_startup_exe_replaces_conf_and_preserves_stub(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "Balatro.exe"
            target = temp / "patched" / "Balatro.exe"

            archive = temp / "payload.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("conf.lua", "function love.conf(t)\nend\n")
                zipped.writestr("main.lua", "print('hello')\n")
            source.write_bytes(b"EXE-STUB" + archive.read_bytes())

            build_tiny_startup_exe(source, target)

            self.assertTrue(target.read_bytes().startswith(b"EXE-STUB"))
            with zipfile.ZipFile(target) as patched:
                self.assertEqual(patched.read("conf.lua").decode("utf-8"), TINY_STARTUP_CONF_LUA)
                self.assertEqual(patched.read("main.lua").decode("utf-8"), "print('hello')\n")

    def test_ensure_tiny_startup_copy_copies_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_dir = temp / "source"
            target_dir = temp / "target"
            source_dir.mkdir()

            archive = source_dir / "payload.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("conf.lua", "function love.conf(t)\nend\n")
            source_exe = source_dir / "Balatro.exe"
            source_exe.write_bytes(b"EXE-STUB" + archive.read_bytes())
            (source_dir / "version.dll").write_text("lovely", encoding="utf-8")

            target_exe, target_lovely = ensure_tiny_startup_copy(source_exe, target_dir)

            self.assertEqual(target_exe, target_dir.resolve() / "Balatro.exe")
            self.assertEqual(target_lovely, target_dir.resolve() / "version.dll")
            self.assertTrue(target_lovely.exists())
            with zipfile.ZipFile(target_exe) as patched:
                self.assertEqual(patched.read("conf.lua").decode("utf-8"), TINY_STARTUP_CONF_LUA)


if __name__ == "__main__":
    unittest.main()
