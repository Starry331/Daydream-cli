"""Tests for the thin MTP install plumbing.

These tests exercise the bookkeeping logic (slugs, layout discovery,
broken-symlink detection) without actually downloading anything.
A separate integration test would need real Hugging Face access.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from daydream.backends.mtp_install import (
    DEFAULT_BASE_TRUNK,
    DEFAULT_SIDECAR_REPO,
    DOWNLOAD_FILES,
    THIN_INSTALL_DOWNLOAD_BYTES,
    _slug,
    find_thin_install,
)


class SlugTests(unittest.TestCase):
    def test_slug_replaces_slashes(self) -> None:
        self.assertEqual(_slug("mlx-community/Qwen3.6-27B-4bit"), "mlx-community--qwen3.6-27b-4bit")
        self.assertEqual(_slug("a/b/c"), "a--b--c")

    def test_slug_lowercases(self) -> None:
        self.assertEqual(_slug("FooBar/Baz"), "foobar--baz")


class FindThinInstallTests(unittest.TestCase):
    """Layout discovery without hitting the network."""

    def _make_install(self, tmp_home: Path, base_trunk: str, *, with_runtime: bool = True, broken: bool = False) -> Path:
        """Fabricate a thin install dir at $DAYDREAM_HOME/mtp/<slug>/."""
        slug_dir = tmp_home / "mtp" / _slug(base_trunk)
        slug_dir.mkdir(parents=True)

        if with_runtime:
            (slug_dir / "mtplx_runtime.json").write_text(json.dumps({
                "arch_id": "qwen3-next-mtp",
                "base_trunk": base_trunk,
                "mtp_sidecar": "test-sidecar",
                "mtp_depth_max": 3,
            }))
        (slug_dir / "mtp.safetensors").write_bytes(b"x" * 100)  # tiny stand-in

        # Required symlinks — point at a fake trunk so non-broken tests
        # see real files. For the "broken" case, point at a missing dir.
        fake_trunk = tmp_home / ("fake-trunk-missing" if broken else "fake-trunk")
        if not broken:
            fake_trunk.mkdir()
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ):
            target = fake_trunk / name
            if not broken:
                target.write_text(f"stub-{name}")
            (slug_dir / name).symlink_to(target)

        return slug_dir

    def test_no_install_returns_none(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DAYDREAM_HOME": tmp}):
                # Re-import after env change so DAYDREAM_HOME picks up.
                import importlib
                from daydream import config, backends
                importlib.reload(config)
                from daydream.backends import mtp_install
                importlib.reload(mtp_install)
                self.assertIsNone(mtp_install.find_thin_install("not-a-trunk"))

    def test_find_install(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DAYDREAM_HOME": tmp}):
                import importlib
                from daydream import config
                importlib.reload(config)
                from daydream.backends import mtp_install
                importlib.reload(mtp_install)

                slug_dir = self._make_install(Path(tmp), DEFAULT_BASE_TRUNK)
                install = mtp_install.find_thin_install(DEFAULT_BASE_TRUNK)
                self.assertIsNotNone(install)
                assert install is not None
                self.assertEqual(install.install_dir, slug_dir)
                self.assertFalse(install.broken_trunk)
                self.assertEqual(install.base_trunk, DEFAULT_BASE_TRUNK)

    def test_detects_broken_symlinks(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DAYDREAM_HOME": tmp}):
                import importlib
                from daydream import config
                importlib.reload(config)
                from daydream.backends import mtp_install
                importlib.reload(mtp_install)

                self._make_install(Path(tmp), DEFAULT_BASE_TRUNK, broken=True)
                install = mtp_install.find_thin_install(DEFAULT_BASE_TRUNK)
                self.assertIsNotNone(install)
                assert install is not None
                self.assertTrue(install.broken_trunk)

    def test_half_finished_install_ignored(self) -> None:
        """Without the runtime JSON, it's a half-finished install — ignore."""
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DAYDREAM_HOME": tmp}):
                import importlib
                from daydream import config
                importlib.reload(config)
                from daydream.backends import mtp_install
                importlib.reload(mtp_install)

                self._make_install(Path(tmp), DEFAULT_BASE_TRUNK, with_runtime=False)
                self.assertIsNone(mtp_install.find_thin_install(DEFAULT_BASE_TRUNK))


class DownloadSizeTests(unittest.TestCase):
    def test_download_size_is_under_500MB(self) -> None:
        # Sanity check on the thin-install promise: the downloaded
        # portion has to stay well below the 16 GB full-repo footprint.
        self.assertLess(THIN_INSTALL_DOWNLOAD_BYTES, 500 * 1024 * 1024)
        # And it's clearly more than just metadata (mtp.safetensors is
        # ~337 MB of real weights).
        self.assertGreater(THIN_INSTALL_DOWNLOAD_BYTES, 300 * 1024 * 1024)

    def test_download_files_listed(self) -> None:
        names = [n for n, _ in DOWNLOAD_FILES]
        self.assertIn("mtp.safetensors", names)
        self.assertIn("mtplx_runtime.json", names)
        # config.json MUST be downloaded (not symlinked) — the trunk's
        # config doesn't declare mtplx_mtp_quantization.
        self.assertIn("config.json", names)


class SymlinkSelectionTests(unittest.TestCase):
    def test_config_json_is_not_symlinked(self) -> None:
        # Regression guard: symlinking config.json from the trunk
        # caused a group_size mismatch (trunk: gs=64; mtp.safetensors:
        # gs=32). MTPLX's augmented config.json declares the right
        # quantization params for the MTP head.
        from daydream.backends.mtp_install import (
            SYMLINK_FILES_REQUIRED,
            SYMLINK_FILES_OPTIONAL,
        )
        self.assertNotIn("config.json", SYMLINK_FILES_REQUIRED)
        self.assertNotIn("config.json", SYMLINK_FILES_OPTIONAL)


if __name__ == "__main__":
    unittest.main()
