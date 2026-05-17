"""Tests for the Daydream backend abstraction (Phase 2A)."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from daydream.backends import (
    GenerationBackend,
    LookupBackend,
    MLXBackend,
    MTPBackend,
    backend_for,
    detect_mtp_sidecar,
)
from daydream.backends.base import BackendCapabilities, SamplingParams, SpeculativeParams
from daydream.backends.mtp import (
    KNOWN_MTP_SIDECAR_REPOS,
    MTPSidecar,
    _read_runtime_json,
    mtplx_available,
)


class BackendDispatchTests(unittest.TestCase):
    def test_backend_for_returns_concrete_class(self) -> None:
        # We can't construct a real MLX model here, but the factory
        # only references the model in __init__ — so a sentinel works.
        sentinel_model = object()
        sentinel_tok = object()
        self.assertIsInstance(backend_for("none", sentinel_model, sentinel_tok), MLXBackend)
        self.assertIsInstance(backend_for("draft", sentinel_model, sentinel_tok), MLXBackend)
        self.assertIsInstance(backend_for("lookup", sentinel_model, sentinel_tok), LookupBackend)
        self.assertIsInstance(backend_for("mtp", sentinel_model, sentinel_tok), MTPBackend)

    def test_backend_is_generation_backend(self) -> None:
        backend = backend_for("none", object(), object())
        self.assertIsInstance(backend, GenerationBackend)


class MTPSidecarDetectionTests(unittest.TestCase):
    """Sidecar detection has to work without any real MLX model loaded
    — it only reads JSON files. These tests fabricate a fake HF cache
    layout in a tmpdir and point HF_HUB_CACHE at it.
    """

    def _make_fake_cache(self, tmpdir: Path, *, with_sidecar: bool, base_trunk: str = "mlx-community/Qwen3.6-27B-4bit") -> Path:
        repo = tmpdir / "models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
        snapshot = repo / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        if with_sidecar:
            (snapshot / "mtplx_runtime.json").write_text(
                json.dumps({
                    "arch_id": "qwen3-next-mtp",
                    "base_trunk": base_trunk,
                    "mtp_sidecar": "Qwen3.6-27B-MTPLX-CyanKiwi",
                    "mtp_depth_max": 3,
                })
            )
        return tmpdir

    def _isolated_env(self, hf_cache: str, dd_home: str):
        """Patch env + reload the modules that resolve DAYDREAM_HOME at
        import time. Returns a context manager."""
        import importlib

        ctx = patch.dict(os.environ, {"HF_HUB_CACHE": hf_cache, "DAYDREAM_HOME": dd_home})
        ctx.__enter__()
        from daydream import config
        importlib.reload(config)
        from daydream.backends import mtp_install
        importlib.reload(mtp_install)
        from daydream.backends import mtp
        importlib.reload(mtp)
        return ctx, mtp

    def test_detect_returns_none_when_cache_empty(self) -> None:
        # Isolate from the user's real `~/.daydream/mtp/` so tests
        # don't see a real install during local development.
        with TemporaryDirectory() as hf_cache, TemporaryDirectory() as dd_home:
            ctx, mtp = self._isolated_env(hf_cache, dd_home)
            try:
                self.assertIsNone(mtp.detect_mtp_sidecar())
            finally:
                ctx.__exit__(None, None, None)

    def test_detect_finds_sidecar(self) -> None:
        with TemporaryDirectory() as hf_cache, TemporaryDirectory() as dd_home:
            self._make_fake_cache(Path(hf_cache), with_sidecar=True)
            ctx, mtp = self._isolated_env(hf_cache, dd_home)
            try:
                sidecar = mtp.detect_mtp_sidecar()
                self.assertIsNotNone(sidecar)
                assert sidecar is not None
                self.assertEqual(sidecar.arch_id, "qwen3-next-mtp")
                self.assertEqual(sidecar.mtp_depth_max, 3)
                self.assertEqual(sidecar.base_trunk, "mlx-community/Qwen3.6-27B-4bit")
            finally:
                ctx.__exit__(None, None, None)

    def test_detect_prefers_matching_base_trunk(self) -> None:
        with TemporaryDirectory() as hf_cache, TemporaryDirectory() as dd_home:
            self._make_fake_cache(Path(hf_cache), with_sidecar=True, base_trunk="mlx-community/Qwen3.6-35B-A3B-4bit")
            ctx, mtp = self._isolated_env(hf_cache, dd_home)
            try:
                # Ask for a different trunk — falls back to first found.
                sidecar = mtp.detect_mtp_sidecar("mlx-community/Qwen3.6-27B-4bit")
                self.assertIsNotNone(sidecar)
                assert sidecar is not None
                self.assertEqual(sidecar.base_trunk, "mlx-community/Qwen3.6-35B-A3B-4bit")
            finally:
                ctx.__exit__(None, None, None)

    def test_read_runtime_json_rejects_invalid(self) -> None:
        with TemporaryDirectory() as tmp:
            snap = Path(tmp)
            # Empty file → no sidecar.
            (snap / "mtplx_runtime.json").write_text("")
            self.assertIsNone(_read_runtime_json(snap))
            # Valid JSON but missing required keys → no sidecar.
            (snap / "mtplx_runtime.json").write_text(json.dumps({"foo": "bar"}))
            self.assertIsNone(_read_runtime_json(snap))


class MTPBackendCapabilityTests(unittest.TestCase):
    """The capability check is the single source of truth used by the
    chat REPL + CLI to decide whether to actually try MTP. Test the
    matrix of (mtplx-installed, sidecar-present) directly."""

    def test_no_mtplx_no_sidecar(self) -> None:
        backend = MTPBackend(object(), object(), model_ref="x")
        with patch("daydream.backends.mtp.mtplx_available", return_value=False), \
             patch("daydream.backends.mtp.detect_mtp_sidecar", return_value=None):
            caps = backend.capabilities()
        self.assertFalse(caps.supports_speculative)
        self.assertIn("mtplx", caps.notes.lower())

    def test_no_mtplx_with_sidecar(self) -> None:
        backend = MTPBackend(object(), object(), model_ref="x")
        fake = MTPSidecar(
            runtime_json_path=Path("/tmp/x/mtplx_runtime.json"),
            model_dir=Path("/tmp/x"),
            sidecar_repo="repo",
            base_trunk="trunk",
            mtp_depth_max=3,
            arch_id="qwen3-next-mtp",
        )
        with patch("daydream.backends.mtp.mtplx_available", return_value=False), \
             patch("daydream.backends.mtp.detect_mtp_sidecar", return_value=fake):
            caps = backend.capabilities()
        self.assertFalse(caps.supports_speculative)
        self.assertIn("mtplx", caps.notes.lower())

    def test_mtplx_present_no_sidecar(self) -> None:
        backend = MTPBackend(object(), object(), model_ref="x")
        with patch("daydream.backends.mtp.mtplx_available", return_value=True), \
             patch("daydream.backends.mtp.detect_mtp_sidecar", return_value=None):
            caps = backend.capabilities()
        self.assertFalse(caps.supports_speculative)
        # Tells the user how to install a sidecar.
        self.assertIn("daydream pull", caps.notes)

    def test_mtplx_present_with_sidecar(self) -> None:
        backend = MTPBackend(object(), object(), model_ref="x")
        fake = MTPSidecar(
            runtime_json_path=Path("/tmp/x/mtplx_runtime.json"),
            model_dir=Path("/tmp/x"),
            sidecar_repo="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            base_trunk="mlx-community/Qwen3.6-27B-4bit",
            mtp_depth_max=3,
            arch_id="qwen3-next-mtp",
        )
        with patch("daydream.backends.mtp.mtplx_available", return_value=True), \
             patch("daydream.backends.mtp.detect_mtp_sidecar", return_value=fake):
            caps = backend.capabilities()
        self.assertTrue(caps.supports_speculative)
        self.assertEqual(caps.diagnostic["sidecar_repo"], fake.sidecar_repo)


class KnownSidecarsTest(unittest.TestCase):
    def test_recommended_pull_lists_at_least_one_repo(self) -> None:
        self.assertTrue(len(KNOWN_MTP_SIDECAR_REPOS) >= 1)
        self.assertTrue(all("/" in r for r in KNOWN_MTP_SIDECAR_REPOS))


if __name__ == "__main__":
    unittest.main()
