"""Custom slots select isolated builds without replacing files or presets."""

import json
import os
import tempfile
import unittest
from unittest import mock

from backend import app
from backend.http import Request
from backend.routes import install, lifecycle, status
from backend.services import llama_manager, process_manager
from tests.backend.test_extracted_routes import DummyResponse
from tests.backend.test_services import make_service_context


class CustomSlotsTests(unittest.TestCase):
    def setUp(self):
        opt_out = mock.patch.dict(os.environ, {"LLAMA_GUI_SKIP_LDD": ""})
        opt_out.start()
        self.addCleanup(opt_out.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = make_service_context(self.tmp.name)
        services = self.ctx.services
        services.current_platform = "win32"
        services.current_arch = "x64"
        services.binary_suffix = ".exe"
        services.llama_tools = ["llama-cli", "llama-server", "llama-bench", "llama-perplexity"]
        services.get_tool_filename = lambda tool: tool + services.binary_suffix
        services.backend_specs = llama_manager.build_backend_specs("win32", "x64")
        services.load_config = lambda: json.loads(self.ctx.paths.config_file.read_text())
        services.save_config = lambda cfg: self.ctx.paths.config_file.write_text(json.dumps(cfg))
        services.find_tool_executable = lambda tool: (
            llama_manager.get_backend_bin_dir(self.ctx, services.load_config().get("backend"))
            / services.get_tool_filename(tool)
        )
        services.get_runtime_files = lambda: []
        services.get_platform_label = lambda: "Windows"
        services.get_llama_api_target = lambda: {"host": "127.0.0.1", "port": 8080}
        services.validate_runtime_dependencies = lambda tools=None: llama_manager.validate_runtime_dependencies(self.ctx, tools)
        services.save_config({"backend": "cpu", "tag": "b123", "version": "b123", "models_dir": "saved-model-root"})
        for backend in ("cpu", "custom", "custom-02"):
            self.write_build(backend)

    def write_build(self, backend):
        directory = llama_manager.get_backend_bin_dir(self.ctx, backend)
        directory.mkdir(parents=True, exist_ok=True)
        for tool in ("llama-cli", "llama-server"):
            (directory / self.ctx.services.get_tool_filename(tool)).write_text(backend)
        (directory / "ggml.dll").write_text(backend)
        grammars = llama_manager.get_backend_grammars_dir(self.ctx, backend)
        grammars.mkdir(parents=True, exist_ok=True)
        (grammars / "local.gbnf").write_text(backend)
        return directory

    def activate(self, backend="custom"):
        response = DummyResponse()
        install.activate_custom(Request("POST", "/api/activate-custom", "", {}, body={"backend": backend}), response, self.ctx)
        return response

    def test_switch_round_trip_persists_identity_and_keeps_both_builds(self):
        original = {p: p.read_bytes() for p in self.ctx.paths.llama.rglob("*") if p.is_file()}
        for backend in ("custom", "custom-02"):
            with self.subTest(backend=backend):
                self.ctx.state.runtime_health_cache[("old", ())] = (0, {"ok": True})
                result = self.activate(backend)
                self.assertTrue(result.payload["ok"])
                self.assertIn("llama-bench.exe", result.payload["missing"])
                self.assertFalse(self.ctx.state.install_in_progress)
                self.assertEqual(self.ctx.state.runtime_health_cache, {})
                # The real app resolver reads the saved config, as it does after restart.
                with mock.patch.object(app, "APP_CONTEXT", self.ctx), mock.patch.object(app, "CONFIG_FILE", self.ctx.paths.config_file), mock.patch.object(app, "BINARY_SUFFIX", ".exe"):
                    self.assertEqual(app.load_config()["backend"], backend)
                    self.assertEqual(app.find_tool_executable("llama-server").read_text(), backend)
                    self.assertEqual([p.read_text() for p in app.get_runtime_files()], [backend])
                cfg = self.ctx.services.load_config()
                self.assertEqual(cfg["tag"], "custom")
                self.assertEqual(cfg["official_install"]["backend"], "cpu")
                self.assertEqual(cfg["models_dir"], "saved-model-root")
        with mock.patch.object(llama_manager, "_probe_official_build", return_value=(True, "b123")):
            result = llama_manager.activate_official_backend(self.ctx, "cpu")
        self.assertTrue(result["ok"])
        self.assertEqual(self.ctx.services.load_config()["tag"], "b123")
        self.assertTrue(self.activate().payload["ok"])
        self.assertEqual(original, {p: p.read_bytes() for p in self.ctx.paths.llama.rglob("*") if p.is_file()})

    def test_missing_tool_in_second_slot_never_falls_back_or_changes_config(self):
        self.activate()
        previous = self.ctx.paths.config_file.read_bytes()
        missing = self.ctx.paths.llama / "custom-02" / "bin" / "llama-server.exe"
        missing.unlink()
        missing.mkdir()  # A directory with an executable name is not a tool.
        result = self.activate("custom-02")
        self.assertFalse(result.payload["ok"])
        self.assertEqual(result.payload["missing_required"], ["llama-server.exe"])
        self.assertEqual(self.ctx.paths.config_file.read_bytes(), previous)
        self.assertFalse(self.ctx.state.install_in_progress)

    def test_linux_validation_and_launch_environment_use_selected_slot(self):
        self.ctx.services.current_platform = "linux"
        self.ctx.services.binary_suffix = ""
        directory = self.write_build("custom-02")
        plugin = directory / "libggml-cuda.so"
        plugin.write_text("plugin")
        with mock.patch.object(llama_manager.os, "access", return_value=True), mock.patch.object(
            llama_manager, "get_linux_missing_libraries", side_effect=lambda path, _directory: ["libcudart.so"] if path == plugin else []
        ) as probe:
            previous = self.ctx.paths.config_file.read_bytes()
            result = self.activate("custom-02")
            self.assertFalse(result.payload["ok"])
            self.assertEqual(result.payload["missing_runtime_files"], ["libcudart.so"])
            self.assertTrue(all(call.args[0].parent == directory and call.args[1] == directory for call in probe.call_args_list))
            self.assertEqual(self.ctx.paths.config_file.read_bytes(), previous)
            probe.side_effect = None
            probe.return_value = []
            self.assertTrue(self.activate("custom-02").payload["ok"])
            probe.reset_mock()
            self.assertTrue(llama_manager.validate_runtime_dependencies(self.ctx)["ok"])
            self.assertTrue(all(call.args[1] == directory for call in probe.call_args_list))
            executable, error = process_manager._validate_launch_environment(self.ctx, "llama-server")
            self.assertIsNone(error)
            self.assertEqual(executable, directory / "llama-server")
        with mock.patch.dict(os.environ, {"PATH": "system-path", "LD_LIBRARY_PATH": "system-libs"}):
            env = process_manager._build_process_env(self.ctx)
        self.assertEqual(env["PATH"], str(directory) + os.pathsep + "system-path")
        self.assertEqual(env["LD_LIBRARY_PATH"], str(directory) + os.pathsep + "system-libs")
        self.assertEqual(process_manager._fit_params_executable(self.ctx), directory / "llama-fit-params")

    def test_linux_ldd_opt_out_allows_status_activation_preflight_and_launch(self):
        self.ctx.services.current_platform = "linux"
        self.ctx.services.binary_suffix = ""
        self.ctx.services.normalize_llama_api_target = lambda host, port: {"host": host, "port": int(port)}
        model = self.ctx.paths.root / "model.gguf"
        model.write_text("model")
        for backend in ("cpu", "custom", "custom-02"):
            with self.subTest(backend=backend):
                directory = self.write_build(backend)
                (directory / "libggml-vulkan.so").write_text("plugin")
                self.ctx.services.save_config({"backend": backend, "tag": "b123"})
                fake_process = mock.Mock(pid=1234)
                fake_process.poll.return_value = None
                with mock.patch.dict(os.environ, {"LLAMA_GUI_SKIP_LDD": "1"}), mock.patch.object(
                    llama_manager.os, "access", return_value=True
                ), mock.patch.object(llama_manager.subprocess, "run") as probe, mock.patch.object(
                    process_manager.subprocess, "Popen", return_value=fake_process
                ) as popen, mock.patch.object(process_manager.threading, "Thread"):
                    if backend != "cpu":
                        activated = self.activate(backend)
                        self.assertTrue(activated.payload["ok"])
                        self.assertFalse(activated.payload["runtime_health"]["checked"])
                        self.assertEqual(activated.payload["runtime_health"]["skip_reason"], "LLAMA_GUI_SKIP_LDD")
                    response = DummyResponse()
                    status.get_status(Request("GET", "/api/status", "", {}), response, self.ctx)
                    self.assertTrue(response.payload["installed"])
                    self.assertFalse(response.payload["config_stale"])
                    self.assertFalse(response.payload["runtime_health"]["checked"])
                    args = ["-m", str(model)]
                    preflight = process_manager.preflight_launch(self.ctx, "llama-server", args, {})
                    self.assertTrue(preflight["ok"])
                    result = process_manager.launch_process(self.ctx, "llama-server", args)
                    self.assertEqual(result["pid"], 1234)
                    self.assertEqual(popen.call_args.args[0], [str(directory / "llama-server"), *args])
                    probe.assert_not_called()
                fake_process.poll.return_value = 0

    def test_linux_ldd_opt_out_keeps_missing_tool_and_permission_guards(self):
        self.ctx.services.current_platform = "linux"
        self.ctx.services.binary_suffix = ""
        for backend in ("cpu", "custom", "custom-02"):
            with self.subTest(backend=backend), mock.patch.dict(
                os.environ, {"LLAMA_GUI_SKIP_LDD": "1"}
            ), mock.patch.object(llama_manager.subprocess, "run") as probe, mock.patch.object(
                process_manager.subprocess, "Popen"
            ) as popen:
                directory = self.write_build(backend)
                self.ctx.services.save_config({"backend": backend, "tag": "b123"})
                executable = directory / "llama-server"
                executable.unlink()
                result = process_manager.preflight_launch(self.ctx, "llama-server", [], {})
                self.assertIn("not found", result["error"])
                self.assertIn("not found", process_manager.launch_process(self.ctx, "llama-server", [])["error"])
                if backend != "cpu":
                    self.assertEqual(self.activate(backend).payload["missing_required"], ["llama-server"])
                response = DummyResponse()
                status.get_status(Request("GET", "/api/status", "", {}), response, self.ctx)
                self.assertFalse(response.payload["installed"])
                executable.write_text("binary")
                with mock.patch.object(llama_manager.os, "access", return_value=False):
                    result = process_manager.preflight_launch(self.ctx, "llama-server", [], {})
                    self.assertIn("not executable", result["error"])
                    self.assertIn("not executable", process_manager.launch_process(self.ctx, "llama-server", [])["error"])
                    if backend != "cpu":
                        self.assertEqual(len(self.activate(backend).payload["not_executable"]), 2)
                probe.assert_not_called()
                popen.assert_not_called()

    def test_linux_ldd_opt_out_preserves_subprocess_launch_errors(self):
        self.ctx.services.current_platform = "linux"
        self.ctx.services.binary_suffix = ""
        self.ctx.services.normalize_llama_api_target = lambda host, port: {"host": host, "port": int(port)}
        self.write_build("cpu")
        with mock.patch.dict(os.environ, {"LLAMA_GUI_SKIP_LDD": "1"}), mock.patch.object(
            llama_manager.os, "access", return_value=True
        ), mock.patch.object(llama_manager.subprocess, "run") as probe, mock.patch.object(
            process_manager.subprocess, "Popen", side_effect=OSError("runtime loader failed")
        ) as popen:
            result = process_manager.launch_process(self.ctx, "llama-server", [])
        self.assertEqual(result["error"], "runtime loader failed")
        self.assertIsNone(self.ctx.state.process)
        popen.assert_called_once()
        probe.assert_not_called()

    def test_macos_missing_runtime_and_permissions_leave_original_active(self):
        self.ctx.services.current_platform = "darwin"
        self.ctx.services.binary_suffix = ""
        directory = self.write_build("custom-02")
        library = "libllama.0.dylib"
        (self.ctx.paths.llama_custom_bin / library).write_text("other slot")
        with mock.patch.object(llama_manager.os, "access", return_value=False):
            result = self.activate("custom-02")
        self.assertFalse(result.payload["ok"])
        self.assertEqual(len(result.payload["not_executable"]), 2)
        with mock.patch.object(llama_manager.os, "access", return_value=True), mock.patch.object(
            llama_manager, "get_macos_rpath_libraries", return_value=[library]
        ):
            result = self.activate("custom-02")
            self.assertFalse(result.payload["ok"])
            self.assertEqual(result.payload["missing_runtime_files"], [library])
            self.assertEqual(self.ctx.services.load_config()["backend"], "cpu")
            (directory / library).write_text("selected slot")
            self.assertTrue(self.activate("custom-02").payload["ok"])

    def test_routes_reject_invalid_slots_busy_installs_and_running_process(self):
        for invalid in ("cpu", "custom-03", "../outside", None, [], {}):
            with self.subTest(backend=invalid):
                self.assertEqual(self.activate(invalid).status, 400)
                self.assertFalse(self.ctx.state.install_in_progress)
        self.ctx.state.install_in_progress = True
        self.assertEqual(self.activate("custom-02").status, 409)
        self.ctx.state.install_in_progress = False
        with mock.patch.object(process_manager, "is_process_running", return_value=True):
            result = self.activate("custom-02")
        self.assertEqual(result.status, 400)
        self.assertIn("Stop running process", result.payload["error"])
        self.assertEqual(self.ctx.services.load_config()["backend"], "cpu")

    def test_second_slot_is_excluded_from_official_download_and_update_routes(self):
        self.activate("custom-02")
        with mock.patch.object(llama_manager, "get_releases") as lookup, mock.patch.object(llama_manager, "install_release") as download:
            response = DummyResponse()
            install.get_releases(Request("GET", "/api/releases", "backend=custom-02", {}), response, self.ctx)
            self.assertEqual(response.payload, [])
            for route, path, body in (
                (install.start_install, "/api/install", {"backend": "custom-02", "tag": "custom"}),
                (install.start_update, "/api/update", {}),
            ):
                response = DummyResponse()
                route(Request("POST", path, "", {}, body=body), response, self.ctx)
                self.assertEqual(response.status, 400)
            lookup.assert_not_called()
            download.assert_not_called()

    def test_status_folder_opening_and_cleanup_preserve_custom_selection(self):
        self.activate("custom-02")
        response = DummyResponse()
        status.get_status(Request("GET", "/api/status", "", {}), response, self.ctx)
        self.assertTrue(response.payload["installed"])
        self.assertEqual(response.payload["backend"], "custom-02")
        slots = [item for item in response.payload["available_backends"] if item.get("custom")]
        self.assertEqual(slots, [
            {"id": "custom", "label": "Custom", "custom": True, "bin_dir": "llama/custom/bin/"},
            {"id": "custom-02", "label": "Custom 02", "custom": True, "bin_dir": "llama/custom-02/bin/"},
        ])
        with mock.patch.object(lifecycle.lifecycle_service, "open_folder_in_file_manager") as open_folder:
            lifecycle.post_open_folder(Request("POST", "/api/open-folder", "", {}, body={"folder": "llama"}), DummyResponse(), self.ctx)
        open_folder.assert_called_once_with(self.ctx.paths.llama / "custom-02")
        self.assertGreater(process_manager.remove_llama_files(self.ctx), 0)
        cfg = self.ctx.services.load_config()
        self.assertEqual(cfg["backend"], "custom-02")
        self.assertNotIn("official_install", cfg)
        for backend in ("custom", "custom-02"):
            self.assertEqual((llama_manager.get_backend_bin_dir(self.ctx, backend) / "llama-server.exe").read_text(), backend)
            self.assertEqual((llama_manager.get_backend_grammars_dir(self.ctx, backend) / "local.gbnf").read_text(), backend)
