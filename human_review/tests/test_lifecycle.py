"""Server lifecycle ownership and readiness, without loading the full catalog."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


_SPEC = importlib.util.spec_from_file_location("review_lifecycle", Path(__file__).resolve().parents[1] / "review.py")
review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review)


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"state_dir": "review_state"}))
        self.metadata, self.log = review.state_paths(self.config)
        self.record = {"pid": 12345, "config": str(self.config.resolve()), "host": "127.0.0.1",
                       "port": 8765, "script": str(Path(review.__file__).resolve()), "proc_start_ticks": "99"}

    def save(self):
        review._save_metadata(self.metadata, self.record)

    def test_parser_foreground_and_mutually_exclusive_lifecycle(self):
        parser = review.build_parser()
        args = parser.parse_args([])
        self.assertEqual((args.host, args.port), ("127.0.0.1", 8765))
        self.assertFalse(args.background)
        self.assertEqual(parser.parse_args(["--port", "0"]).port, 0)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--background", "--stop"])

    def test_state_paths_are_relative_to_config(self):
        self.assertEqual(self.metadata, self.root / "review_state" / "server.json")
        self.assertEqual(self.log, self.root / "review_state" / "server.log")

    def test_ownership_rejects_actual_unrelated_pid(self):
        self.record["pid"] = os.getpid()
        self.assertFalse(review._owned(self.record, self.config))

    def test_ownership_rejects_recycled_pid_and_wrong_cli(self):
        with patch.object(review, "_alive", return_value=True), patch.object(review, "_process_state", return_value={"start_ticks": "100"}):
            self.assertFalse(review._owned(self.record, self.config))
        self.record["proc_start_ticks"] = "99"
        argv = ["python", "-u", str(Path(review.__file__).resolve()), "--config", str(self.config),
                "--host", "127.0.0.1", "--port", "9999"]
        with patch.object(review, "_alive", return_value=True), patch.object(review, "_process_state", return_value={"start_ticks": "99"}), patch.object(Path, "read_bytes", return_value=b"\0".join(x.encode() for x in argv) + b"\0"):
            self.assertFalse(review._owned(self.record, self.config))

    def test_ownership_accepts_exact_foreground_command(self):
        argv = ["python", "-u", str(Path(review.__file__).resolve()), "--config", str(self.config),
                "--host", "127.0.0.1", "--port", "8765"]
        with patch.object(review, "_alive", return_value=True), patch.object(review, "_process_state", return_value={"start_ticks": "99"}), patch.object(Path, "read_bytes", return_value=b"\0".join(x.encode() for x in argv) + b"\0"):
            self.assertTrue(review._owned(self.record, self.config))

    def test_unknown_pid_is_never_stopped_or_replaced(self):
        self.save()
        before = self.metadata.read_bytes()
        with patch.object(review, "_alive", return_value=True), patch.object(review, "_owned", return_value=False), patch.object(review.os, "kill") as kill, patch.object(review.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "another process"):
                review.background(self.config)
            with self.assertRaisesRegex(ValueError, "Nothing was stopped"):
                review.stop(self.config)
            self.assertEqual(review.status(self.config)["state"], "foreign")
            kill.assert_not_called()
            popen.assert_not_called()
        self.assertEqual(self.metadata.read_bytes(), before)

    def test_occupied_port_is_never_taken_over(self):
        with patch.object(review, "_port_occupied", return_value=True), patch.object(review.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "occupied"):
                review.background(self.config)
            popen.assert_not_called()

    def test_existing_owned_server_is_reused(self):
        self.save()
        with patch.object(review, "_alive", return_value=True), patch.object(review, "_owned", return_value=True), patch.object(review, "_health", return_value=True), patch.object(review.subprocess, "Popen") as popen:
            result = review.background(self.config)
            self.assertTrue(result["reused"])
            self.assertEqual(result["pid"], 12345)
            self.assertEqual(review.status(self.config)["state"], "running")
            popen.assert_not_called()
            with self.assertRaisesRegex(ValueError, "changing its address"):
                review.background(self.config, port=8888)

    def test_background_detaches_logs_and_records_ready_process(self):
        process = Mock(pid=22222)
        process.poll.return_value = None
        with patch.object(review, "_port_occupied", return_value=False), patch.object(review, "_process_state", return_value={"start_ticks": "123"}), patch.object(review, "_owned", return_value=True), patch.object(review, "_health", return_value=True), patch.object(review.subprocess, "Popen", return_value=process) as popen:
            result = review.background(self.config)
        args, kwargs = popen.call_args
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(kwargs["stdin"], review.subprocess.DEVNULL)
        self.assertEqual(args[0][-4:], ["--host", "127.0.0.1", "--port", "8765"])
        self.assertFalse(result["reused"])
        metadata = json.loads(self.metadata.read_text())
        self.assertEqual(metadata["pid"], 22222)
        self.assertEqual(metadata["proc_start_ticks"], "123")
        self.assertTrue(self.log.is_file())

    def test_failed_start_clears_own_metadata(self):
        process = Mock(pid=22222)
        process.poll.return_value = 1
        with patch.object(review, "_port_occupied", return_value=False), patch.object(review, "_process_state", return_value=None), patch.object(review.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                review.background(self.config)
        self.assertFalse(self.metadata.exists())

    def test_timeout_terminates_only_new_child(self):
        process = Mock(pid=22222)
        process.poll.return_value = None
        with patch.object(review, "_port_occupied", return_value=False), patch.object(review, "_process_state", return_value=None), patch.object(review.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                review.background(self.config, readiness_timeout=0)
        process.terminate.assert_called_once()
        self.assertFalse(self.metadata.exists())

    def test_stop_requires_owned_process_and_keeps_labels(self):
        self.save()
        labels = self.metadata.parent / "labels.sqlite3"
        labels.write_bytes(b"saved-review")
        with patch.object(review, "_alive", side_effect=[True, False, False]), patch.object(review, "_owned", return_value=True), patch.object(review.os, "kill") as kill:
            self.assertEqual(review.stop(self.config)["state"], "stopped")
        kill.assert_called_once_with(12345, review.signal.SIGTERM)
        self.assertFalse(self.metadata.exists())
        self.assertEqual(labels.read_bytes(), b"saved-review")

    def test_absent_and_dead_pid_are_idempotently_stopped(self):
        self.assertEqual(review.stop(self.config)["state"], "stopped")
        self.save()
        with patch.object(review, "_alive", return_value=False), patch.object(review.os, "kill") as kill:
            self.assertEqual(review.stop(self.config)["state"], "stopped")
            kill.assert_not_called()
        self.assertFalse(self.metadata.exists())


if __name__ == "__main__":
    unittest.main()
