# Copyright © 2026 Apple Inc.

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlx._distributed_utils import launch
from mlx._distributed_utils.common import Host


@unittest.skipUnless(os.name == "posix", "The launcher uses POSIX pipes and signals")
class TestLaunchExitStatus(unittest.TestCase):
    def run_launcher(self, arguments, setup=""):
        # Match the console-script entry point, including sys.exit(main()).
        source = (
            "import sys\n"
            "from mlx._distributed_utils import launch\n"
            + textwrap.dedent(setup)
            + "\nsys.exit(launch.main())\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ, TMPDIR=tmpdir)
            proc = subprocess.Popen(
                [sys.executable, "-c", source, *arguments],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                self.fail("The launcher did not finish within 15 seconds")
            finally:
                # Remove any peers left behind by a failed launcher.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            return proc.returncode, stdout.decode(), stderr.decode()

    def run_local(self, source, ranks=1):
        return self.run_launcher(
            ["--backend", "ring", "-n", str(ranks), sys.executable, "-c", source]
        )

    def test_success(self):
        result, _, _ = self.run_local("pass", ranks=2)
        self.assertEqual(result, 0)

    def test_failure_codes(self):
        for code in (1, 7, 255):
            with self.subTest(code=code):
                result, _, stderr = self.run_local(f"raise SystemExit({code})")
                self.assertEqual(result, code, stderr)

    def test_signal_exit(self):
        result, _, stderr = self.run_local(
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
        )
        self.assertEqual(result, 128 + signal.SIGTERM, stderr)

    def test_successful_rank_does_not_stop_other_ranks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "completed"
            source = textwrap.dedent(
                f"""
                import os, time
                from pathlib import Path
                if os.environ["MLX_RANK"] == "1":
                    time.sleep(0.2)
                    Path({str(marker)!r}).write_text("done")
                """
            )
            result, _, stderr = self.run_local(source, ranks=2)
            self.assertEqual(result, 0, stderr)
            self.assertEqual(marker.read_text(), "done")

    def test_failure_code_is_not_replaced_by_peer_termination(self):
        # A higher-ranked failure must win over lower-ranked SIGTERM statuses.
        with tempfile.TemporaryDirectory() as tmpdir:
            source = textwrap.dedent(
                f"""
                import os, signal, time
                from pathlib import Path
                root = Path({tmpdir!r})
                rank = int(os.environ["MLX_RANK"])
                if rank == 2:
                    deadline = time.monotonic() + 5
                    while not all((root / f"ready-{{i}}").exists() for i in (0, 1)):
                        if time.monotonic() > deadline:
                            raise SystemExit(99)
                        time.sleep(0.01)
                    raise SystemExit(23)
                def on_term(signum, frame):
                    (root / f"stopped-{{rank}}").write_text("terminated")
                    raise SystemExit(128 + signum)
                signal.signal(signal.SIGTERM, on_term)
                (root / f"ready-{{rank}}").write_text("ready")
                time.sleep(30)
                """
            )
            result, _, stderr = self.run_local(source, ranks=3)
            self.assertEqual(result, 23, stderr)
            for rank in (0, 1):
                self.assertEqual(
                    (Path(tmpdir) / f"stopped-{rank}").read_text(), "terminated"
                )

    def test_failure_after_other_rank_succeeds(self):
        result, _, stderr = self.run_local(
            'import os, time; rank = int(os.environ["MLX_RANK"]); '
            "time.sleep(0.2 * rank); raise SystemExit(11 * rank)",
            ranks=2,
        )
        self.assertEqual(result, 11, stderr)

    def test_missing_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = str(Path(tmpdir) / "does-not-exist")
            result, _, stderr = self.run_launcher(["--backend", "ring", missing])
            self.assertEqual(result, 127, stderr)

    def test_worker_startup_failure(self):
        setup = """
            def fail_to_start(*args, **kwargs):
                raise OSError("injected launcher startup failure")
            launch.RemoteProcess = fail_to_start
        """
        result, _, stderr = self.run_launcher(
            ["--backend", "ring", sys.executable, "-c", "pass"], setup
        )
        self.assertIn("injected launcher startup failure", stderr)
        self.assertEqual(result, 1)

    def test_failure_before_first_supervisor_poll(self):
        # Run workers to completion before the supervisor inspects them.
        setup = """
            class FinishedProcess(launch.CommandProcess):
                def __init__(self, rank, *args, **kwargs):
                    self.p = launch.Popen(
                        [sys.executable, "-c", "raise SystemExit(17)"],
                        stdin=launch.PIPE, stdout=launch.PIPE, stderr=launch.PIPE,
                    )
                    self.p.wait()
                @property
                def process(self):
                    return self.p
                @property
                def exit_status(self):
                    return self.p.returncode, False
            class InlineThread:
                def __init__(self, target, args, kwargs):
                    self.target, self.args, self.kwargs = target, args, kwargs
                def start(self):
                    self.target(*self.args, **self.kwargs)
                def is_alive(self):
                    return False
                def join(self):
                    pass
            launch.RemoteProcess = FinishedProcess
            launch.threading.Thread = InlineThread
        """
        result, _, stderr = self.run_launcher(
            ["--backend", "ring", "-n", "2", sys.executable, "-c", "pass"],
            setup,
        )
        self.assertEqual(result, 17, stderr)

    def test_output_is_forwarded_on_failure(self):
        source = (
            "import sys, time; "
            "time.sleep(0.1); "
            "print('rank stdout', flush=True); "
            "print('rank stderr', file=sys.stderr, flush=True); "
            "time.sleep(0.1); raise SystemExit(9)"
        )
        result, stdout, stderr = self.run_local(source)
        self.assertIn("rank stdout", stdout)
        self.assertIn("rank stderr", stderr)
        self.assertEqual(result, 9)

    def test_remote_backend_adapters(self):
        hosts = [
            Host(0, "127.0.0.1", ["127.0.0.1"], [None, "rdma_en1"]),
            Host(1, "127.0.0.1", ["127.0.0.1"], ["rdma_en1", None]),
        ]
        for backend in ("ring", "nccl", "jaccl", "jaccl-ring"):
            for code in (0, 23):
                with self.subTest(backend=backend, code=code):
                    args = SimpleNamespace(
                        backend=backend,
                        starting_port=32323,
                        connections_per_ip=1,
                        env=[],
                        verbose=False,
                        cwd=None,
                        python=None,
                        nccl_port=12345,
                        repeat_hosts=1,
                    )
                    adapter = getattr(launch, "launch_" + backend.split("-")[0])
                    with patch.object(launch, "_launch_with_io", return_value=code):
                        self.assertEqual(
                            adapter(None, hosts, args, ["script.py"]), code
                        )

    def test_main_returns_backend_status(self):
        for backend in ("ring", "nccl", "jaccl", "jaccl-ring", "mpi"):
            for code in (0, 23):
                with self.subTest(backend=backend, code=code):
                    name = "launch_" + backend.split("-")[0]
                    with (
                        patch.object(launch, name, return_value=code),
                        patch.object(
                            sys,
                            "argv",
                            ["mlx.launch", "--backend", backend, "script.py"],
                        ),
                    ):
                        self.assertEqual(launch.main(), code)

    def test_mpi_exit_status(self):
        hosts = [Host(0, "127.0.0.1", ["127.0.0.1"], [])]
        for code, expected in ((0, 0), (7, 7), (255, 255), (-15, 143)):
            with self.subTest(code=code):
                args = SimpleNamespace(env=[], verbose=False, cwd=None, mpi_arg=[])
                results = [
                    SimpleNamespace(stdout=b"/usr/bin/mpirun\n"),
                    SimpleNamespace(returncode=code),
                ]
                with (
                    patch.object(launch, "get_mpi_libname", return_value=None),
                    patch.object(launch, "run", side_effect=results),
                ):
                    self.assertEqual(
                        launch.launch_mpi(None, hosts, args, ["script.py"]), expected
                    )

    def test_mpi_interrupt(self):
        hosts = [Host(0, "127.0.0.1", ["127.0.0.1"], [])]
        args = SimpleNamespace(env=[], verbose=False, cwd=None, mpi_arg=[])
        with (
            patch.object(launch, "get_mpi_libname", return_value=None),
            patch.object(
                launch,
                "run",
                side_effect=[
                    SimpleNamespace(stdout=b"/usr/bin/mpirun\n"), KeyboardInterrupt()
                ],
            ),
        ):
            self.assertEqual(launch.launch_mpi(None, hosts, args, ["script.py"]), 130)

    def test_print_python(self):
        result, stdout, _ = self.run_launcher(["--print-python"])
        self.assertEqual(result, 0)
        self.assertEqual(stdout.strip(), sys.executable)

    def test_unknown_backend(self):
        result, _, stderr = self.run_launcher(["--backend", "invalid", "script.py"])
        self.assertEqual(result, 2)
        self.assertIn("The backend should be one of", stderr)


if __name__ == "__main__":
    unittest.main()
