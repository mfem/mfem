#!/usr/bin/env python3
"""Runner unit/integration tests; the fake driver never starts MPI."""

import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import xml.etree.ElementTree as ElementTree

import benchmark_2d as benchmark


def driver_row(method="pspg", solver="fgmres", preconditioner="block-lower-p1-amg"):
    row = {field: 0 for field in benchmark.INTEGER_FIELDS + benchmark.FLOAT_FIELDS}
    row.update(dimension=2, level=0, periodic=0, internal_interface_bcs=0,
               method=method, solver=solver, preconditioner=preconditioner,
               mms_case="smooth", pressure_gauge="auto", pressure_nullspace_detected=1,
               pressure_gauge_active=1, assembly_performed=1, geometry_order=1,
               integration_order=6, order=2, pressure_order=1 if method == "taylor-hood" else 2,
               elements=7952, velocity_dofs=32408, pressure_dofs=16204,
               converged=1, outer_iterations=0 if solver == "mumps" else 48,
               absolute_residual=1e-13, relative_residual=1e-13, hole_velocity_l2=2.7,
               velocity_l2=0.00018, velocity_h1=0.026, pressure_l2=0.0036,
               divergence_l2=0.009, boundary_l2_error=7e-6, boundary_dof_error=1e-13,
               boundary_flux=1e-15, gauge_multiplier=1e-10,
               assembly_seconds=0.1, preconditioner_setup_seconds=0 if solver == "mumps" else 0.02,
               direct_setup_seconds=0.2 if solver == "mumps" else 0,
               linear_solve_seconds=0.1, finalization_seconds=0.001, solve_seconds=0.5,
               boundary_projection_seconds=0.01)
    return row


def fake_driver():
    arguments = sys.argv[1:]
    def option(name):
        return arguments[arguments.index("--" + name) + 1]
    output = Path(option("csv"))
    mode = os.environ.get("STOKES_TEST_FAKE_MODE", "success")
    if mode == "sleep":
        child = subprocess.Popen([sys.executable, "-c",
                                  "import sys,time\nfrom pathlib import Path\n"
                                  "while True:\n Path(sys.argv[1]).write_text(str(time.monotonic()))\n time.sleep(0.02)",
                                  str(output.with_suffix(".heartbeat"))])
        output.with_suffix(".childpid").write_text(str(child.pid))
        time.sleep(60)
    if mode == "unavailable" and option("solver") == "mumps":
        print("MUMPS is unavailable", file=sys.stderr)
        return 1
    row = driver_row(option("method"), option("solver"), option("preconditioner"))
    refinement = int(option("refine-serial"))
    for field, order in benchmark.REFINEMENT_ORDERS.items():
        row[field] *= 2 ** ((1 - refinement) * order)
    for field in ("elements", "velocity_dofs", "pressure_dofs"):
        row[field] = int(row[field] * 4 ** (refinement - 1))
    for field in benchmark.TIMINGS[:-1]:
        row[field] *= 4 ** (refinement - 1)
    if option("solver") != "mumps":
        row["outer_iterations"] += (refinement - 1) * 6
    if mode == "nonconverged" or int(option("max-iterations")) == 1:
        row.update(converged=0, outer_iterations=int(option("max-iterations")))
    if mode == "accuracy":
        row["pressure_l2"] = 1
    if mode == "nonfinite":
        row["linear_solve_seconds"] = "nan"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print("fake solver completed")
    return 2 if not row["converged"] else 0


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.mesh = self.root / "mesh with spaces.mesh"
        self.mesh.write_text("MFEM mesh v1.0\n\ndimension\n2\n\nelements\n1988\n")
        self.output = self.root / "study"
        self.arguments = ["run", "--output", str(self.output), "--executable", str(Path(__file__).resolve()),
                          "--mesh", str(self.mesh), "--launcher", sys.executable,
                          "--methods", "pspg", "--solvers", "fgmres",
                          "--preconditioners", "block-lower-p1-amg", "--repetitions", "2"]
        self.environment = os.environ.copy()
        self.environment.pop("STOKES_TEST_FAKE_MODE", None)
        self.settings = {"ranks": 4, "relative_tolerance": 1e-12, "absolute_tolerance": 1e-14,
                         "restart": 100, "max_iterations": 2000, "accuracy_bounds": benchmark.BOUNDS,
                         "minimum_hole_velocity_l2": 0.01, "quadrature_order": 6}
        self.case = {"id": "example", "method": "pspg", "solver": "fgmres", "preconditioner": "block-lower-p1-amg"}

    def manifest(self, configurations=None, repetitions=2):
        configurations = configurations or [self.case]
        record = {
            "schema_version": benchmark.SCHEMA_VERSION, "settings": self.settings,
            "configurations": configurations,
            "protocol": {"timeout_seconds": 10, "execution_class": "exploratory"},
            "provenance": {"executable": {"path": str(Path(__file__).resolve()),
                                          "sha256": benchmark.file_hash(__file__)},
                           "mesh": {"path": str(self.mesh)}, "launcher": {"tokens": [sys.executable]}},
            "schedule": benchmark.schedule(configurations, 1, repetitions, 123),
        }
        record["fingerprint"] = benchmark.digest(record)
        return record

    def run_quietly(self, manifest, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return benchmark.run_study(self.output, manifest, self.environment, **kwargs)

    def test_matrix_and_schedule(self):
        configurations = benchmark.configuration_matrix(self.settings, benchmark.METHODS, benchmark.SOLVERS, benchmark.PRECONDITIONERS)
        self.assertEqual(len(configurations), 60)
        self.assertEqual(len({case["id"] for case in configurations}), 60)
        self.assertEqual(sum(case["solver"] == "mumps" for case in configurations), 4)
        jobs = benchmark.schedule(configurations, 1, 3, 7)
        self.assertEqual(len(jobs), 240)
        self.assertEqual(sum(not job["warmup"] for job in jobs), 180)
        self.assertEqual(len({(job["configuration_id"], job["sample_id"]) for job in jobs}), 240)
        self.assertEqual(jobs, benchmark.schedule(configurations, 1, 3, 7))
        self.assertNotEqual(jobs, benchmark.schedule(configurations, 1, 3, 8))
        for start in range(0, 240, 60):
            self.assertEqual({job["configuration_id"] for job in jobs[start:start+60]}, {case["id"] for case in configurations})

    def test_parameter_change_changes_case_id(self):
        first = benchmark.configuration_matrix(self.settings, ["pspg"], ["fgmres"], benchmark.PRECONDITIONERS)
        changed = {**self.settings, "max_iterations": 1}
        second = benchmark.configuration_matrix(changed, ["pspg"], ["fgmres"], benchmark.PRECONDITIONERS)
        self.assertNotEqual(first[0]["id"], second[0]["id"])

    def refinement_manifest(self, solvers=("fgmres", "mumps")):
        self.settings.update(refinement_levels=[0, 1, 2], input_elements=1988,
                             accuracy_bounds_by_refinement={str(level): benchmark.refinement_bounds(level)
                                                           for level in (0, 1, 2)})
        configurations = benchmark.configuration_matrix(self.settings, ["pspg"], solvers, ["block-lower-p1-amg"])
        return self.manifest(configurations, repetitions=1)

    def test_refinement_matrix_and_shuffled_rounds(self):
        settings = {**self.settings, "refinement_levels": [0, 1, 2]}
        configurations = benchmark.configuration_matrix(settings, benchmark.METHODS, benchmark.SOLVERS, benchmark.PRECONDITIONERS)
        self.assertEqual(len(configurations), 180)
        self.assertEqual(len({case["id"] for case in configurations}), 180)
        self.assertEqual(sum(case["solver"] == "mumps" for case in configurations), 12)
        jobs = benchmark.schedule(configurations, 1, 3, 20261101)
        self.assertEqual(len(jobs), 720)
        for start in range(0, 720, 180):
            self.assertEqual({job["configuration_id"] for job in jobs[start:start + 180]},
                             {case["id"] for case in configurations})

    def test_level_specific_bounds_and_element_counts(self):
        manifest = self.refinement_manifest()
        for case in manifest["configurations"]:
            effective = benchmark.case_settings(case, manifest["settings"])
            refinement = case["serial_refinements"]
            self.assertEqual(effective["expected_elements"], 1988 * 4 ** refinement)
            self.assertEqual(effective["accuracy_bounds"]["velocity_l2"], 0.001 * 8 ** (1 - refinement))
            self.assertEqual(effective["accuracy_bounds"]["pressure_l2"], 0.02 * 4 ** (1 - refinement))
            self.assertEqual(effective["accuracy_bounds"]["boundary_dof_error"], 1e-8)
            command = benchmark.build_command(manifest, case, self.root / "output.csv")
            self.assertEqual(command[command.index("--refine-serial") + 1], str(refinement))
            self.assertEqual(command[command.index("--refine-parallel") + 1], "0")
        coarse = manifest["configurations"][0]
        row = driver_row()
        row.update(elements=1988, velocity_l2=0.005)
        self.assertEqual(benchmark.validate_row(row, coarse, self.settings), [])
        with self.assertRaisesRegex(ValueError, "element count"):
            benchmark.validate_row(driver_row(), coarse, self.settings)
        row["level"] = 1
        with self.assertRaisesRegex(ValueError, "Wrong level"):
            benchmark.validate_row(row, coarse, self.settings)

    def test_refinement_reports_reference_matching_and_rates(self):
        manifest = self.refinement_manifest()
        self.assertEqual(self.run_quietly(manifest), 0)
        samples, summary = benchmark.summarize(self.output)
        self.assertEqual(len(samples), 12)
        self.assertEqual(len(summary), 6)
        self.assertEqual({row["serial_refinements"] for row in samples}, {0, 1, 2})
        self.assertTrue(all(row["driver_level"] == 0 for row in samples))
        self.assertTrue(all(row["direct_reference"] == "available" and row["mms_norm_difference_max"] == 0 for row in summary))
        comparisons = benchmark.write_refinement_summary(self.output, summary)
        self.assertEqual(len(comparisons), 4)
        self.assertTrue(all(row["comparison_eligible"] for row in comparisons))
        for row in comparisons:
            expected_ratio = ((48612 * 4 ** (row["to_refinement"] - 1) + 1) /
                              (48612 * 4 ** (row["from_refinement"] - 1) + 1))
            self.assertEqual(row["dof_ratio"], expected_ratio)
            self.assertEqual(row["solve_seconds_ratio"], 4)
            self.assertAlmostEqual(row["velocity_l2_rate"], 3)
            self.assertAlmostEqual(row["pressure_l2_rate"], 2)
            if row["solver"] == "mumps":
                self.assertEqual(row["outer_iterations_ratio"], "")
        self.assertEqual(len(list((self.output / "figures").glob("*.svg"))), 5)
        for path in (self.output / "figures").glob("*.svg"):
            ElementTree.parse(path)

    def test_refinement_failure_does_not_bridge_levels(self):
        manifest = self.refinement_manifest(solvers=("fgmres",))
        self.assertEqual(self.run_quietly(manifest), 0)
        _, summary = benchmark.summarize(self.output)
        summary[1]["ranking_eligible"] = False
        summary[1]["status_counts"] = '{"accuracy_failed": 1}'
        comparisons = benchmark.write_refinement_summary(self.output, summary)
        self.assertEqual(len(comparisons), 2)
        for row in comparisons:
            self.assertFalse(row["comparison_eligible"])
            self.assertEqual(row["solve_seconds_ratio"], "")
            self.assertEqual(row["velocity_l2_rate"], "")
        benchmark.render_refinement_figures(self.output / "figures", manifest, summary)
        image = (self.output / "figures/pspg-refinement-total.svg").read_text()
        self.assertNotIn(' L ', image)

    def test_missing_same_level_direct_reference(self):
        manifest = self.refinement_manifest()
        cases = [case for case in manifest["configurations"] if not (case["serial_refinements"] == 0 and case["solver"] == "mumps")]
        manifest = self.manifest(cases, repetitions=1)
        self.assertEqual(self.run_quietly(manifest), 0)
        _, summary = benchmark.summarize(self.output)
        coarse = next(row for row in summary if row["serial_refinements"] == 0)
        self.assertEqual(coarse["direct_reference"], "unavailable")
        self.assertEqual(coarse["mms_norm_difference_max"], "")

    def test_legacy_schema_one_offline_reporting(self):
        manifest = self.manifest()
        manifest["schema_version"] = 1
        manifest["settings"]["serial_refinements"] = 1
        manifest["fingerprint"] = benchmark.manifest_fingerprint(manifest)
        self.assertEqual(self.run_quietly(manifest), 0)
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("Offline report started a process")):
            samples, summary = benchmark.summarize(self.output)
        self.assertEqual(summary[0]["serial_refinements"], 1)
        self.assertTrue(all(row["serial_refinements"] == 1 for row in samples))
        self.assertTrue((self.output / "figures/pspg-total.svg").exists())

    def test_refinement_settings_change_refuses_resume(self):
        manifest = self.refinement_manifest()
        self.assertEqual(self.run_quietly(manifest), 0)
        changed = copy.deepcopy(manifest)
        changed["settings"]["refinement_levels"] = [0, 2]
        changed["fingerprint"] = benchmark.manifest_fingerprint(changed)
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            self.run_quietly(changed, resume=True)

    def test_nonconsecutive_refinement_rates_use_level_gap(self):
        manifest = self.refinement_manifest(solvers=("fgmres",))
        cases = [case for case in manifest["configurations"] if case["serial_refinements"] != 1]
        self.assertEqual(self.run_quietly(self.manifest(cases, repetitions=1)), 0)
        _, summary = benchmark.summarize(self.output)
        comparison, = benchmark.write_refinement_summary(self.output, summary)
        self.assertEqual(comparison["solve_seconds_ratio"], 16)
        self.assertAlmostEqual(comparison["velocity_l2_rate"], 3)
        self.assertAlmostEqual(comparison["velocity_h1_rate"], 2)

    def test_all_failed_refinement_retains_rows_and_empty_plots(self):
        manifest = self.refinement_manifest(solvers=("fgmres",))
        self.environment["STOKES_TEST_FAKE_MODE"] = "accuracy"
        self.assertEqual(self.run_quietly(manifest), 2)
        _, summary = benchmark.summarize(self.output)
        comparisons = benchmark.write_refinement_summary(self.output, summary)
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(not row["comparison_eligible"] and row["dof_ratio"] == "" for row in comparisons))
        image = self.output / "figures/pspg-refinement-total.svg"
        ElementTree.parse(image)
        self.assertIn("No fully successful configurations", image.read_text())

    def test_manifest_canonical_refinement_order_and_bounds(self):
        first = benchmark.parse_arguments(self.arguments + ["--refinements", "2", "0", "1"])
        second = benchmark.parse_arguments(self.arguments + ["--refinements", "0", "1", "2"])
        with mock.patch.object(benchmark, "provenance", return_value={}):
            manifest, _ = benchmark.make_manifest(first)
            reordered, _ = benchmark.make_manifest(second)
        self.assertEqual(manifest["fingerprint"], reordered["fingerprint"])
        self.assertEqual(manifest["settings"]["input_elements"], 1988)
        self.assertEqual(manifest["settings"]["accuracy_bounds_by_refinement"]["2"]["velocity_l2"], 0.000125)

    def test_command_no_shell_and_explicit_settings(self):
        manifest = self.manifest()
        manifest["provenance"]["launcher"]["tokens"] = ["mpiexec", "-n", "{ranks}", "-bind-to", "core"]
        path = self.root / "with spaces;literal.csv"
        command = benchmark.build_command(manifest, self.case, path)
        self.assertEqual(command[:5], ["mpiexec", "-n", "4", "-bind-to", "core"])
        self.assertIn(str(path), command)
        self.assertIn(str(self.mesh), command)
        self.assertEqual(command[command.index("--refine-serial")+1], "1")
        self.assertEqual(command[command.index("--quadrature-order")+1], "6")
        self.assertIn("--no-paraview", command)

    def test_csv_validation_and_absolute_tolerance(self):
        row = driver_row()
        self.assertEqual(benchmark.validate_row(row, self.case, self.settings), [])
        row.update(absolute_residual=1e-15, relative_residual=1e-6)
        self.assertEqual(benchmark.validate_row(row, self.case, self.settings), [])
        row["absolute_residual"] = 1e-4
        self.assertIn("true residual", benchmark.validate_row(row, self.case, self.settings)[0])

    def test_wrong_metadata_invalid_timing_and_accuracy(self):
        for field, value in [("order", 3), ("geometry_order", 2), ("periodic", 1), ("method", "cip"),
                             ("preconditioner", "block-diagonal-amg"), ("elements", 0),
                             ("linear_solve_seconds", -1), ("direct_setup_seconds", 1),
                             ("linear_solve_seconds", 100), ("outer_iterations", 99999)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                benchmark.validate_row({**driver_row(), field: value}, self.case, self.settings)
        row = driver_row()
        row["pressure_l2"] = 0.1
        self.assertTrue(benchmark.validate_row(row, self.case, self.settings))

    def test_csv_missing_duplicate_nonfinite_and_row_count(self):
        path = self.root / "driver.csv"
        def write(rows, columns=None):
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        row = driver_row()
        write([row])
        self.assertEqual(benchmark.read_driver_csv(path)["outer_iterations"], 48)
        write([row, row])
        with self.assertRaises(ValueError):
            benchmark.read_driver_csv(path)
        write([{**row, "absolute_residual": "nan"}])
        with self.assertRaises(ValueError):
            benchmark.read_driver_csv(path)
        write([row], list(row) + ["solver"])
        with self.assertRaises(ValueError):
            benchmark.read_driver_csv(path)
        path.write_text("solver,converged\nfgmres,1\n")
        with self.assertRaises(ValueError):
            benchmark.read_driver_csv(path)

    def test_success_resume_offline_report_and_figures(self):
        manifest = self.manifest()
        self.assertEqual(self.run_quietly(manifest), 0)
        logs = {str(path): benchmark.file_hash(path) for path in self.output.glob("runs/**/*") if path.is_file()}
        with mock.patch.object(benchmark, "execute_attempt", side_effect=AssertionError("Unexpected rerun")):
            self.assertEqual(self.run_quietly(manifest, resume=True), 0)
        self.assertEqual(logs, {str(path): benchmark.file_hash(path) for path in self.output.glob("runs/**/*") if path.is_file()})
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("Report launched a process")):
            samples, summary = benchmark.summarize(self.output)
        self.assertEqual(len(samples), 3)
        self.assertEqual(summary[0]["successful_samples"], 2)
        self.assertTrue(summary[0]["ranking_eligible"])
        ElementTree.parse(self.output / "figures/pspg-total.svg")
        self.assertEqual(summary[0]["solve_seconds_median"], 0.5)

    def test_manifest_changes_refuse_resume(self):
        manifest = self.manifest()
        self.run_quietly(manifest)
        changed = copy.deepcopy(manifest)
        changed["fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            self.run_quietly(changed, resume=True)
        with self.assertRaisesRegex(ValueError, "Study exists"):
            self.run_quietly(manifest)

    def test_changed_raw_artifact_detected(self):
        self.run_quietly(self.manifest())
        csv_path = next(self.output.glob("runs/**/driver.csv"))
        csv_path.write_text(csv_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Modified/missing"):
            benchmark.summarize(self.output)

    def test_failure_continuation_and_immutable_retry(self):
        self.environment["STOKES_TEST_FAKE_MODE"] = "accuracy"
        manifest = self.manifest()
        self.assertEqual(self.run_quietly(manifest), 2)
        samples, summary = benchmark.summarize(self.output)
        self.assertTrue(all(row["status"] == "accuracy_failed" for row in samples))
        self.assertFalse(summary[0]["ranking_eligible"])
        self.assertEqual(summary[0]["solve_seconds_median"], "")
        old_statuses = {str(path): path.read_bytes() for path in self.output.glob("runs/**/status.json")}
        self.environment.pop("STOKES_TEST_FAKE_MODE")
        self.assertEqual(self.run_quietly(manifest, resume=True), 2)
        self.assertEqual(self.run_quietly(manifest, resume=True, retry_failed=True), 0)
        for path, data in old_statuses.items():
            self.assertEqual(Path(path).read_bytes(), data)
        self.assertEqual(len(list(self.output.glob("runs/**/status.json"))), 6)
        _, summary = benchmark.summarize(self.output)
        self.assertEqual(summary[0]["historical_failed_attempts"], 3)
        self.assertFalse(summary[0]["ranking_eligible"])

    def test_unavailable_backend_does_not_stop_other_cases(self):
        self.environment["STOKES_TEST_FAKE_MODE"] = "unavailable"
        direct = {"id": "direct", "method": "pspg", "solver": "mumps", "preconditioner": "not-applicable"}
        self.assertEqual(self.run_quietly(self.manifest([self.case, direct], repetitions=1)), 2)
        samples, summary = benchmark.summarize(self.output)
        self.assertEqual({row["status"] for row in samples}, {"success", "unavailable"})
        self.assertEqual(summary[0]["direct_reference"], "unavailable")

    def test_iteration_limit_and_invalid_output(self):
        for mode, expected in [("nonconverged", "iteration_limit"), ("nonfinite", "invalid_output")]:
            with self.subTest(mode=mode):
                self.environment["STOKES_TEST_FAKE_MODE"] = mode
                self.output = self.root / mode
                self.assertEqual(self.run_quietly(self.manifest()), 2)
                samples, _ = benchmark.summarize(self.output)
                self.assertTrue(all(row["status"] == expected for row in samples))

    def test_incomplete_attempt_is_preserved(self):
        manifest = self.manifest()
        self.output.mkdir()
        benchmark.write_json(self.output / "manifest.json", manifest)
        job = manifest["schedule"][0]
        abandoned = self.output / "runs" / job["configuration_id"] / job["sample_id"] / "attempt-001"
        abandoned.mkdir(parents=True)
        (abandoned / "stdout.log").write_text("partial")
        self.assertEqual(self.run_quietly(manifest, resume=True), 0)
        self.assertEqual((abandoned / "stdout.log").read_text(), "partial")
        self.assertTrue((abandoned.parent / "attempt-002/status.json").exists())

    def test_timeout_kills_launcher_and_local_child(self):
        self.environment["STOKES_TEST_FAKE_MODE"] = "sleep"
        manifest = self.manifest()
        manifest["protocol"]["timeout_seconds"] = 0.5
        directory = self.root / "timeout"
        started = time.monotonic()
        result = benchmark.execute_attempt(directory, manifest, self.case, manifest["schedule"][0], self.environment)
        self.assertEqual(result["status"], "timeout")
        self.assertLess(time.monotonic() - started, 5)
        heartbeat = directory / "driver.heartbeat"
        initial = heartbeat.read_text()
        time.sleep(0.15)
        self.assertEqual(heartbeat.read_text(), initial, "Timed-out child is still writing")

    def test_manifest_tampering_refused(self):
        manifest = self.manifest()
        self.run_quietly(manifest)
        saved = benchmark.read_json(self.output / "manifest.json")
        saved["settings"]["ranks"] = 8
        benchmark.write_json(self.output / "manifest.json", saved)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            benchmark.summarize(self.output)

    def test_warmup_exclusion_and_partial_success(self):
        manifest = self.manifest()
        self.environment["STOKES_TEST_FAKE_MODE"] = "accuracy"
        self.run_quietly(manifest)
        self.environment.pop("STOKES_TEST_FAKE_MODE")
        job = next(job for job in manifest["schedule"] if not job["warmup"])
        path = self.output / "runs" / job["configuration_id"] / job["sample_id"] / "attempt-002"
        benchmark.execute_attempt(path, manifest, self.case, job, self.environment)
        _, summary = benchmark.summarize(self.output)
        self.assertEqual(summary[0]["successful_samples"], 1)
        self.assertEqual(summary[0]["success_fraction"], 0.5)
        self.assertFalse(summary[0]["ranking_eligible"])
        self.assertEqual(summary[0]["warmup_failures"], 1)

    def test_sigterm_records_interruption(self):
        self.environment["STOKES_TEST_FAKE_MODE"] = "sleep"
        command = [sys.executable, str(benchmark.__file__)] + self.arguments
        process = subprocess.Popen(command, env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not list(self.output.glob("runs/**/driver.heartbeat")):
            if process.poll() is not None:
                self.fail(process.communicate())
            time.sleep(0.03)
        self.assertTrue(list(self.output.glob("runs/**/driver.heartbeat")))
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        self.assertEqual(process.returncode, 130)
        samples, _ = benchmark.summarize(self.output)
        self.assertEqual(sum(row["status"] == "interrupted" for row in samples), 1)
        self.assertEqual(sum(row["status"] == "pending" for row in samples), 2)

    def test_lock_and_nonempty_output(self):
        self.output.mkdir()
        with benchmark.study_lock(self.output):
            with self.assertRaisesRegex(ValueError, "Another runner"):
                with benchmark.study_lock(self.output):
                    pass
        (self.output / "unrelated.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            self.run_quietly(self.manifest())

    def test_plan_is_read_only_and_cli_validation(self):
        arguments = self.arguments[3:]
        with mock.patch.object(benchmark, "provenance", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(benchmark.main(["plan"] + arguments), 0)
        self.assertFalse(self.output.exists())
        for arguments in (["plan", "--timeout", "nan"], ["plan", "--ranks", "0"],
                          ["plan", "--refinements", "-1"], ["plan", "--refinements", "3"],
                          ["plan", "--refinements", "0", "0"],
                          ["plan", "--methods", "pspg", "pspg"],
                          ["run", "--output", "x", "--retry-failed"],
                          ["plan", "--execution-class", "controlled"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.parse_arguments(arguments)

    def test_manifest_records_hashes_and_thread_control(self):
        args = benchmark.parse_arguments(self.arguments)
        with mock.patch.object(benchmark, "command_output", return_value="static"):
            manifest, environment = benchmark.make_manifest(args)
            second, _ = benchmark.make_manifest(args)
        self.assertEqual(manifest["fingerprint"], second["fingerprint"])
        self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(manifest["provenance"]["mesh"]["sha256"], benchmark.file_hash(self.mesh))


if __name__ == "__main__":
    if "--mesh" in sys.argv:
        sys.exit(fake_driver())
    unittest.main()
