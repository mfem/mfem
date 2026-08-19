#!/usr/bin/env python3
"""Compare two distributed raw-design optimization checkpoints."""

import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import sys


DESIGN_MAGIC = 0x52484F31
HEADER = struct.Struct("=iiq")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare matching design.* rank files from two optimization "
            "checkpoint directories. The MPI partition must be identical."
        )
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def read_metadata(directory):
    path = directory / "metadata.txt"
    metadata = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected KEY VALUE")
        metadata[fields[0]] = fields[1]
    return metadata


def design_files(directory, metadata):
    files = sorted(directory.glob("design.[0-9][0-9][0-9][0-9][0-9][0-9]"))
    expected = int(metadata.get("n_mpi_ranks", "-1"))
    if expected < 1 or len(files) != expected:
        raise ValueError(
            f"{directory}: found {len(files)} design files, metadata expects {expected}"
        )
    return files


def read_piece(path):
    try:
        with path.open("rb") as stream:
            header = stream.read(HEADER.size)
            if len(header) != HEADER.size:
                raise ValueError(f"{path}: truncated header")
            magic, design_index, size = HEADER.unpack(header)
            if magic != DESIGN_MAGIC or size < 0:
                raise ValueError(f"{path}: invalid magic or vector size")
            values = array("d")
            values.fromfile(stream, size)
            if stream.read(1):
                raise ValueError(f"{path}: trailing data")
    except (OSError, EOFError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    return design_index, values


def compare(candidate_dir, reference_dir):
    candidate_metadata = read_metadata(candidate_dir)
    reference_metadata = read_metadata(reference_dir)
    candidate_files = design_files(candidate_dir, candidate_metadata)
    reference_files = design_files(reference_dir, reference_metadata)

    sums = {"candidate_sq": 0.0, "reference_sq": 0.0,
            "difference_sq": 0.0, "dot": 0.0}
    maximum_absolute_difference = 0.0
    count = 0
    embedded_indices = {"candidate": set(), "reference": set()}
    for candidate_file, reference_file in zip(candidate_files, reference_files):
        candidate_index, candidate = read_piece(candidate_file)
        reference_index, reference = read_piece(reference_file)
        embedded_indices["candidate"].add(candidate_index)
        embedded_indices["reference"].add(reference_index)
        if len(candidate) != len(reference):
            raise ValueError(
                f"partition mismatch: {candidate_file} has {len(candidate)} values, "
                f"{reference_file} has {len(reference)}"
            )
        for left, right in zip(candidate, reference):
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("checkpoint contains a non-finite design value")
            difference = left - right
            sums["candidate_sq"] += left * left
            sums["reference_sq"] += right * right
            sums["difference_sq"] += difference * difference
            sums["dot"] += left * right
            maximum_absolute_difference = max(
                maximum_absolute_difference, abs(difference)
            )
        count += len(candidate)

    candidate_norm = math.sqrt(sums["candidate_sq"])
    reference_norm = math.sqrt(sums["reference_sq"])
    difference_norm = math.sqrt(sums["difference_sq"])
    relative_l2 = difference_norm / reference_norm if reference_norm else None
    cosine = None
    if candidate_norm and reference_norm:
        cosine = sums["dot"] / (candidate_norm * reference_norm)

    compatibility_keys = (
        "n_mpi_ranks", "refinement_level", "design_fe_order"
    )
    metadata_mismatches = [
        key for key in compatibility_keys
        if candidate_metadata.get(key) != reference_metadata.get(key)
    ]
    return {
        "candidate": str(candidate_dir),
        "reference": str(reference_dir),
        "global_design_dofs": count,
        "candidate_embedded_design_indices": sorted(embedded_indices["candidate"]),
        "reference_embedded_design_indices": sorted(embedded_indices["reference"]),
        "candidate_metadata_design_iteration": candidate_metadata.get(
            "design_iteration", candidate_metadata.get("iteration")
        ),
        "reference_metadata_design_iteration": reference_metadata.get(
            "design_iteration", reference_metadata.get("iteration")
        ),
        "candidate_l2_norm": candidate_norm,
        "reference_l2_norm": reference_norm,
        "difference_l2_norm": difference_norm,
        "relative_l2_error": relative_l2,
        "cosine": cosine,
        "maximum_absolute_error": maximum_absolute_difference,
        "metadata_compatibility_mismatches": metadata_mismatches,
    }


def printable(value):
    return "undefined" if value is None else f"{value:.16e}"


def main():
    args = parse_arguments()
    try:
        report = compare(args.candidate, args.reference)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Candidate: {report['candidate']}")
        print(f"Reference: {report['reference']}")
        print(f"Global design DOFs: {report['global_design_dofs']}")
        print(f"Relative L2 error: {printable(report['relative_l2_error'])}")
        print(f"Cosine: {printable(report['cosine'])}")
        print(f"Maximum absolute error: {report['maximum_absolute_error']:.16e}")
        if report["metadata_compatibility_mismatches"]:
            print(
                "WARNING: checkpoint metadata differs in "
                + ", ".join(report["metadata_compatibility_mismatches"])
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
