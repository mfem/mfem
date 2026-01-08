#!/usr/bin/env python3
import argparse
import re

def parse_benchmark_file(path):
    """
    Parse Google Benchmark TSV-like output (text table).
    Returns a dict: { benchmark_name : time_in_ms }
    """
    benchmarks = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header and separators
            if not line or line.startswith('-') or line.startswith('Benchmark'):
                continue

            # Example line:
            # BP1Mass/1                  4.02 ms         4.01 ms          175  286.923/s
            parts = re.split(r'\s{2,}', line)
            if len(parts) < 2:
                continue

            name = parts[0]
            time_str = parts[1]

            # Extract numeric time and unit
            match = re.match(r'([\d.]+)\s*(ns|us|ms|s)', time_str)
            if not match:
                continue

            val, unit = match.groups()
            time_ms = float(val)
            if unit == 'ns':
                time_ms /= 1e6
            elif unit == 'us':
                time_ms /= 1e3
            elif unit == 's':
                time_ms *= 1e3

            benchmarks[name] = time_ms
    return benchmarks

def compute_speedups(base, new):
    results = []
    for name, base_time in base.items():
        if name in new:
            new_time = new[name]
            speedup = base_time / new_time if new_time != 0 else float('inf')
            results.append((name, base_time, new_time, speedup))
    return sorted(results, key=lambda x: x[3], reverse=True)

def main():
    parser = argparse.ArgumentParser(
        description="Compare two Google Benchmark TSV outputs and compute speedups"
    )
    parser.add_argument("base", help="Baseline benchmark file (TSV-style)")
    parser.add_argument("new", help="New benchmark file (TSV-style)")
    args = parser.parse_args()

    base = parse_benchmark_file(args.base)
    new = parse_benchmark_file(args.new)

    results = compute_speedups(base, new)

    print(f"{'Benchmark':30} {'Base (ms)':>12} {'New (ms)':>12} {'Speedup':>10}")
    print("-" * 70)
    for name, base_time, new_time, speedup in results:
        print(f"{name:30} {base_time:12.3f} {new_time:12.3f} {speedup:10.2f}x")

if __name__ == "__main__":
    main()
