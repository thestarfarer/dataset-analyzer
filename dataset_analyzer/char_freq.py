#!/usr/bin/env python3
"""Character frequency analysis with multiprocessing."""

import argparse
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


DEFAULT_INPUT = "data.txt"


def count_chars(sample: str) -> Counter:
    """Count characters in a single sample."""
    return Counter(sample)


def main():
    parser = argparse.ArgumentParser(description="Count unique characters in dataset")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input file path")
    args = parser.parse_args()

    # Read and split samples
    print(f"Reading {args.input}...")
    with open(args.input, "r") as f:
        samples = [s.strip() for s in f.read().split("<BREAK>") if s.strip()]
    print(f"Loaded {len(samples):,} samples")

    # Count chars in parallel
    print("Counting characters...")
    num_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        counters = list(executor.map(count_chars, samples, chunksize=1000))

    # Aggregate
    total = Counter()
    for c in counters:
        total.update(c)

    total_chars = sum(total.values())

    # Output
    print(f"\nTotal unique chars: {len(total):,}")
    print(f"Total chars: {total_chars:,}\n")
    print(f"{'Char':<8} | {'Count':>12} | {'%':>6}")
    print("-" * 32)

    for char, count in total.most_common():
        pct = 100 * count / total_chars
        # Display special chars nicely
        if char == "\n":
            display = "\\n"
        elif char == "\t":
            display = "\\t"
        elif char == " ":
            display = "SPACE"
        elif char == "\r":
            display = "\\r"
        else:
            display = repr(char)[1:-1]  # Strip quotes from repr
        print(f"{display:<8} | {count:>12,} | {pct:>5.2f}%")


if __name__ == "__main__":
    main()
