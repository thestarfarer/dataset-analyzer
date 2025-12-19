#!/usr/bin/env python3
"""Run all language analyses."""

import argparse
import time

from .utils import DEFAULT_INPUT, load_samples, print_header
from .lexical import analyze_lexical, print_results as print_lexical
from .ngrams import analyze_ngrams, print_results as print_ngrams
from .structural import analyze_structural, print_results as print_structural
from .quality import analyze_all_quality, print_results as print_quality


def main():
    parser = argparse.ArgumentParser(description="Run all language analyses")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input file path")
    parser.add_argument("--skip", "-s", nargs="+", choices=["lexical", "ngrams", "structural", "quality"],
                        default=[], help="Analyses to skip")
    args = parser.parse_args()

    # Load samples once
    print(f"Loading samples from {args.input}...")
    start = time.time()
    samples = load_samples(args.input)
    print(f"Loaded {len(samples):,} samples in {time.time() - start:.1f}s\n")

    total_start = time.time()

    # Run each analysis
    if "lexical" not in args.skip:
        print("\n" + "=" * 80)
        print("Running LEXICAL analysis...")
        print("=" * 80)
        start = time.time()
        results = analyze_lexical(samples)
        print_lexical(results)
        print(f"\n[Completed in {time.time() - start:.1f}s]")

    if "ngrams" not in args.skip:
        print("\n" + "=" * 80)
        print("Running N-GRAM analysis...")
        print("=" * 80)
        start = time.time()
        results = analyze_ngrams(samples)
        print_ngrams(results)
        print(f"\n[Completed in {time.time() - start:.1f}s]")

    if "structural" not in args.skip:
        print("\n" + "=" * 80)
        print("Running STRUCTURAL analysis...")
        print("=" * 80)
        start = time.time()
        results = analyze_structural(samples)
        print_structural(results)
        print(f"\n[Completed in {time.time() - start:.1f}s]")

    if "quality" not in args.skip:
        print("\n" + "=" * 80)
        print("Running QUALITY analysis...")
        print("=" * 80)
        start = time.time()
        results = analyze_all_quality(samples)
        print_quality(results)
        print(f"\n[Completed in {time.time() - start:.1f}s]")

    print("\n" + "=" * 80)
    print(f"ALL ANALYSES COMPLETED in {time.time() - total_start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
