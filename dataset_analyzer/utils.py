#!/usr/bin/env python3
"""Shared utilities for language analysis."""

import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from typing import List

DEFAULT_INPUT = "data.txt"
CHUNKSIZE = 1000


def load_samples(path: str = DEFAULT_INPUT) -> List[str]:
    """Load dataset, split by <BREAK>, strip whitespace."""
    with open(path, "r") as f:
        samples = [s.strip() for s in f.read().split("<BREAK>") if s.strip()]
    return samples


def tokenize_words(text: str) -> List[str]:
    """Simple word tokenization - extract word characters."""
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences at sentence-ending punctuation."""
    # Split on .!? followed by space or newline (handles abbreviations better)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_num_workers() -> int:
    """Get CPU count for worker pool."""
    return multiprocessing.cpu_count()


def run_parallel(func, items, chunksize: int = CHUNKSIZE):
    """Run function in parallel over items, return list of results."""
    num_workers = get_num_workers()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(func, items, chunksize=chunksize))
    return results


def print_header(title: str):
    """Print section header."""
    print("=" * 80)
    print(title.upper())
    print("=" * 80)


def print_subheader(title: str):
    """Print subsection header."""
    print(f"\n--- {title} ---")
