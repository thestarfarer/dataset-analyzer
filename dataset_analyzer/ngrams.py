#!/usr/bin/env python3
"""N-gram analysis with multiprocessing."""

import argparse
import re
from collections import Counter
from typing import Tuple

from .utils import (
    DEFAULT_INPUT,
    load_samples,
    tokenize_words,
    run_parallel,
    print_header,
    print_subheader,
)

# Common dialogue tags to track
DIALOGUE_TAGS = [
    "said", "asked", "replied", "answered", "whispered", "shouted",
    "yelled", "screamed", "murmured", "muttered", "exclaimed", "demanded",
    "insisted", "suggested", "added", "continued", "agreed", "admitted",
    "announced", "argued", "began", "begged", "called", "complained",
    "confessed", "cried", "declared", "denied", "explained", "groaned",
    "growled", "hissed", "laughed", "lied", "moaned", "nodded",
    "observed", "offered", "ordered", "pleaded", "promised", "protested",
    "questioned", "repeated", "responded", "sighed", "snapped", "sobbed",
    "stammered", "stated", "urged", "warned", "wondered",
]


TOP_N_PER_WORKER = 100  # Keep only top N per sample to reduce memory


def extract_ngrams(sample: str) -> Tuple[Counter, Counter, Counter]:
    """Extract bigrams, trigrams, and dialogue tags from sample."""
    words = tokenize_words(sample.lower())

    # Bigrams
    bigrams = Counter()
    for i in range(len(words) - 1):
        bigrams[(words[i], words[i+1])] += 1

    # Trigrams
    trigrams = Counter()
    for i in range(len(words) - 2):
        trigrams[(words[i], words[i+1], words[i+2])] += 1

    # Dialogue tags
    tags = Counter()
    for word in words:
        if word in DIALOGUE_TAGS:
            tags[word] += 1

    # Keep only top-N to reduce memory
    return (
        Counter(dict(bigrams.most_common(TOP_N_PER_WORKER))),
        Counter(dict(trigrams.most_common(TOP_N_PER_WORKER))),
        tags
    )


def analyze_ngrams(samples: list, top_n: int = 50) -> dict:
    """Run full n-gram analysis."""
    print(f"Analyzing {len(samples):,} samples...")

    # Extract n-grams in parallel
    results = run_parallel(extract_ngrams, samples)

    # Aggregate
    total_bigrams = Counter()
    total_trigrams = Counter()
    total_tags = Counter()

    for bigrams, trigrams, tags in results:
        total_bigrams.update(bigrams)
        total_trigrams.update(trigrams)
        total_tags.update(tags)

    # Find repeated phrases (trigrams appearing 100+ times)
    repeated_phrases = [(t, c) for t, c in total_trigrams.most_common() if c >= 100]

    return {
        "bigrams": total_bigrams,
        "trigrams": total_trigrams,
        "dialogue_tags": total_tags,
        "repeated_phrases": repeated_phrases[:100],  # Top 100 repeated
        "top_n": top_n,
    }


def print_results(results: dict):
    """Print analysis results."""
    print_header("N-gram Analysis")

    top_n = results["top_n"]

    print_subheader(f"Top {top_n} Word Bigrams")
    print(f"{'Rank':<6} {'Bigram':<30} {'Count':>12}")
    print("-" * 50)
    for i, (bigram, count) in enumerate(results['bigrams'].most_common(top_n), 1):
        bigram_str = " ".join(bigram)
        print(f"{i:<6} {bigram_str:<30} {count:>12,}")

    print_subheader(f"Top {top_n} Word Trigrams")
    print(f"{'Rank':<6} {'Trigram':<40} {'Count':>12}")
    print("-" * 55)
    for i, (trigram, count) in enumerate(results['trigrams'].most_common(top_n), 1):
        trigram_str = " ".join(trigram)
        print(f"{i:<6} {trigram_str:<40} {count:>12,}")

    print_subheader("Dialogue Tags Frequency")
    print(f"{'Tag':<15} {'Count':>12} {'%':>8}")
    print("-" * 40)
    total_tags = sum(results['dialogue_tags'].values())
    for tag, count in results['dialogue_tags'].most_common():
        pct = 100 * count / total_tags if total_tags > 0 else 0
        print(f"{tag:<15} {count:>12,} {pct:>7.1f}%")
    print(f"\nTotal dialogue tags: {total_tags:,}")

    print_subheader(f"Repeated Phrases (trigrams appearing 100+ times)")
    print(f"Found {len(results['repeated_phrases']):,} repeated phrases")
    print(f"\n{'Rank':<6} {'Phrase':<45} {'Count':>10}")
    print("-" * 65)
    for i, (phrase, count) in enumerate(results['repeated_phrases'][:30], 1):
        phrase_str = " ".join(phrase)
        print(f"{i:<6} {phrase_str:<45} {count:>10,}")
    if len(results['repeated_phrases']) > 30:
        print(f"... and {len(results['repeated_phrases']) - 30} more")


def main():
    parser = argparse.ArgumentParser(description="N-gram analysis of dataset")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input file path")
    parser.add_argument("--top", "-n", type=int, default=50, help="Number of top n-grams to show")
    args = parser.parse_args()

    samples = load_samples(args.input)
    print(f"Loaded {len(samples):,} samples")

    results = analyze_ngrams(samples, top_n=args.top)
    print_results(results)


if __name__ == "__main__":
    main()
