#!/usr/bin/env python3
"""Quality and anomaly detection with multiprocessing."""

import argparse
import re
from collections import Counter
from typing import Tuple, Set, Dict

from .utils import (
    DEFAULT_INPUT,
    load_samples,
    tokenize_words,
    tokenize_sentences,
    run_parallel,
    print_header,
    print_subheader,
)

# Unusual punctuation patterns to detect
PUNCT_PATTERNS = [
    (r'\?\?\?+', '???+'),
    (r'!!!+', '!!!+'),
    (r'\.\.\.\.+', '....+'),
    (r'---+', '---+'),
    (r'\*\*\*+', '***+'),
    (r'~~~+', '~~~+'),
    (r'___+', '___+'),
    (r'===+', '===+'),
]


def analyze_quality(sample: str) -> Tuple[Set[str], Dict[str, int], Counter, int, int]:
    """Analyze quality issues in a single sample.

    Returns:
        sentences: set of sentences (for duplicate detection)
        punct_counts: counts of unusual punctuation patterns
        caps_words: Counter of all-caps words
        number_count: count of numeric tokens
        total_tokens: total token count
    """
    # Extract sentences for duplicate detection
    sentences = set(tokenize_sentences(sample))

    # Unusual punctuation
    punct_counts = {}
    for pattern, name in PUNCT_PATTERNS:
        matches = re.findall(pattern, sample)
        if matches:
            punct_counts[name] = len(matches)

    # All-caps words (3+ chars to avoid I, A, etc.)
    words = tokenize_words(sample)
    caps_words = Counter()
    for word in words:
        if len(word) >= 3 and word.isupper():
            caps_words[word] += 1

    # Number density - count directly from text since tokenize_words skips numbers
    numbers = re.findall(r'\b\d+\b', sample)
    number_count = len(numbers)
    total_tokens = len(words) + number_count

    return sentences, punct_counts, caps_words, number_count, total_tokens


def find_duplicate_sentences(all_sentences: list) -> Counter:
    """Find sentences that appear in multiple samples."""
    # Count how many samples each sentence appears in
    sentence_counts = Counter()
    for sentences_set in all_sentences:
        for s in sentences_set:
            # Normalize: lowercase and strip
            s_norm = s.lower().strip()
            if len(s_norm) > 20:  # Only count meaningful sentences
                sentence_counts[s_norm] += 1

    # Return sentences appearing 2+ times
    return Counter({s: c for s, c in sentence_counts.items() if c >= 2})


def analyze_all_quality(samples: list) -> dict:
    """Run full quality analysis."""
    print(f"Analyzing {len(samples):,} samples...")

    # Analyze in parallel
    results = run_parallel(analyze_quality, samples)

    # Aggregate
    all_sentences = []
    total_punct = Counter()
    total_caps = Counter()
    total_numbers = 0
    total_tokens = 0

    for sentences, punct_counts, caps_words, num_count, tok_count in results:
        all_sentences.append(sentences)
        for pattern, count in punct_counts.items():
            total_punct[pattern] += count
        total_caps.update(caps_words)
        total_numbers += num_count
        total_tokens += tok_count

    # Find duplicate sentences
    print("Finding duplicate sentences...")
    dup_sentences = find_duplicate_sentences(all_sentences)

    return {
        "punct_patterns": total_punct,
        "caps_words": total_caps,
        "number_count": total_numbers,
        "total_tokens": total_tokens,
        "number_density": total_numbers / total_tokens if total_tokens > 0 else 0,
        "duplicate_sentences": dup_sentences,
    }


def print_results(results: dict):
    """Print analysis results."""
    print_header("Quality / Anomaly Detection")

    print_subheader("Unusual Punctuation Patterns")
    if results['punct_patterns']:
        print(f"{'Pattern':<15} {'Count':>12}")
        print("-" * 30)
        for pattern, count in results['punct_patterns'].most_common():
            print(f"{pattern:<15} {count:>12,}")
    else:
        print("No unusual punctuation patterns found.")

    print_subheader("All-Caps Words (3+ chars)")
    total_caps = sum(results['caps_words'].values())
    print(f"Total all-caps occurrences: {total_caps:,}")
    print(f"Unique all-caps words: {len(results['caps_words']):,}")
    if results['caps_words']:
        print(f"\nTop 30 all-caps words:")
        print(f"{'Word':<20} {'Count':>12}")
        print("-" * 35)
        for word, count in results['caps_words'].most_common(30):
            print(f"{word:<20} {count:>12,}")

    print_subheader("Number Density")
    print(f"Numeric tokens:  {results['number_count']:,}")
    print(f"Total tokens:    {results['total_tokens']:,}")
    print(f"Number density:  {results['number_density']*100:.3f}%")

    print_subheader("Duplicate Sentences (appearing in 2+ samples)")
    dup = results['duplicate_sentences']
    print(f"Total duplicate sentences: {len(dup):,}")
    if dup:
        # Show most common duplicates
        print(f"\nTop 20 most repeated sentences:")
        print(f"{'Count':>8}  Sentence")
        print("-" * 70)
        for sentence, count in dup.most_common(20):
            # Truncate long sentences
            display = sentence[:60] + "..." if len(sentence) > 60 else sentence
            print(f"{count:>8}  {display}")

        # Distribution of duplicate counts
        print(f"\nDuplicate frequency distribution:")
        counts = list(dup.values())
        bins = [(2, 2), (3, 5), (6, 10), (11, 50), (51, 100), (101, float('inf'))]
        labels = ["2x", "3-5x", "6-10x", "11-50x", "51-100x", "100+x"]
        print(f"{'Frequency':<12} {'Sentences':>12}")
        print("-" * 25)
        for (low, high), label in zip(bins, labels):
            count = sum(1 for c in counts if low <= c <= high)
            if count > 0:
                print(f"{label:<12} {count:>12,}")


def main():
    parser = argparse.ArgumentParser(description="Quality analysis of dataset")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input file path")
    args = parser.parse_args()

    samples = load_samples(args.input)
    print(f"Loaded {len(samples):,} samples")

    results = analyze_all_quality(samples)
    print_results(results)


if __name__ == "__main__":
    main()
