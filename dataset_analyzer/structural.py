#!/usr/bin/env python3
"""Text structure analysis with multiprocessing."""

import argparse
import re
from typing import Tuple, List

import numpy as np

from .utils import (
    DEFAULT_INPUT,
    load_samples,
    tokenize_words,
    tokenize_sentences,
    run_parallel,
    print_header,
    print_subheader,
)


def analyze_structure(sample: str) -> Tuple[List[int], List[int], int, int, int, int]:
    """Analyze structure of a single sample.

    Returns:
        sentence_lengths: list of word counts per sentence
        para_lengths: list of sentence counts per paragraph
        quoted_chars: characters inside quotes
        total_chars: total characters
        double_quotes: count of " characters
        single_quotes: count of ' characters used as quotes
    """
    # Sentence lengths (in words)
    sentences = tokenize_sentences(sample)
    sentence_lengths = [len(tokenize_words(s)) for s in sentences]

    # Paragraph lengths (in sentences) - split on double newlines
    paragraphs = re.split(r'\n\s*\n', sample)
    para_lengths = []
    for para in paragraphs:
        para = para.strip()
        if para:
            para_sentences = tokenize_sentences(para)
            para_lengths.append(len(para_sentences))

    # Dialogue analysis - count chars inside quotes
    total_chars = len(sample)

    # Find text inside double quotes
    double_quoted = re.findall(r'"[^"]*"', sample)
    double_quote_chars = sum(len(q) - 2 for q in double_quoted)  # -2 for the quotes themselves

    # Find text inside single quotes (dialogue style, not apostrophes)
    # Look for 'word word' patterns, not contractions like don't
    single_quoted = re.findall(r"'[^']{2,}'", sample)
    single_quote_chars = sum(len(q) - 2 for q in single_quoted)

    quoted_chars = double_quote_chars + single_quote_chars

    double_quotes = sample.count('"')
    single_quotes = len(single_quoted) * 2  # Count quote marks used for dialogue

    return sentence_lengths, para_lengths, quoted_chars, total_chars, double_quotes, single_quotes


def analyze_structural(samples: list) -> dict:
    """Run full structural analysis."""
    print(f"Analyzing {len(samples):,} samples...")

    # Analyze in parallel
    results = run_parallel(analyze_structure, samples)

    # Aggregate
    all_sentence_lengths = []
    all_para_lengths = []
    total_quoted = 0
    total_chars = 0
    total_double_quotes = 0
    total_single_quotes = 0

    for sent_lens, para_lens, quoted, chars, double_q, single_q in results:
        all_sentence_lengths.extend(sent_lens)
        all_para_lengths.extend(para_lens)
        total_quoted += quoted
        total_chars += chars
        total_double_quotes += double_q
        total_single_quotes += single_q

    sent_arr = np.array(all_sentence_lengths) if all_sentence_lengths else np.array([0])
    para_arr = np.array(all_para_lengths) if all_para_lengths else np.array([0])

    return {
        "sentence_count": len(all_sentence_lengths),
        "sentence_min": int(sent_arr.min()),
        "sentence_max": int(sent_arr.max()),
        "sentence_mean": float(sent_arr.mean()),
        "sentence_median": float(np.median(sent_arr)),
        "sentence_std": float(sent_arr.std()),
        "paragraph_count": len(all_para_lengths),
        "paragraph_min": int(para_arr.min()),
        "paragraph_max": int(para_arr.max()),
        "paragraph_mean": float(para_arr.mean()),
        "paragraph_median": float(np.median(para_arr)),
        "paragraph_std": float(para_arr.std()),
        "dialogue_ratio": total_quoted / total_chars if total_chars > 0 else 0,
        "total_quoted_chars": total_quoted,
        "total_chars": total_chars,
        "double_quotes": total_double_quotes,
        "single_quotes": total_single_quotes,
        "sentence_lengths": sent_arr,
        "para_lengths": para_arr,
    }


def print_results(results: dict):
    """Print analysis results."""
    print_header("Structural Analysis")

    print_subheader("Sentence Length (in words)")
    print(f"Total sentences: {results['sentence_count']:,}")
    print(f"Min:    {results['sentence_min']:,}")
    print(f"Max:    {results['sentence_max']:,}")
    print(f"Mean:   {results['sentence_mean']:.1f}")
    print(f"Median: {results['sentence_median']:.1f}")
    print(f"Std:    {results['sentence_std']:.1f}")

    print_subheader("Paragraph Length (in sentences)")
    print(f"Total paragraphs: {results['paragraph_count']:,}")
    print(f"Min:    {results['paragraph_min']:,}")
    print(f"Max:    {results['paragraph_max']:,}")
    print(f"Mean:   {results['paragraph_mean']:.1f}")
    print(f"Median: {results['paragraph_median']:.1f}")
    print(f"Std:    {results['paragraph_std']:.1f}")

    print_subheader("Dialogue Analysis")
    print(f"Dialogue ratio:     {results['dialogue_ratio']*100:.1f}% of text in quotes")
    print(f"Quoted characters:  {results['total_quoted_chars']:,}")
    print(f"Total characters:   {results['total_chars']:,}")

    print_subheader("Quote Style")
    total_quotes = results['double_quotes'] + results['single_quotes']
    if total_quotes > 0:
        double_pct = 100 * results['double_quotes'] / total_quotes
        single_pct = 100 * results['single_quotes'] / total_quotes
    else:
        double_pct = single_pct = 0
    print(f"Double quotes (\"):  {results['double_quotes']:,} ({double_pct:.1f}%)")
    print(f"Single quotes ('):  {results['single_quotes']:,} ({single_pct:.1f}%)")

    # Sentence length distribution
    print_subheader("Sentence Length Distribution")
    sent_arr = results['sentence_lengths']
    bins = [0, 5, 10, 15, 20, 30, 50, 100, float('inf')]
    labels = ["1-5", "6-10", "11-15", "16-20", "21-30", "31-50", "51-100", "100+"]
    print(f"{'Range':<12} {'Count':>12} {'%':>8}")
    print("-" * 35)
    for i in range(len(bins) - 1):
        count = np.sum((sent_arr > bins[i]) & (sent_arr <= bins[i+1]))
        pct = 100 * count / len(sent_arr) if len(sent_arr) > 0 else 0
        print(f"{labels[i]:<12} {count:>12,} {pct:>7.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Structural analysis of dataset")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input file path")
    args = parser.parse_args()

    samples = load_samples(args.input)
    print(f"Loaded {len(samples):,} samples")

    results = analyze_structural(samples)
    print_results(results)


if __name__ == "__main__":
    main()
