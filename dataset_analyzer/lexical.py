#!/usr/bin/env python3
"""Lexical/vocabulary analysis with multiprocessing."""

import argparse
from collections import Counter

from .utils import (
    DEFAULT_INPUT,
    load_samples,
    tokenize_words,
    run_parallel,
    print_header,
    print_subheader,
)

# Top 10K English words (simplified - using top 1000 common words for OOV check)
# Full list would be loaded from file in production
TOP_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "was", "are", "been", "has", "had", "were", "said", "did", "get",
    "may", "been", "being", "does", "done", "got", "goes", "going", "made", "making",
    "man", "woman", "hand", "eye", "head", "face", "thing", "place", "life", "world",
    "house", "room", "door", "night", "day", "water", "long", "little", "own", "old",
    "right", "big", "high", "small", "large", "next", "young", "last", "great", "same",
    "few", "many", "much", "more", "most", "such", "very", "still", "again", "never",
    "always", "often", "here", "where", "why", "down", "off", "away", "before", "through",
    "around", "while", "once", "during", "without", "between", "under", "against", "another", "each",
    "every", "both", "either", "neither", "should", "might", "must", "shall", "need", "seem",
    "let", "keep", "put", "set", "run", "move", "live", "believe", "hold", "bring",
    "happen", "write", "provide", "sit", "stand", "lose", "pay", "meet", "include", "continue",
    "learn", "change", "lead", "understand", "watch", "follow", "stop", "create", "speak", "read",
    "allow", "add", "spend", "grow", "open", "walk", "win", "offer", "remember", "love",
    "consider", "appear", "buy", "wait", "serve", "die", "send", "expect", "build", "stay",
    "fall", "cut", "reach", "kill", "remain", "suggest", "raise", "pass", "sell", "require",
    "report", "decide", "pull", "turn", "ask", "tell", "show", "try", "leave", "call",
    "feel", "seem", "begin", "start", "help", "become", "end", "point", "part", "kind",
    "hear", "mean", "find", "give", "play", "run", "move", "need", "might", "something",
    "anything", "nothing", "everything", "someone", "anyone", "everyone", "nobody", "somebody", "everybody",
    "herself", "himself", "itself", "myself", "yourself", "themselves", "ourselves", "though", "although", "however",
    "whether", "rather", "quite", "already", "enough", "too", "perhaps", "maybe", "certainly", "probably",
    "actually", "really", "simply", "almost", "soon", "ago", "yet", "ever", "just", "only",
}


def count_words(sample: str) -> Counter:
    """Count words in a single sample."""
    words = tokenize_words(sample.lower())
    return Counter(words)


def analyze_lexical(samples: list) -> dict:
    """Run full lexical analysis."""
    print(f"Analyzing {len(samples):,} samples...")

    # Count words in parallel
    counters = run_parallel(count_words, samples)

    # Aggregate
    total_words = Counter()
    for c in counters:
        total_words.update(c)

    total_count = sum(total_words.values())
    vocab_size = len(total_words)

    # Type-token ratio
    ttr = vocab_size / total_count if total_count > 0 else 0

    # Hapax legomena (words appearing once)
    hapax = [w for w, c in total_words.items() if c == 1]
    hapax_count = len(hapax)

    # OOV rate
    oov_words = [w for w in total_words if w not in TOP_ENGLISH_WORDS]
    oov_count = sum(total_words[w] for w in oov_words)
    oov_rate = oov_count / total_count if total_count > 0 else 0

    # Zipf's law check (top words should follow power law)
    top_100 = total_words.most_common(100)

    return {
        "total_words": total_count,
        "vocab_size": vocab_size,
        "ttr": ttr,
        "hapax_count": hapax_count,
        "hapax_pct": hapax_count / vocab_size if vocab_size > 0 else 0,
        "oov_rate": oov_rate,
        "top_100": top_100,
        "word_counts": total_words,
    }


def print_results(results: dict):
    """Print analysis results."""
    print_header("Lexical Analysis")

    print(f"\nTotal words:      {results['total_words']:,}")
    print(f"Vocabulary size:  {results['vocab_size']:,} unique words")
    print(f"Type-token ratio: {results['ttr']:.6f}")
    print(f"Hapax legomena:   {results['hapax_count']:,} ({results['hapax_pct']*100:.1f}% of vocabulary)")
    print(f"OOV rate:         {results['oov_rate']*100:.2f}% (vs top common words)")

    print_subheader("Top 50 Words")
    print(f"{'Rank':<6} {'Word':<20} {'Count':>12} {'%':>8}")
    print("-" * 50)
    for i, (word, count) in enumerate(results['top_100'][:50], 1):
        pct = 100 * count / results['total_words']
        print(f"{i:<6} {word:<20} {count:>12,} {pct:>7.2f}%")

    print_subheader("Zipf's Law Check (rank × frequency should be ~constant)")
    print(f"{'Rank':<6} {'Word':<15} {'Freq':>12} {'Rank×Freq':>15}")
    print("-" * 50)
    for rank in [1, 2, 5, 10, 20, 50, 100]:
        if rank <= len(results['top_100']):
            word, freq = results['top_100'][rank-1]
            print(f"{rank:<6} {word:<15} {freq:>12,} {rank*freq:>15,}")


def main():
    parser = argparse.ArgumentParser(description="Lexical analysis of dataset")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input file path")
    args = parser.parse_args()

    samples = load_samples(args.input)
    print(f"Loaded {len(samples):,} samples")

    results = analyze_lexical(samples)
    print_results(results)


if __name__ == "__main__":
    main()
