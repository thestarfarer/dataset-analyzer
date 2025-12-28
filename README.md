# Dataset Analyzer

Know what's in your training data before you poison a model with it.

A multiprocessed inspection toolkit for text datasets. Throws stats at your data so you can catch problems before they become weights.

## Why This Exists

I built this while preparing data for [Ministral-3-14B-writer](https://huggingface.co/thestarfarer/Ministral-3-14B-writer). Needed to answer questions like:

- How diverse is the vocabulary actually?
- What dialogue tags dominate? (if everything is "said said said", the model learns that)
- How much of this is duplicate or near-duplicate?
- What's the sentence length distribution? Paragraph structure?
- Any weird artifacts? All-caps spam? Broken punctuation?

Couldn't find a tool that did all of this fast on large datasets. So here we are.

## Usage

```bash
# Full analysis
python -m dataset_analyzer.run_all -i data.txt

# Individual modules
python -m dataset_analyzer.lexical -i data.txt
python -m dataset_analyzer.ngrams -i data.txt
python -m dataset_analyzer.structural -i data.txt
python -m dataset_analyzer.quality -i data.txt
python -m dataset_analyzer.char_freq -i data.txt
```

## Data Format

Text file with samples separated by `<BREAK>`:

```
First sample text here...
<BREAK>
Second sample text here...
<BREAK>
Third sample...
```

## What It Measures

| Module | What You Learn |
|--------|----------------|
| `lexical` | Vocabulary size, type-token ratio, hapax legomena, OOV rate, Zipf's law fit |
| `ngrams` | Bigrams, trigrams, dialogue tag distribution, repeated phrases |
| `structural` | Sentence/paragraph lengths, dialogue ratio, quote styles |
| `quality` | Unusual punctuation, all-caps words, number density, duplicate sentences |
| `char_freq` | Character frequency distribution |

## Output

Prints to stdout. Redirect if you want to keep it:

```bash
python -m dataset_analyzer.run_all -i data.txt > analysis.md
```

<details>
<summary><b>Example output (wikitext-103, 500 samples)</b></summary>

```
Loading samples from wikitext_sample.txt...
Loaded 500 samples in 0.0s

================================================================================
Running LEXICAL analysis...
================================================================================

Total words:      55,272
Vocabulary size:  10,000 unique words
Type-token ratio: 0.180923
Hapax legomena:   5,408 (54.1% of vocabulary)
OOV rate:         53.55% (vs top common words)

--- Top 50 Words ---
Rank   Word                        Count        %
--------------------------------------------------
1      the                         4,091    7.40%
2      of                          1,866    3.38%
3      and                         1,723    3.12%
4      in                          1,516    2.74%
5      a                           1,290    2.33%
...

--- Zipf's Law Check (rank × frequency should be ~constant) ---
Rank   Word                    Freq       Rank×Freq
--------------------------------------------------
1      the                    4,091           4,091
2      of                     1,866           3,732
5      a                      1,290           6,450
10     as                       488           4,880
20     at                       279           5,580
50     other                     79           3,950
100    day                       46           4,600

================================================================================
Running N-GRAM analysis...
================================================================================

--- Top 50 Word Bigrams ---
Rank   Bigram                                Count
--------------------------------------------------
1      of the                                  582
2      in the                                  345
3      to the                                  156
4      on the                                  150
5      and the                                 133
...

--- Top 50 Word Trigrams ---
Rank   Trigram                                         Count
-------------------------------------------------------
1      one of the                                         33
2      a number of                                        20
3      as well as                                         15
4      end of the                                         14
5      part of the                                        14
6      the united states                                  14
...

--- Dialogue Tags Frequency ---
Tag                    Count        %
----------------------------------------
said                      34    14.3%
called                    32    13.5%
began                     31    13.1%
stated                    21     8.9%
continued                 21     8.9%
...

================================================================================
Running STRUCTURAL analysis...
================================================================================

--- Sentence Length (in words) ---
Total sentences: 2,661
Min:    0
Max:    93
Mean:   20.8
Median: 20.0
Std:    11.5

--- Dialogue Analysis ---
Dialogue ratio:     17.7% of text in quotes

--- Quote Style ---
Double quotes ("):  995 (71.1%)
Single quotes ('):  404 (28.9%)

--- Sentence Length Distribution ---
Range               Count        %
-----------------------------------
1-5                   138     5.2%
6-10                  291    10.9%
11-15                 435    16.3%
16-20                 509    19.1%
21-30                 779    29.3%
31-50                 422    15.9%
51-100                 43     1.6%

================================================================================
Running QUALITY analysis...
================================================================================

--- All-Caps Words (3+ chars) ---
Total all-caps occurrences: 262
Unique all-caps words: 102

Top 30 all-caps words:
Word                        Count
-----------------------------------
NBA                            21
NHL                            17
AML                            17
...

--- Number Density ---
Numeric tokens:  1,950
Total tokens:    57,222
Number density:  3.408%

--- Duplicate Sentences (appearing in 2+ samples) ---
Total duplicate sentences: 0

================================================================================
ALL ANALYSES COMPLETED in 0.8s
================================================================================
```

</details>

## Performance

Multiprocessed across all cores. Handles millions of samples without dying. The bottleneck is usually disk I/O on the initial load.

## Requirements

```
numpy
```

That's it.

## License

MIT
