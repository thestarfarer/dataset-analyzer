# dataset-analyzer

Multiprocessing toolkit for analyzing text training datasets. Lexical stats, n-grams, structural patterns, quality metrics.

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

## Analyses

| Module | Metrics |
|--------|---------|
| `lexical` | Vocabulary size, TTR, hapax legomena, OOV rate, Zipf's law |
| `ngrams` | Bigrams, trigrams, dialogue tags, repeated phrases |
| `structural` | Sentence/paragraph lengths, dialogue ratio, quote styles |
| `quality` | Unusual punctuation, all-caps, number density, duplicate sentences |
| `char_freq` | Character frequency distribution |

## Output

All stats printed to stdout. Redirect to file:

```bash
python -m dataset_analyzer.run_all -i data.txt > analysis.md
```

## Used for

- [Ministral-3-14B-writer](https://huggingface.co/thestarfarer/Ministral-3-14B-writer)

## Requirements

```
numpy
```

## License

MIT
