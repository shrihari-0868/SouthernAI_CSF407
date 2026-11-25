# Representations: turning fragments into vectors

Three ways of describing a 1,000 bp fragment as numbers, all built from overlapping k-mers for
k = 3, 4 and 6.

- **Bag-of-Words** — how often each of the 4^k possible words occurs. The full vocabulary is
  supplied explicitly, so a word that never appears still gets a column of zeros and its absence
  stays visible to the model.
- **TF-IDF** — the same counts, with words that occur in nearly every fragment weighted down.
- **Word2Vec** — a 100-dimensional vector learned for each word from the words around it, then
  averaged across a fragment, which is the sequence analogue of a document embedding.

Everything is fitted on the 18 training genomes only. The six held-out genomes contribute nothing
to the vocabulary, the IDF weights or the Word2Vec model.

## Running it

```bash
python build_representations.py ../data/dataset.csv out
```

Takes a few minutes, nearly all of it Word2Vec. Writes into `out/`:

| File | Contents |
|---|---|
| `k{k}_bow_{split}.npz` | raw counts, sparse float32 |
| `k{k}_tfidf_{split}.npz` | TF-IDF weighted, sparse float32 |
| `k{k}_w2v_{split}.npy` | mean Word2Vec vector per fragment, dense, 100 dimensions |
| `meta_{split}.csv` | fragment id, genome, genus, organism, label, GC |
| `y_{split}.npy` | labels |
| `k{k}_vocab.csv` | the word list, in column order |
| `k{k}_kmer_embeddings.csv` | the learned vector for each individual word |

Row order follows `dataset.csv` within each split, so row *i* of a matrix always corresponds to
row *i* of `meta_{split}.csv`. Nothing is shuffled, and the later stages depend on that.

These matrices total about 84 MB and are therefore not stored in the repository. Rebuild them
before running stages 3 and 4.

## Reproducibility

Checked against the matrices used to produce the reported results:

| Output | Agreement |
|---|---|
| Bag-of-Words counts, all k, both splits | **bit-identical** |
| TF-IDF, all k, both splits | **bit-identical** |
| `meta_*.csv`, `y_*.npy` | **identical** |
| Word2Vec vectors | numerically different — see below |

Word2Vec initialises randomly and the seed of the original run was not recorded, so those vectors
differ between runs. Re-running the classifiers on freshly built matrices reproduces every
non-Word2Vec result to within 0.002 accuracy, while Word2Vec rows move by up to 0.065. That is
smaller than the gap between the models and the random-label null, so no conclusion in the report
depends on it. `WORKERS = 1` and a fixed seed at the top of the script make a run repeatable on
one machine.

## What the comparison showed

| Representation | k | Train acc | Test acc |
|---|---|---|---|
| BoW | 3 | 0.7367 | 0.5846 |
| TF-IDF | 3 | 0.7344 | 0.5863 |
| Word2Vec | 3 | 0.7249 | 0.5996 |
| BoW | 4 | 0.7754 | 0.5754 |
| TF-IDF | 4 | 0.7756 | 0.5729 |
| Word2Vec | 4 | 0.7535 | 0.5687 |
| BoW | 6 | 1.0000 | 0.5558 |
| TF-IDF | 6 | 1.0000 | 0.5558 |
| Word2Vec | 6 | 0.7513 | 0.5925 |
| GC content only | – | – | 0.5179 |

(`results_table.csv`, logistic regression on the fixed six-genome test split.)

No representation is reliably better than another, and all of them sit close to a model that knows
nothing but GC content. The k = 6 rows are the clearest warning: perfect accuracy on the training
genomes and the worst accuracy on the unseen ones, which is memorisation rather than learning.
