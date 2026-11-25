"""Build the sequence representations that stages 3-4 consume.

    python build_representations.py ../data/dataset.csv out

Writes, for k = 3, 4 and 6, and for the train and test split separately:
    k{k}_bow_{split}.npz     raw k-mer counts, sparse, float32
    k{k}_tfidf_{split}.npz   TF-IDF weighted, sparse, float32
    k{k}_w2v_{split}.npy     mean Word2Vec vector per fragment, dense, 100 dimensions
plus meta_{split}.csv, y_{split}.npy, k{k}_vocab.csv and k{k}_kmer_embeddings.csv.

Row order follows dataset.csv within each split, so matrix row i always corresponds to row i of
meta_{split}.csv. Nothing is shuffled: the downstream scripts rely on this.

Everything is fitted on the training genomes only. The six test genomes contribute nothing to the
vocabulary, the IDF weights or the Word2Vec model.

Reproducibility: the counts and TF-IDF matrices reproduce the original run exactly (to float32
precision). Word2Vec involves random initialisation and the seed of the original run was not
recorded, so those vectors differ numerically between runs while giving the same accuracies to
within a percentage point. Set WORKERS = 1 below to make a run repeatable on one machine.
"""
import sys, os, itertools
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import Word2Vec

CSV, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
KS = (3, 4, 6)
DIM, WINDOW, EPOCHS, SEED, WORKERS = 100, 5, 5, 42, 1

df = pd.read_csv(CSV)
split = {s: df[df.split_suggested == s].reset_index(drop=True) for s in ('train', 'test')}
META = ['fragment_id', 'genome_id', 'genus', 'organism', 'label', 'gc']
for s, d in split.items():
    d[META].to_csv(f'{OUT}/meta_{s}.csv', index=False)
    np.save(f'{OUT}/y_{s}.npy', d.label.to_numpy(np.int64))
    print(f'{s}: {len(d)} fragments from {d.genome_id.nunique()} genomes', flush=True)


def vocabulary(k):
    return [''.join(p) for p in itertools.product('ACGT', repeat=k)]


def tokens(seq, k):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


for k in KS:
    words = vocabulary(k)
    cv = CountVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False,
                         vocabulary=words, dtype=np.int32)
    counts = {s: cv.transform(d.sequence) for s, d in split.items()}

    # TF-IDF weights are learned on the training fragments and applied unchanged to the test ones
    tfidf = TfidfTransformer().fit(counts['train'])
    for s in split:
        sp.save_npz(f'{OUT}/k{k}_bow_{s}.npz', counts[s].astype(np.float32))
        sp.save_npz(f'{OUT}/k{k}_tfidf_{s}.npz', tfidf.transform(counts[s]).astype(np.float32))

    # Word2Vec treats each fragment as a sentence of overlapping k-mers and learns a vector per
    # word from the words around it; a fragment is then the average of its words' vectors.
    sentences = [tokens(s_, k) for s_ in split['train'].sequence]
    w2v = Word2Vec(sentences, vector_size=DIM, window=WINDOW, min_count=1, sg=1,
                   epochs=EPOCHS, seed=SEED, workers=WORKERS)
    table = np.zeros((len(words), DIM), np.float32)
    for i, w in enumerate(words):
        if w in w2v.wv:
            table[i] = w2v.wv[w]
    index = {w: i for i, w in enumerate(words)}
    for s, d in split.items():
        M = np.zeros((len(d), DIM), np.float32)
        for r, seq in enumerate(d.sequence):
            idx = [index[t] for t in tokens(seq, k) if t in index]
            if idx:
                M[r] = table[idx].mean(0)
        np.save(f'{OUT}/k{k}_w2v_{s}.npy', M)

    pd.DataFrame({'kmer': words}).to_csv(f'{OUT}/k{k}_vocab.csv', index=False)
    emb = pd.DataFrame(table, columns=[str(i) for i in range(DIM)])
    emb.insert(0, 'kmer', words)
    emb.to_csv(f'{OUT}/k{k}_kmer_embeddings.csv', index=False)
    print(f'k={k}: {len(words)} words, matrices written', flush=True)

print('done ->', OUT, flush=True)
