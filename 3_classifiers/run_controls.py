"""Task 4 controls: baselines, GC-removed features, and a random-label null. Run: python3 -I run_controls.py DATA_DIR OUT_DIR"""
import sys, warnings, numpy as np, pandas as pd, scipy.sparse as sp
warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
D, OUT = sys.argv[1], sys.argv[2]
mt = pd.read_csv(f'{D}/meta_train.csv'); ms = pd.read_csv(f'{D}/meta_test.csv')
ytr = np.load(f'{D}/y_train.npy'); yte = np.load(f'{D}/y_test.npy')
gtr, gte = mt.gc.values.astype(np.float32), ms.gc.values.astype(np.float32)
gid, gen = mt.genome_id.values, mt.genus.values

def load(rep, k, split):
    if rep == 'w2v': return np.load(f'{D}/k{k}_w2v_{split}.npy').astype(np.float32)
    X = sp.load_npz(f'{D}/k{k}_{rep}_{split}.npz').toarray().astype(np.float32)
    return X / X.sum(1, keepdims=True) if rep == 'bow' else X

class GCResidualizer(BaseEstimator, TransformerMixin):
    """Last column of X is GC. Remove the part of every feature that a quadratic function of GC explains (fit on training rows only)."""
    def _g(self, X): g = X[:, -1:]; return np.hstack([g, g**2])
    def fit(self, X, y=None): self.r_ = LinearRegression().fit(self._g(X), X[:, :-1]); return self
    def transform(self, X): return X[:, :-1] - self.r_.predict(self._g(X))

def cv(make, X, groups, n):
    return np.mean([(make().fit(X[a], ytr[a]).predict(X[b]) == ytr[b]).mean() for a, b in GroupKFold(n).split(X, ytr, groups)])

rows = []
# 1. baselines
for nm, mk, Xa, Xb in [('Majority-class', lambda: DummyClassifier(), gtr[:, None], gte[:, None]),
                       ('GC-only LogReg', lambda: LogisticRegression(), gtr[:, None], gte[:, None])]:
    m = mk().fit(Xa, ytr)
    rows.append(dict(experiment=nm, rep='-', k='-', model=nm, train_acc=(m.predict(Xa) == ytr).mean(), test_acc=(m.predict(Xb) == yte).mean(),
                     cv_genome=cv(mk, Xa, gid, 6), cv_pair=cv(mk, Xa, gen, 7)))
    print(rows[-1], flush=True)
# 2. GC removed from the features
for k in (3, 4, 6):
    for rep in ('bow', 'tfidf', 'w2v'):
        Xa = np.hstack([load(rep, k, 'train'), gtr[:, None]]); Xb = np.hstack([load(rep, k, 'test'), gte[:, None]])
        for name, mk in [('LogReg', lambda: make_pipeline(GCResidualizer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))),
                         ('DecisionTree', lambda: make_pipeline(GCResidualizer(), DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=0)))]:
            m = mk().fit(Xa, ytr)
            rows.append(dict(experiment='GC removed', rep=rep, k=k, model=name, train_acc=(m.predict(Xa) == ytr).mean(),
                             test_acc=(m.predict(Xb) == yte).mean(), cv_genome=cv(mk, Xa, gid, 6), cv_pair=cv(mk, Xa, gen, 7)))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/controls_task4.csv', index=False)

# 3. random-label null: give genomes random labels (9/9 train, 3/3 test) and see what "accuracy" looks like by luck
rng = np.random.default_rng(0); null = []
trg, teg = np.unique(gid), np.unique(ms.genome_id.values)
for rep, k in [('w2v', 6), ('bow', 3)]:
    Xtr, Xte = load(rep, k, 'train'), load(rep, k, 'test')
    for i in range(100):
        lt = dict(zip(trg, rng.permutation([0]*9 + [1]*9))); le = dict(zip(teg, rng.permutation([0]*3 + [1]*3)))
        a = np.array([lt[g] for g in gid]); b = np.array([le[g] for g in ms.genome_id.values])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xtr, a)
        null.append(dict(rep=rep, k=k, run=i, test_acc=(m.predict(Xte) == b).mean()))
n = pd.DataFrame(null); n.to_csv(f'{OUT}/null_random_labels.csv', index=False)
print(n.groupby(['rep', 'k']).test_acc.describe(percentiles=[.5, .9, .95]).round(3).to_string(), flush=True)
print('CONTROLS DONE', flush=True)
