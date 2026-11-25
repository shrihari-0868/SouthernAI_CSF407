"""Train and compare classifiers on the representations from stage 2.

Run: python run_classifiers.py REPRESENTATIONS_DIR OUT_DIR"""
import sys, time, warnings, numpy as np, pandas as pd, scipy.sparse as sp
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier

D, OUT = sys.argv[1], sys.argv[2]
mt = pd.read_csv(f'{D}/meta_train.csv'); ms = pd.read_csv(f'{D}/meta_test.csv')
ytr = np.load(f'{D}/y_train.npy'); yte = np.load(f'{D}/y_test.npy')
assert (ytr == mt.label.values).all() and (yte == ms.label.values).all()
# one group per pair (same genus) so a pathogen and its close relative stay together
gen_tr = mt.genus.values; gid_tr = mt.genome_id.values
org = ms.groupby('genome_id').organism.first().to_dict()
lab = ms.groupby('genome_id').label.first().to_dict()

def load(rep, k, split):
    if rep == 'w2v':
        return np.load(f'{D}/k{k}_w2v_{split}.npy').astype(np.float32)
    X = sp.load_npz(f'{D}/k{k}_{rep}_{split}.npz').toarray().astype(np.float32)
    if rep == 'bow':                      # counts -> frequencies (fragments all ~1000 bp, so this just rescales)
        X = X / X.sum(1, keepdims=True)
    return X

def make(name):
    if name == 'LogReg': return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    if name == 'DecisionTree': return DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=0)
    if name == 'RandomForest': return RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=2, random_state=0)
    if name == 'MLP': return make_pipeline(StandardScaler(), MLPClassifier((64,), alpha=0.1, max_iter=40, random_state=0))
MODELS = ['LogReg', 'DecisionTree', 'RandomForest', 'MLP']

def cv_scores(name, X, groups, n):
    s = []
    for a, b in GroupKFold(n).split(X, ytr, groups):
        s.append((make(name).fit(X[a], ytr[a]).predict(X[b]) == ytr[b]).mean())
    return np.mean(s), np.std(s)

rows, pg_rows = [], []
t0 = time.time()
for k in (3, 4, 6):
    for rep in ('bow', 'tfidf', 'w2v'):
        Xtr, Xte = load(rep, k, 'train'), load(rep, k, 'test')
        for name in MODELS:
            m = make(name).fit(Xtr, ytr)
            ptr, pte = m.predict(Xtr), m.predict(Xte)
            pg = pd.DataFrame({'g': ms.genome_id.values, 'ok': pte == yte, 'p': pte}).groupby('g').agg(acc=('ok', 'mean'), frac1=('p', 'mean'))
            gcorrect = int(sum(((pg.frac1[g] > .5) == bool(lab[g])) for g in pg.index))
            for g in pg.index:
                pg_rows.append(dict(rep=rep, k=k, model=name, genome_id=g, organism=org[g], label=lab[g], acc=pg.acc[g]))
            cg = cv_scores(name, Xtr, gid_tr, 6); cp = cv_scores(name, Xtr, gen_tr, 7)
            rows.append(dict(rep=rep, k=k, model=name, train_acc=(ptr == ytr).mean(), test_acc=(pte == yte).mean(),
                             test_genomes_correct_of6=gcorrect, cv_genome_mean=cg[0], cv_genome_std=cg[1],
                             cv_pair_mean=cp[0], cv_pair_std=cp[1]))
            r = rows[-1]; print(f"{rep} k={k} {name}: train {r['train_acc']:.3f} test {r['test_acc']:.3f} g{gcorrect}/6 cvG {cg[0]:.3f} cvP {cp[0]:.3f} [{time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/results_task4.csv', index=False)
        pd.DataFrame(pg_rows).to_csv(f'{OUT}/per_genome_accuracy.csv', index=False)
print('GRID DONE', flush=True)
