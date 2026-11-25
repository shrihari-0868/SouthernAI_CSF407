"""Fair grouped CV: leave one genus-group out (pairs stay together), class_weight=balanced, pooled out-of-fold accuracy.
Run: python3 -I run_cv_fix.py DATA_DIR OUT_DIR"""
import sys, warnings, numpy as np, pandas as pd, scipy.sparse as sp
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneGroupOut
D, OUT = sys.argv[1], sys.argv[2]
mt = pd.read_csv(f'{D}/meta_train.csv'); ytr = np.load(f'{D}/y_train.npy'); gen = mt.genus.values
def load(rep, k):
    if rep == 'w2v': return np.load(f'{D}/k{k}_w2v_train.npy').astype(np.float32)
    X = sp.load_npz(f'{D}/k{k}_{rep}_train.npz').toarray().astype(np.float32)
    return X / X.sum(1, keepdims=True) if rep == 'bow' else X
def oof(make, X):
    p = np.zeros(len(ytr), int)
    for a, b in LeaveOneGroupOut().split(X, ytr, gen): p[b] = make().fit(X[a], ytr[a]).predict(X[b])
    return (p == ytr).mean(), pd.Series(p == ytr).groupby(mt.genome_id.values).mean()
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))
DT = lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, class_weight='balanced', random_state=0)
rows = []
g = mt.gc.values[:, None].astype(np.float32)
for nm, mk, X in [('Majority', lambda: DummyClassifier(strategy='prior'), g), ('GC-only LogReg', lambda: LogisticRegression(class_weight='balanced'), g)]:
    a, _ = oof(mk, X); rows.append(dict(rep='-', k='-', model=nm, cv_leave_group_out=a)); print(rows[-1], flush=True)
for k in (3, 4, 6):
    for rep in ('bow', 'tfidf', 'w2v'):
        X = load(rep, k)
        for nm, mk in [('LogReg', LR), ('DecisionTree', DT)]:
            a, _ = oof(mk, X); rows.append(dict(rep=rep, k=k, model=nm, cv_leave_group_out=a)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/cv_leave_group_out_balanced.csv', index=False); print('CV DONE')
