"""What has the model learned? Coefficients, their stability, the matched-pair permutation test
and genome-identity probes.

Run: python run_interpretation.py REPRESENTATIONS_DIR OUT_DIR"""
import sys, itertools, warnings, numpy as np, pandas as pd, scipy.sparse as sp
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GroupKFold, cross_val_score

D, OUT = sys.argv[1], sys.argv[2]
mt = pd.read_csv(f'{D}/meta_train.csv'); ms = pd.read_csv(f'{D}/meta_test.csv')
ytr = np.load(f'{D}/y_train.npy'); yte = np.load(f'{D}/y_test.npy')
meta = pd.concat([mt, ms], ignore_index=True)
gen_tr = mt.genus.values; gid_tr, gid_te = mt.genome_id.values, ms.genome_id.values
ginfo = meta.groupby('genome_id').agg(genus=('genus', 'first'), label=('label', 'first'), organism=('organism', 'first'))
pairs = [g for g, d in ginfo.groupby('genus') if len(d) == 2 and d.label.nunique() == 2]
print('pairs', pairs, flush=True)
test_pairs = [g for g in pairs if g in set(ms.genus)]; train_pairs = [g for g in pairs if g not in test_pairs]

def freq(k, split):
    X = sp.load_npz(f'{D}/k{k}_bow_{split}.npz').toarray().astype(np.float32)
    return X / X.sum(1, keepdims=True)
words = {k: [''.join(p) for p in itertools.product('ACGT', repeat=k)] for k in (3, 4, 6)}
wgc = {k: np.array([sum(c in 'GC' for c in w) / k for w in words[k]]) for k in words}
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
summary = []
topk_rows, pair_rows = [], []

for k in (3, 4, 6):
    Xtr, Xte = freq(k, 'train'), freq(k, 'test'); Xall = np.vstack([Xtr, Xte]); gall = np.concatenate([gid_tr, gid_te])
    gcfrag = meta.gc.values
    # sanity: column order. Words richer in G/C should correlate more positively with fragment GC
    col_corr = np.array([np.corrcoef(Xall[:, j], gcfrag)[0, 1] for j in range(Xall.shape[1])])
    order_check = np.corrcoef(col_corr, wgc[k])[0, 1]
    # ---- A. coefficients
    m = LR().fit(Xtr, ytr); coef = m.named_steps['logisticregression'].coef_[0]
    # stability: refit leaving one genus-group out; share of folds with same sign as the full model
    signs = []
    for a, b in LeaveOneGroupOut().split(Xtr, ytr, gen_tr):
        signs.append(np.sign(LR().fit(Xtr[a], ytr[a]).named_steps['logisticregression'].coef_[0]))
    stab = (np.array(signs) == np.sign(coef)).mean(0)
    # ---- B. pair-controlled differences (genome-mean frequency, pathogen minus non-pathogen)
    G = pd.DataFrame(Xall).groupby(gall).mean()
    sd = G.std(0).values + 1e-12
    dif = {}
    for g in pairs:
        ids = ginfo[ginfo.genus == g]; p = ids.index[ids.label == 1][0]; n = ids.index[ids.label == 0][0]
        dif[g] = (G.loc[p].values - G.loc[n].values) / sd          # standardized within-pair difference
    Dm = np.array([dif[g] for g in pairs])                           # pairs x words
    allsame = (np.abs(np.sign(Dm).sum(0)) == len(pairs))
    # exact permutation: flip which genome of each pair is "pathogen" (2^6 = 64 ways) and recount
    cnts = []
    for flips in itertools.product([1, -1], repeat=len(pairs)):
        S = np.sign(Dm) * np.array(flips)[:, None]; cnts.append(int((np.abs(S.sum(0)) == len(pairs)).sum()))
    obs = int(allsame.sum()); pperm = np.mean(np.array(cnts) >= obs)
    rho_all = spearmanr(coef, Dm.mean(0)).correlation
    tp = [pairs.index(g) for g in test_pairs]; trp = [pairs.index(g) for g in train_pairs]
    rho_test = spearmanr(coef, Dm[tp].mean(0)).correlation      # pairs the model never saw
    rho_train = spearmanr(coef, Dm[trp].mean(0)).correlation
    rho_gc_coef = spearmanr(coef, wgc[k]).correlation
    r2_gc = np.corrcoef(coef, wgc[k])[0, 1] ** 2
    summary.append(dict(k=k, column_order_check_corr=order_check, n_words=len(words[k]), words_same_sign_in_all_6_pairs=obs,
                        median_count_when_pair_labels_flipped=int(np.median(cnts)), perm_p=pperm,
                        spearman_coef_vs_pairdiff_all6=rho_all, spearman_coef_vs_pairdiff_trainpairs=rho_train,
                        spearman_coef_vs_pairdiff_TESTpairs=rho_test, spearman_coef_vs_wordGC=rho_gc_coef, R2_coef_by_wordGC=r2_gc))
    print(summary[-1], flush=True)
    df = pd.DataFrame(dict(k=k, word=words[k], coef_std=coef, word_gc=wgc[k], loo_sign_stability=stab,
                           pair_mean_std_diff=Dm.mean(0), pairs_agree_with_coef=(np.sign(Dm) == np.sign(coef)[None, :]).sum(0), all6_same_sign=allsame))
    df['rank'] = df.coef_std.rank(ascending=False).astype(int)
    topk_rows.append(df.sort_values('coef_std', ascending=False))
    # per-pair rows
    for g in pairs:
        pair_rows.append(dict(k=k, pair=g, in_training=g in train_pairs, spearman_coef_vs_this_pair=spearmanr(coef, dif[g]).correlation))

pd.concat(topk_rows).to_csv(f'{OUT}/task5_kmer_coefficients.csv', index=False)
pd.DataFrame(summary).to_csv(f'{OUT}/task5_pair_consistency_summary.csv', index=False)
pd.DataFrame(pair_rows).to_csv(f'{OUT}/task5_per_pair_agreement.csv', index=False)

# ---- C/D. probes and unseen-pair tests on representative representations
def load_rep(rep, k, split):
    return np.load(f'{D}/k{k}_w2v_{split}.npy') if rep == 'w2v' else freq(k, split)
probe = []
for rep, k in [('bow', 3), ('bow', 4), ('w2v', 3), ('w2v', 6)]:
    Xtr, Xte = load_rep(rep, k, 'train'), load_rep(rep, k, 'test')
    # (i) can the features identify the genome? random fragment-level split, 18-way
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    gid_acc = cross_val_score(LR(), Xtr, gid_tr, cv=skf).mean()
    # (ii) pathogen label, random fragment split (fragments of a genome on both sides) vs genome-grouped split
    rand_acc = cross_val_score(LR(), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    grp_acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced')), Xtr, ytr, cv=GroupKFold(6), groups=gid_tr).mean()
    # (iii) within-pair AUC on genomes the model never saw: can it rank the pathogen above its harmless relative?
    aucs = {}
    for g in pairs:
        if g in test_pairs: Xa, ya, Xb, yb = Xtr, ytr, Xte, yte; sel = (ms.genus.values == g)
        else:
            tr = (gen_tr != g); Xa, ya = Xtr[tr], ytr[tr]; Xb, yb = Xtr[~tr], ytr[~tr]; sel = np.ones(len(yb), bool)
        s = LR().fit(Xa, ya).predict_proba(Xb[sel] if g in test_pairs else Xb)[:, 1]
        aucs[g] = roc_auc_score(yb[sel] if g in test_pairs else yb, s)
    probe.append(dict(rep=rep, k=k, genome_identity_acc_random_split=gid_acc, chance_genome_id=1 / 18,
                      pathogen_acc_random_fragment_split=rand_acc, pathogen_acc_genome_grouped_split=grp_acc,
                      mean_within_pair_AUC=np.mean(list(aucs.values())), **{f'AUC_{g}': v for g, v in aucs.items()}))
    print(probe[-1], flush=True)
pd.DataFrame(probe).to_csv(f'{OUT}/task5_probes.csv', index=False)
print('TASK5 DONE', flush=True)
