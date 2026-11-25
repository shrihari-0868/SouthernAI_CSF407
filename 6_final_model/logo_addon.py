"""Supplement to run_final_model.py: genome-level leave-one-genus-out + 200-run permutation null for k-mer models
that were not in the first run's LOGO list (the CV-selected final model, 2-mer). Merges the rows into the
existing task7_genome_level_logo.csv / task7_permutation_null.csv / task7_genome_scores.csv and redraws the null figure.
(run_final_model.py now includes 2-mer itself; this file only exists so the 45-minute CNN part need not be rerun.)

Run:  python logo_addon.py ../data/dataset.csv results 2-mer
"""
import sys, os, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

CSV, OUT, MODELS = sys.argv[1], sys.argv[2], sys.argv[3:]
df = pd.read_csv(CSV)
y = df.label.to_numpy(int); gid = df.genome_id.to_numpy(str); genus = df.genus.to_numpy(str)
ginfo = df.groupby('genome_id').agg(genus=('genus', 'first'), organism=('organism', 'first'), label=('label', 'first'))
pairs = [g for g, d in ginfo.groupby('genus') if len(d) == 2 and d.label.nunique() == 2]
G_ORDER = ginfo.index.values; g_lab = ginfo.label.values; genera = np.unique(genus)
def words(k): return [''.join(p) for p in itertools.product('ACGT', repeat=k)]
def kmer_freq(S, k):
    X = CountVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False, vocabulary=words(k), dtype=np.float32).transform(S).toarray()
    return X / X.sum(1, keepdims=True)
def LR(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))

def logo(X, labels_by_genome):            # identical logic to run_final_model.py
    yy = pd.Series(labels_by_genome, index=G_ORDER).loc[gid].values
    scores = pd.Series(0.0, index=G_ORDER)
    for gn in genera:
        te = genus == gn; tr = ~te
        if len(np.unique(yy[tr])) < 2: return None
        p = LR().fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
        for g_, v in pd.Series(p).groupby(gid[te]).mean().items(): scores[g_] = v
    return scores
def summarize(scores, labels):
    lab = pd.Series(labels, index=G_ORDER)
    acc = ((scores > 0.5).astype(int) == lab).mean(); auc = roc_auc_score(lab, scores); pw = 0
    for gn in pairs:
        ids = ginfo[ginfo.genus == gn].index; p_ = ids[lab[ids] == 1]; n_ = ids[lab[ids] == 0]
        if len(p_) == 1 and len(n_) == 1: pw += int(scores[p_[0]] > scores[n_[0]])
    return acc, auc, pw
perm_labels = [np.random.default_rng(1000 + i).permutation(g_lab) for i in range(200)]   # same seeds as run_final_model.py

L = pd.read_csv(f'{OUT}/task7_genome_level_logo.csv'); N = pd.read_csv(f'{OUT}/task7_permutation_null.csv'); S = pd.read_csv(f'{OUT}/task7_genome_scores.csv')
for name in MODELS:
    X = kmer_freq(df.sequence, int(name[0]))
    s = logo(X, g_lab); acc, auc, pw = summarize(s, g_lab)
    nul = [summarize(sn, pl) for sn, pl in zip(Parallel(n_jobs=14)(delayed(logo)(X, pl) for pl in perm_labels), perm_labels) if sn is not None]
    na, nu, _ = map(np.array, zip(*nul))
    row = dict(model=name, genome_acc_of24=acc, n_genomes_correct=int(round(acc * 24)), genome_auc=auc,
               perm_p_acc=(np.sum(na >= acc) + 1) / (len(na) + 1), perm_p_auc=(np.sum(nu >= auc) + 1) / (len(nu) + 1),
               null_acc_95pct=np.quantile(na, 0.95), matched_pairs_pathogen_scored_higher_of6=pw,
               binom_p_pairs=binomtest(pw, 6, 0.5, alternative='greater').pvalue)
    print(row, flush=True)
    L = pd.concat([L[L.model != name], pd.DataFrame([row])], ignore_index=True)
    N = pd.concat([N[N.model != name], pd.DataFrame([dict(model=name, genome_acc=a, genome_auc=u, pairs_won=w) for a, u, w in nul])], ignore_index=True)
    S = pd.concat([S[S.model != name], pd.DataFrame([dict(model=name, genome_id=g, organism=ginfo.organism[g], genus=ginfo.genus[g],
                                                          label=int(ginfo.label[g]), logo_mean_p_pathogen=s[g]) for g in G_ORDER])], ignore_index=True)
order = ['GC-only', 'Letters (1-mer)', '2-mer', '3-mer', '4-mer', 'Multi-scale 1-4-mer', 'CNN (raw sequence)']
key = lambda c: c.map({m: i for i, m in enumerate(order)})
L = L.sort_values('model', key=key); N = N.sort_values('model', key=key, kind='stable'); S = S.sort_values('model', key=key, kind='stable')
L.to_csv(f'{OUT}/task7_genome_level_logo.csv', index=False); N.to_csv(f'{OUT}/task7_permutation_null.csv', index=False); S.to_csv(f'{OUT}/task7_genome_scores.csv', index=False)

mods = [m for m in order if m in set(N.model)]
fig, axs = plt.subplots(1, len(mods), figsize=(3 * len(mods), 3), sharey=True)
for a, m in zip(axs, mods):
    a.hist(N[N.model == m].genome_acc * 24, bins=np.arange(-0.5, 25), color='0.7')
    a.axvline(L[L.model == m].n_genomes_correct.iloc[0], color='tab:red', lw=2); a.set_title(m, fontsize=9); a.set_xlabel('genomes correct of 24')
axs[0].set_ylabel('permutation runs'); fig.suptitle('Observed (red) vs 200 random genome labellings', fontsize=10)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_permutation_null.png', dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5)); mods = [m for m in order if m in set(S.model)]
for i, m in enumerate(mods):
    d = S[S.model == m]; jit = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.25
    ax.scatter(i + jit, d.logo_mean_p_pathogen, c=np.where(d.label == 1, 'tab:red', 'tab:blue'), s=22, alpha=0.8)
ax.axhline(0.5, color='k', lw=0.8, ls='--'); ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('mean P(pathogen) for a held-out genome'); ax.set_title('Leave-one-genus-out, every genome (red = pathogen, blue = non-pathogen)')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_genome_scores_logo.png', dpi=130); plt.close()
print('ADDON DONE', flush=True)
