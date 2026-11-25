"""Task 7 - final challenge: can the model "read" DNA?

Run:  python -I run_final_model.py ../data/dataset.csv OUT_DIR

Steps (all hyper-parameters fixed in advance, nothing tuned on the 6 test genomes):
  A. Candidate final models, compared by genus-grouped cross-validation on the 18 TRAINING genomes only.
       letters (1-mers) / 2-mers / 3-mers / 4-mers / multi-scale 1..4-mers with logistic regression,
       GC-only and majority baselines, and a 1-D CNN that reads the raw one-hot sequence
       (learns its own "words" as convolution filters instead of being given k-mers).
  B. The CV winner and every other candidate are scored once on the 6 never-seen test genomes
     (fragment accuracy, AUC, per-genome accuracy, genome-level call by averaging fragment scores).
  C. "Grammar" tests on the test genomes: re-score the same fragments after
       (i) shuffling letters inside each fragment (same composition, no order),
       (ii) shuffling 10-bp blocks (keeps short words, destroys longer order),
       (iii) reverse-complementing (same DNA, other strand).
     A model that reads order should lose accuracy under (i)/(ii); a model that only counts letters should not.
  D. Genome-level leave-one-genus-out over ALL 24 genomes (a genome and its genus partner are always
     held out together) + a 200-run label-permutation null. This answers: "given a genome it has never
     seen, from a genus it has never seen, does the model call pathogen vs non-pathogen better than luck?"
     Also: inside each of the 6 matched pairs, does the pathogen get the higher score?
"""
import sys, os, itertools, time, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SEED = 0
rng = np.random.default_rng(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

df = pd.read_csv(CSV)
assert (df.length == 1000).all()
seqs = df.sequence.values
y = df.label.values.astype(int)
gid = df.genome_id.values
genus = df.genus.values
is_tr = (df.split_suggested == 'train').values
is_te = ~is_tr
ginfo = df.groupby('genome_id').agg(genus=('genus', 'first'), organism=('organism', 'first'), label=('label', 'first'))
pairs = [g for g, d in ginfo.groupby('genus') if len(d) == 2 and d.label.nunique() == 2]
log('fragments', len(df), 'train genomes', df[is_tr].genome_id.nunique(), 'test genomes', df[is_te].genome_id.nunique())

# ------------------------------------------------------------------ features
def words(k): return [''.join(p) for p in itertools.product('ACGT', repeat=k)]
def kmer_freq(S, k):
    cv = CountVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False, vocabulary=words(k), dtype=np.float32)
    X = cv.transform(S).toarray()
    return X / X.sum(1, keepdims=True)
def feats(S, name):
    if name == 'gc': return np.array([[(s.count('G') + s.count('C')) / len(s)] for s in S], np.float32)
    if name == 'multi1-4': return np.hstack([kmer_freq(S, k) for k in (1, 2, 3, 4)])
    return kmer_freq(S, int(name[0]))

LR_FEATS = {'GC-only': 'gc', 'Letters (1-mer)': '1', '2-mer': '2', '3-mer': '3', '4-mer': '4', 'Multi-scale 1-4-mer': 'multi1-4'}
X_all = {n: feats(seqs, f) for n, f in LR_FEATS.items()}
log('features built', {n: X.shape[1] for n, X in X_all.items()})

def LR(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))

# ------------------------------------------------------------------ CNN that reads raw sequence
LUT = np.zeros(256, np.int64); LUT[[ord(c) for c in 'ACGT']] = [0, 1, 2, 3]
def onehot(S):
    idx = np.stack([LUT[np.frombuffer(s.encode(), np.uint8)] for s in S])
    X = np.zeros((len(S), 4, idx.shape[1]), np.float32)
    np.put_along_axis(X, idx[:, None, :], 1.0, axis=1)
    return torch.from_numpy(X)
def revcomp_t(X): return torch.flip(X, dims=[1, 2])           # A<->T, C<->G (channel order ACGT) and reverse

class SeqCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(4, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
                               nn.Conv1d(64, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU())
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(s, x):
        h = s.conv(x)
        return s.head(torch.cat([h.amax(-1), h.mean(-1)], 1)).squeeze(1)

class CNNModel:
    """sklearn-like wrapper. Fixed recipe: 6 epochs, Adam 1e-3, wd 1e-4, batch 128, random reverse-complement
    augmentation, class-balanced loss, prediction = mean over both strands."""
    def __init__(s, epochs=6, seed=SEED): s.epochs, s.seed = epochs, seed
    def fit(s, X, yy):
        torch.manual_seed(s.seed); g = np.random.default_rng(s.seed)
        s.m = SeqCNN(); opt = torch.optim.Adam(s.m.parameters(), 1e-3, weight_decay=1e-4)
        yt = torch.tensor(yy, dtype=torch.float32)
        pw = torch.tensor((yy == 0).sum() / max(1, (yy == 1).sum()), dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        for ep in range(s.epochs):
            s.m.train()
            for b in np.array_split(g.permutation(len(yy)), max(1, len(yy) // 128)):
                xb = X[b]
                flip = torch.from_numpy(g.random(len(b)) < 0.5)
                xb = torch.where(flip[:, None, None], revcomp_t(xb), xb)
                opt.zero_grad(); lossf(s.m(xb), yt[b]).backward(); opt.step()
        return s
    @torch.no_grad()
    def predict_proba(s, X):
        s.m.eval(); out = []
        for b in torch.split(torch.arange(len(X)), 512):
            xb = X[b]; out.append(0.5 * (torch.sigmoid(s.m(xb)) + torch.sigmoid(s.m(revcomp_t(xb)))))
        p = torch.cat(out).numpy(); return np.c_[1 - p, p]

OH = onehot(seqs)
log('one-hot built', tuple(OH.shape))

def fit_predict(name, tr, te, X_override=None):
    """train candidate `name` on rows tr, return P(pathogen) for rows te (or for X_override if given)."""
    if name == 'Majority': return np.full(len(te) if X_override is None else len(X_override), 0.5)
    if name == 'CNN (raw sequence)':
        m = CNNModel().fit(OH[tr], y[tr]); return m.predict_proba(OH[te] if X_override is None else X_override)[:, 1]
    X = X_all[name]; m = LR().fit(X[tr], y[tr])
    return m.predict_proba(X[te] if X_override is None else X_override)[:, 1]

CANDS = ['Majority', 'GC-only', 'Letters (1-mer)', '2-mer', '3-mer', '4-mer', 'Multi-scale 1-4-mer', 'CNN (raw sequence)']

# ------------------------------------------------------------------ A. model selection on training genomes only
tr_idx = np.where(is_tr)[0]; te_idx = np.where(is_te)[0]
gkf = list(GroupKFold(7).split(tr_idx, y[tr_idx], genus[tr_idx]))     # 14 training genera -> 7 folds of 2 genera
def genome_acc(p, rows):
    g = pd.DataFrame(dict(g=gid[rows], p=p, y=y[rows])).groupby('g').mean()
    return ((g.p > 0.5).astype(int) == g.y).mean()
cv_rows = []
for name in CANDS:
    P = np.zeros(len(tr_idx))
    for a, b in gkf: P[b] = fit_predict(name, tr_idx[a], tr_idx[b])
    yy = y[tr_idx]
    cv_rows.append(dict(model=name, cv_fragment_acc=((P > 0.5) == yy).mean(),
                        cv_auc=roc_auc_score(yy, P) if name != 'Majority' else 0.5,
                        cv_genome_acc=genome_acc(P, tr_idx)))
    log('CV', cv_rows[-1])
cv = pd.DataFrame(cv_rows)
real = cv[~cv.model.isin(['Majority'])]
FINAL = real.sort_values(['cv_fragment_acc', 'cv_auc'], ascending=False).model.iloc[0]
log('FINAL MODEL chosen by training-genome CV:', FINAL)

# ------------------------------------------------------------------ B. one-shot evaluation on the 6 unseen genomes
test_rows, per_genome, P_TEST = [], [], {}
for name in CANDS:
    p = fit_predict(name, tr_idx, te_idx); P_TEST[name] = p
    yy = y[te_idx]
    g = pd.DataFrame(dict(g=gid[te_idx], p=p, y=yy, c=((p > 0.5) == yy))).groupby('g').mean()
    test_rows.append(dict(model=name, chosen_as_final=name == FINAL, test_fragment_acc=((p > 0.5) == yy).mean(),
                          test_auc=roc_auc_score(yy, p) if name != 'Majority' else 0.5,
                          test_genomes_called_correctly_of6=int(((g.p > 0.5).astype(int) == g.y).sum()),
                          corr_with_fragment_GC=np.corrcoef(p, X_all['GC-only'][te_idx, 0])[0, 1] if name != 'Majority' else np.nan))
    for gg, r in g.iterrows():
        per_genome.append(dict(model=name, genome_id=gg, organism=ginfo.organism[gg], label=int(r.y),
                               fragment_acc=r.c, mean_p_pathogen=r.p, genome_call=int(r.p > 0.5)))
    log('TEST', test_rows[-1])
res = cv.merge(pd.DataFrame(test_rows), on='model')
res.to_csv(f'{OUT}/task7_model_comparison.csv', index=False)
pd.DataFrame(per_genome).to_csv(f'{OUT}/task7_test_per_genome.csv', index=False)

# ------------------------------------------------------------------ C. grammar / shortcut tests
def shuffle_letters(s, r): a = np.frombuffer(s.encode(), np.uint8).copy(); r.shuffle(a); return a.tobytes().decode()
def shuffle_blocks(s, r, b=10): bl = [s[i:i + b] for i in range(0, len(s), b)]; r.shuffle(bl); return ''.join(bl)
COMP = str.maketrans('ACGT', 'TGCA')
def revcomp(s): return s.translate(COMP)[::-1]
r = np.random.default_rng(1)
TEST_SEQS = {'original': seqs[te_idx],
             'letters shuffled (composition kept, order destroyed)': np.array([shuffle_letters(s, r) for s in seqs[te_idx]]),
             '10-bp blocks shuffled (short words kept)': np.array([shuffle_blocks(s, r) for s in seqs[te_idx]]),
             'reverse complement (same DNA, other strand)': np.array([revcomp(s) for s in seqs[te_idx]])}
grammar = []
fitted = {}
for name in ['GC-only', 'Letters (1-mer)', '3-mer', '4-mer', 'Multi-scale 1-4-mer', 'CNN (raw sequence)']:
    if name == 'CNN (raw sequence)': fitted[name] = CNNModel().fit(OH[tr_idx], y[tr_idx])
    else: fitted[name] = LR().fit(X_all[name][tr_idx], y[tr_idx])
    base_p = None
    for cond, S in TEST_SEQS.items():
        Xc = onehot(S) if name == 'CNN (raw sequence)' else feats(S, LR_FEATS[name])
        p = fitted[name].predict_proba(Xc)[:, 1]
        if base_p is None: base_p = p
        grammar.append(dict(model=name, condition=cond, test_fragment_acc=((p > 0.5) == y[te_idx]).mean(),
                            test_auc=roc_auc_score(y[te_idx], p),
                            corr_with_original_scores=np.corrcoef(p, base_p)[0, 1],
                            mean_abs_change_in_p=np.abs(p - base_p).mean()))
        log('GRAMMAR', grammar[-1])
pd.DataFrame(grammar).to_csv(f'{OUT}/task7_grammar_tests.csv', index=False)

# ------------------------------------------------------------------ D. genome-level leave-one-genus-out on all 24 genomes + permutation null
genera = np.unique(genus)
G_ORDER = ginfo.index.values
g_lab = ginfo.label.values
def logo_genome_scores(name, labels_by_genome):
    yy = pd.Series(labels_by_genome, index=G_ORDER).loc[gid].values
    scores = pd.Series(0.0, index=G_ORDER)
    X = X_all[name]
    for gn in genera:
        te = genus == gn; tr = ~te
        if len(np.unique(yy[tr])) < 2: return None
        m = LR().fit(X[tr], yy[tr]); p = m.predict_proba(X[te])[:, 1]
        for g_, v in pd.Series(p).groupby(gid[te]).mean().items(): scores[g_] = v
    return scores
def summarize(scores, labels):
    lab = pd.Series(labels, index=G_ORDER)
    acc = ((scores > 0.5).astype(int) == lab).mean()
    auc = roc_auc_score(lab, scores)
    pw = 0
    for gn in pairs:
        ids = ginfo[ginfo.genus == gn].index; p_ = ids[lab[ids] == 1]; n_ = ids[lab[ids] == 0]
        if len(p_) == 1 and len(n_) == 1: pw += int(scores[p_[0]] > scores[n_[0]])
    return acc, auc, pw

LOGO_MODELS = [m for m in ['GC-only', 'Letters (1-mer)', '2-mer', '3-mer', '4-mer', 'Multi-scale 1-4-mer'] if m in X_all]
N_PERM = 200
perm_labels = []
for i in range(N_PERM):
    perm_labels.append(np.random.default_rng(1000 + i).permutation(g_lab))
logo_rows, null_rows, genome_scores = [], [], []
for name in LOGO_MODELS:
    s = logo_genome_scores(name, g_lab); acc, auc, pw = summarize(s, g_lab)
    for g_ in G_ORDER: genome_scores.append(dict(model=name, genome_id=g_, organism=ginfo.organism[g_], genus=ginfo.genus[g_],
                                                 label=int(ginfo.label[g_]), logo_mean_p_pathogen=s[g_]))
    null = Parallel(n_jobs=min(14, os.cpu_count() or 1))(delayed(logo_genome_scores)(name, pl) for pl in perm_labels)
    nul = [summarize(sn, pl) for sn, pl in zip(null, perm_labels) if sn is not None]
    na, nu, npw = map(np.array, zip(*nul))
    for a_, u_, w_ in nul: null_rows.append(dict(model=name, genome_acc=a_, genome_auc=u_, pairs_won=w_))
    logo_rows.append(dict(model=name, genome_acc_of24=acc, n_genomes_correct=int(round(acc * 24)), genome_auc=auc,
                          perm_p_acc=(np.sum(na >= acc) + 1) / (len(na) + 1), perm_p_auc=(np.sum(nu >= auc) + 1) / (len(nu) + 1),
                          null_acc_95pct=np.quantile(na, 0.95), matched_pairs_pathogen_scored_higher_of6=pw,
                          binom_p_pairs=binomtest(pw, 6, 0.5, alternative='greater').pvalue))
    log('LOGO', logo_rows[-1])

# CNN leave-one-genus-out (no permutation null: 200 x 18 CNN fits is too slow on CPU)
s = pd.Series(0.0, index=G_ORDER)
for gn in genera:
    te = genus == gn
    p = CNNModel().fit(OH[~te], y[~te]).predict_proba(OH[te])[:, 1]
    for g_, v in pd.Series(p).groupby(gid[te]).mean().items(): s[g_] = v
acc, auc, pw = summarize(s, g_lab)
for g_ in G_ORDER: genome_scores.append(dict(model='CNN (raw sequence)', genome_id=g_, organism=ginfo.organism[g_], genus=ginfo.genus[g_],
                                             label=int(ginfo.label[g_]), logo_mean_p_pathogen=s[g_]))
gc_null = pd.DataFrame([r_ for r_ in null_rows if r_['model'] == 'GC-only'])
logo_rows.append(dict(model='CNN (raw sequence)', genome_acc_of24=acc, n_genomes_correct=int(round(acc * 24)), genome_auc=auc,
                      perm_p_acc=(np.sum(gc_null.genome_acc >= acc) + 1) / (len(gc_null) + 1),
                      perm_p_auc=(np.sum(gc_null.genome_auc >= auc) + 1) / (len(gc_null) + 1),
                      null_acc_95pct=gc_null.genome_acc.quantile(0.95), matched_pairs_pathogen_scored_higher_of6=pw,
                      binom_p_pairs=binomtest(pw, 6, 0.5, alternative='greater').pvalue))
log('LOGO', logo_rows[-1], '(p-values use the GC-only permutation null as an approximation)')
pd.DataFrame(logo_rows).to_csv(f'{OUT}/task7_genome_level_logo.csv', index=False)
pd.DataFrame(null_rows).to_csv(f'{OUT}/task7_permutation_null.csv', index=False)
pd.DataFrame(genome_scores).to_csv(f'{OUT}/task7_genome_scores.csv', index=False)

# ------------------------------------------------------------------ CNN filters: which "words" did the network invent?
cnn = fitted['CNN (raw sequence)'].m
W = cnn.conv[0].weight.detach().numpy()                         # 64 filters x 4 letters x 15 positions
motif_rows = []
for fi, f in enumerate(W):
    best = ''.join('ACGT'[j] for j in f.argmax(0))
    # activation of each filter on train fragments, compared between classes and against fragment GC
    with torch.no_grad():
        act = torch.relu(cnn.conv[1](cnn.conv[0](OH[tr_idx[::4]])))[:, fi].amax(-1).numpy()
    yy = y[tr_idx[::4]]; gcv = X_all['GC-only'][tr_idx[::4], 0]
    motif_rows.append(dict(filter=fi, consensus=best, consensus_gc=sum(c in 'GC' for c in best) / 15,
                           mean_act_pathogen=act[yy == 1].mean(), mean_act_nonpathogen=act[yy == 0].mean(),
                           corr_act_with_fragment_gc=np.corrcoef(act, gcv)[0, 1],
                           auc_act_for_label=roc_auc_score(yy, act)))
pd.DataFrame(motif_rows).sort_values('auc_act_for_label', ascending=False).to_csv(f'{OUT}/task7_cnn_filters.csv', index=False)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(res)); w = 0.38
ax.bar(x - w / 2, res.cv_fragment_acc, w, label='CV on 18 training genomes (grouped by genus)')
ax.bar(x + w / 2, res.test_fragment_acc, w, label='6 unseen test genomes')
ax.axhline(0.5, color='k', lw=0.8, ls='--'); ax.set_xticks(x); ax.set_xticklabels(res.model, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('fragment accuracy'); ax.set_ylim(0.3, 0.8); ax.legend(fontsize=8); ax.set_title('Task 7: candidate final models')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_model_comparison.png', dpi=130); plt.close()

gs = pd.DataFrame(genome_scores)
fig, ax = plt.subplots(figsize=(9, 4.5))
mods = gs.model.unique()
for i, mname in enumerate(mods):
    d = gs[gs.model == mname]
    jit = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.25
    ax.scatter(i + jit, d.logo_mean_p_pathogen, c=np.where(d.label == 1, 'tab:red', 'tab:blue'), s=22, alpha=0.8)
ax.axhline(0.5, color='k', lw=0.8, ls='--'); ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('mean P(pathogen) for a held-out genome'); ax.set_title('Leave-one-genus-out, every genome (red = pathogen, blue = non-pathogen)')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_genome_scores_logo.png', dpi=130); plt.close()

nd = pd.DataFrame(null_rows); lr = pd.DataFrame(logo_rows)
fig, axs = plt.subplots(1, len(LOGO_MODELS), figsize=(3 * len(LOGO_MODELS), 3), sharey=True)
for a, mname in zip(axs, LOGO_MODELS):
    a.hist(nd[nd.model == mname].genome_acc * 24, bins=np.arange(-0.5, 25), color='0.7')
    a.axvline(lr[lr.model == mname].n_genomes_correct.iloc[0], color='tab:red', lw=2)
    a.set_title(mname, fontsize=9); a.set_xlabel('genomes correct of 24')
axs[0].set_ylabel('permutation runs'); fig.suptitle('Observed (red) vs 200 random genome labellings', fontsize=10)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_permutation_null.png', dpi=130); plt.close()

json.dump(dict(final_model=FINAL, runtime_s=time.time() - T0), open(f'{OUT}/task7_final_model.json', 'w'), indent=1)
log('TASK7 DONE, final =', FINAL)
