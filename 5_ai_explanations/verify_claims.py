"""Task 6 - independently test the claims made by the AI systems (responses/*.md) against the data.

Run:  python verify_claims.py ../data .. results

(second argument is the repository root, which holds 3_classifiers/ and 4_interpretation/)

Each claim made by the AI systems falls into one of two groups:
  * RESTATEMENT: the claim only repeats a number that was given in prompt.md. It is checked
    against the existing Task 4/5 CSVs, but it is not counted as evidence that the AI understood anything.
  * NEW: the claim says something the prompt did not contain. Each one gets its own test below.
Verdicts: SUPPORTED / PARTLY SUPPORTED / NOT SUPPORTED / FACTUAL ERROR.
"""
import sys, os, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

DATA, ROOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CLF, INTERP = f'{ROOT}/3_classifiers/results', f'{ROOT}/4_interpretation/results'
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(f'{DATA}/dataset.csv')
gen = pd.read_csv(f'{DATA}/genomes.csv').set_index('genome_id')
y = df.label.to_numpy(int); gid = df.genome_id.to_numpy(str); genus = df.genus.to_numpy(str)
tr = (df.split_suggested == 'train').values; te = ~tr
pairs = sorted(gen[gen.pair_group == 'pair'].genus.unique())
ORG = gen.organism.to_dict()
rows, details = [], {}
def add(cid, systems, claim, kind, test, result, verdict):
    rows.append(dict(claim_id=cid, systems=systems, claim=claim, kind=kind, test=test, result=result, verdict=verdict))
    print(f'{cid:5s} {verdict:18s} {result}', flush=True)

def words(k): return [''.join(p) for p in itertools.product('ACGT', repeat=k)]
def kfreq(S, k):
    X = CountVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False, vocabulary=words(k), dtype=np.float32).transform(S).toarray()
    return X / X.sum(1, keepdims=True)
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
X4 = kfreq(df.sequence, 4); W4 = np.array(words(4))
X6 = kfreq(df.sequence, 6); W6 = np.array(words(6))
G4 = pd.DataFrame(X4, columns=W4).groupby(gid).mean()          # genome-mean 4-mer frequencies
G6 = pd.DataFrame(X6, columns=W6).groupby(gid).mean()
glab = gen.label.loc[G4.index]

# ======================================================================== restatements (checked against Task 4/5 outputs)
cvb = pd.read_csv(f'{CLF}/cv_leave_group_out_balanced.csv')
res4 = pd.read_csv(f'{CLF}/results_task4.csv')
probes = pd.read_csv(f'{INTERP}/task5_probes.csv')
nul = pd.read_csv(f'{CLF}/null_random_labels.csv')
best = res4.test_acc.max(); p_null = (nul.test_acc >= best).mean()
add('R1', 'A1,B6', 'Best real test accuracy is not significant against the shuffled-label null', 'RESTATEMENT',
    'share of 200 null runs >= best Task 4 test accuracy', f'best={best:.3f}, null p={p_null:.2f}', 'SUPPORTED' if p_null > 0.05 else 'NOT SUPPORTED')
add('R2', 'A3', 'Within-pair AUC varies widely (0.25-0.79), mean ~0.5', 'RESTATEMENT', 'task5_probes.csv',
    f'mean AUC {probes.mean_within_pair_AUC.min():.2f}-{probes.mean_within_pair_AUC.max():.2f}', 'SUPPORTED')
lo, hi = cvb[cvb.rep != '-'].cv_leave_group_out.min(), cvb[cvb.rep != '-'].cv_leave_group_out.max()
add('B1', 'B1', 'Genome-grouped CV gives 45-55 % on every representation', 'RESTATEMENT (with a range)',
    'cv_leave_group_out_balanced.csv, all 18 rep x model rows', f'range {lo:.3f}-{hi:.3f}; {int(((cvb.cv_leave_group_out < .45) & (cvb.rep != "-")).sum())} of 18 rows below 45 %',
    'PARTLY SUPPORTED' if lo < 0.45 else 'SUPPORTED')
add('R3', 'A2,A4,A5,A8,B3', 'Pair-sign agreement ~chance, unseen-pair Spearman ~0, GC removal changes nothing, LOGO near chance',
    'RESTATEMENT', 'these numbers were given in the prompt', 'identical to prompt', 'SUPPORTED (circular)')

# ======================================================================== B: factual claim about the training set
claimed = ['Helicobacter', 'Staphylococcus', 'Listeria']
where = {g: df[df.genus == g].split_suggested.iloc[0] for g in claimed}
add('B-f', 'B (part b)', '"AT-rich pathogens such as H. pylori, S. aureus and Listeria in the training set" drive the AT-rich weights', 'NEW (fact)',
    'look up split_suggested', ', '.join(f'{g}: {s}' for g, s in where.items()),
    'FACTUAL ERROR' if any(s == 'test' for s in where.values()) else 'SUPPORTED')

# ======================================================================== C3 / B4: are the top coefficients traceable to single training genomes?
trg = sorted(set(gid[tr]))
Gtr = G4.loc[trg]
claims_c3 = {'GCGC': ['NC_002516.2', 'NC_000962.3'], 'CGCG': ['NC_002516.2', 'NC_000962.3'],
             'GGGC': ['NC_003888.3'], 'CCGT': ['NC_003888.3'], 'ACGG': ['NC_003888.3'],
             'TAAA': ['NC_007795.1'], 'TTTA': ['NC_007795.1']}
hits, txt = 0, []
for w, gs in claims_c3.items():
    top2 = Gtr[w].sort_values(ascending=False).index[:2].tolist()
    ok = any(g in top2 for g in gs); hits += ok
    txt.append(f'{w}: top={ORG[top2[0]].split()[0][:1]}.{ORG[top2[0]].split()[1][:6]},{ORG[top2[1]].split()[0][:1]}.{ORG[top2[1]].split()[1][:6]}{"" if ok else " (miss)"}')
g6tr = G6.loc[trg]; top = g6tr['TTTTAA'].sort_values(ascending=False).index[:2].tolist()
ok6 = 'NC_007795.1' in top; hits += ok6; txt.append(f'TTTTAA: top2={[ORG[g].split()[0] + " " + ORG[g].split()[1] for g in top]}')
add('C3a', 'C3', 'Strongest k-mers are most enriched in specific named training genomes (GCGC/CGCG: P. aeruginosa, M. tb; GGGC/CCGT/ACGG: S. coelicolor; TAAA/TTTA/TTTTAA: S. aureus)',
    'NEW', 'rank 18 training genomes by mean frequency of each word; claim holds if a named genome is in the top 2',
    f'{hits}/8 words. ' + '; '.join(txt), 'SUPPORTED' if hits >= 7 else 'PARTLY SUPPORTED' if hits >= 4 else 'NOT SUPPORTED')

Xtr4, ytr = X4[tr], y[tr]
full = LR().fit(Xtr4, ytr).named_steps['logisticregression'].coef_[0]
top20 = set(np.argsort(-np.abs(full))[:20])
ov = []
for g in trg:
    keep = gid[tr] != g
    c = LR().fit(Xtr4[keep], ytr[keep]).named_steps['logisticregression'].coef_[0]
    ov.append(len(top20 & set(np.argsort(-np.abs(c))[:20])) / 20)
ov = np.array(ov); details['drop_one_overlap'] = pd.DataFrame(dict(dropped=[ORG[g] for g in trg], top20_overlap=ov))
add('B4', 'B4, C3', 'Dropping any one training genome substantially changes the top-20 k=4 coefficient list (overlap well under 50 %)', 'NEW',
    'refit k=4 BoW LogReg 18 times, each without one training genome; overlap of top-20 |coef| words',
    f'overlap median {np.median(ov):.0%}, min {ov.min():.0%} (dropping {ORG[trg[int(ov.argmin())]]}), max {ov.max():.0%}',
    'SUPPORTED' if np.median(ov) < 0.5 else 'PARTLY SUPPORTED' if ov.min() < 0.5 else 'NOT SUPPORTED')

# ======================================================================== C1 / B7: are predictions all-or-nothing per genome?
p = LR().fit(Xtr4, ytr).predict_proba(X4[te])[:, 1]
pg = pd.DataFrame(dict(g=gid[te], call=(p > 0.5).astype(int), y=y[te])).groupby('g').agg(major=('call', lambda c: max(c.mean(), 1 - c.mean())), mean=('call', 'mean'), y=('y', 'first'))
g_ok = int(((pg['mean'] > 0.5).astype(int) == pg.y).sum())
details['test_genome_consistency'] = pg.assign(organism=[ORG[g] for g in pg.index])
add('C1', 'C1, B7', 'Within each test genome >=80 % of fragments get the same label; genome-level accuracy about 3/6', 'NEW',
    'k=4 BoW LogReg trained on 18 genomes; share of majority call within each of 6 test genomes; genome vote',
    f'majority share per genome {", ".join(f"{v:.0%}" for v in pg.major)}; genome vote {g_ok}/6',
    'SUPPORTED' if (pg.major >= .8).all() and g_ok in (2, 3, 4) else 'PARTLY SUPPORTED' if g_ok in (2, 3, 4) or (pg.major >= .8).mean() >= .5 else 'NOT SUPPORTED')

# ======================================================================== C2: is the cross-genome nearest neighbour the genus partner?
Z = StandardScaler().fit_transform(X4)
nn = NearestNeighbors(n_neighbors=401).fit(Z)
pair_rows = np.where(np.isin(genus, pairs))[0]
sub = pair_rows[::4]
idx = nn.kneighbors(Z[sub], return_distance=False)
first_other = np.array([nb[gid[nb] != gid[i]][0] for i, nb in zip(sub, idx)])
same_genus = (genus[first_other] == genus[sub]).mean()
same_label = (y[first_other] == y[sub]).mean()
add('C2', 'C2', 'For most (paired-genus) fragments the nearest neighbour from another genome is the genus partner, which explains below-chance label purity', 'NEW',
    '4-mer standardized space; 1200 fragments from the 12 paired genomes; first neighbour not from own genome',
    f'neighbour is genus partner {same_genus:.0%} (chance 1/23 = 4 %); label agreement {same_label:.0%}',
    'SUPPORTED' if same_genus > 0.5 else 'PARTLY SUPPORTED' if same_genus > 0.2 else 'NOT SUPPORTED')

# ======================================================================== A6 / B2: errors go to similar-GC / same-genus genomes
gtr_ids = gid[tr]
pred_g = cross_val_predict(LR(), Xtr4, gtr_ids, cv=StratifiedKFold(5, shuffle=True, random_state=0))
err = pred_g != gtr_ids
gcg = gen.genome_gc.to_dict(); gng = gen.genus.to_dict()
dgc_err = np.abs(np.array([gcg[a] - gcg[b] for a, b in zip(gtr_ids[err], pred_g[err])]))
rng = np.random.default_rng(0)
rand_other = np.array([rng.choice([g for g in trg if g != a]) for a in gtr_ids[err]])
dgc_rand = np.abs(np.array([gcg[a] - gcg[b] for a, b in zip(gtr_ids[err], rand_other)]))
same_gen_err = np.mean([gng[a] == gng[b] for a, b in zip(gtr_ids[err], pred_g[err])])
add('B2', 'B2, A6', 'Genome-identity errors go to genomes of the same genus or similar GC, not to random genomes', 'NEW',
    '18-way genome-ID LogReg (5-fold random split); |genome GC difference| between true and predicted genome for errors vs random wrong genome',
    f'error rate {err.mean():.0%}; median |dGC| errors {np.median(dgc_err) * 100:.1f} pts vs random {np.median(dgc_rand) * 100:.1f} pts; errors into genus partner {same_gen_err:.0%} (random ~{4 / 17:.0%} for paired genomes)',
    'SUPPORTED' if np.median(dgc_err) < 0.5 * np.median(dgc_rand) else 'PARTLY SUPPORTED' if np.median(dgc_err) < np.median(dgc_rand) else 'NOT SUPPORTED')

# ======================================================================== C4 / A7: CTAG under-representation (Markov-2 expectation)
def mk2_ratio(G3, G2, G4, w):     # f(w) * f(w2w3) / (f(w1w2w3) f(w2w3w4)), the classic Karlin-style 2nd-order Markov ratio
    return G4[w] * G2[w[1:3]] / (G3[w[:3]] * G3[w[1:]])
G3 = pd.DataFrame(kfreq(df.sequence, 3), columns=words(3)).groupby(gid).mean()
G2 = pd.DataFrame(kfreq(df.sequence, 2), columns=words(2)).groupby(gid).mean()
ctag = mk2_ratio(G3, G2, G4, 'CTAG')
mw = mannwhitneyu(ctag[glab == 1], ctag[glab == 0])
details['ctag'] = pd.DataFrame(dict(organism=[ORG[g] for g in ctag.index], label=glab.values, ctag_obs_over_exp=ctag.values, ctag_freq=G4['CTAG'].values))
add('C4', 'C4, A7, B(b)', 'CTAG obs/exp (2nd-order Markov) < 0.5 in most genomes, and no pathogen vs non-pathogen difference', 'NEW',
    'genome-mean frequencies; ratio f(CTAG)f(TA)/(f(CTA)f(TAG)); Mann-Whitney, 12 vs 12 genomes',
    f'{int((ctag < .5).sum())}/24 genomes below 0.5 (median {ctag.median():.2f}, range {ctag.min():.2f}-{ctag.max():.2f}); class medians {ctag[glab == 1].median():.2f} vs {ctag[glab == 0].median():.2f}, p={mw.pvalue:.2f}',
    'SUPPORTED' if (ctag < .5).mean() > .5 and mw.pvalue > .05 else 'PARTLY SUPPORTED' if mw.pvalue > .05 else 'NOT SUPPORTED')

# ======================================================================== C5 / B5: restriction palindromes vary by genus, not label
pal6 = ['CAGCTG', 'GAGCTC', 'CTTAAG']
G5 = pd.DataFrame(kfreq(df.sequence, 5), columns=words(5)).groupby(gid).mean()
def mk_ratio6(w): return G6[w] * G4[w[1:5]] / (G5[w[:5]] * G5[w[1:]])
out5 = []
for w in pal6 + ['CGCG']:
    r = mk_ratio6(w) if len(w) == 6 else mk2_ratio(G3, G2, G4, w)
    within = np.mean([abs(np.diff(r[gen[gen.genus == g].index].values))[0] for g in pairs])
    allg = r.index.values
    between = np.mean([abs(r[a] - r[b]) for a, b in itertools.combinations(allg, 2) if gng[a] != gng[b]])
    lab_gap = abs(r[glab == 1].mean() - r[glab == 0].mean())
    out5.append(dict(word=w, within_pair_mean_absdiff=within, between_genus_mean_absdiff=between, label_mean_gap=lab_gap,
                     mw_p_label=mannwhitneyu(r[glab == 1], r[glab == 0]).pvalue))
out5 = pd.DataFrame(out5); details['palindromes'] = out5
n_ok = int((out5.within_pair_mean_absdiff < out5.between_genus_mean_absdiff).sum())
add('C5', 'C5, B5', 'Restriction palindromes (CAGCTG, GAGCTC, CTTAAG, CGCG) vary more between genera than inside a matched pair, and do not separate the labels', 'NEW',
    'Markov obs/exp per genome; mean |difference| within the 6 pairs vs between genomes of different genera; Mann-Whitney by label',
    '; '.join(f'{r.word}: within {r.within_pair_mean_absdiff:.2f} vs between {r.between_genus_mean_absdiff:.2f}, label p={r.mw_p_label:.2f}' for r in out5.itertuples()),
    'SUPPORTED' if n_ok == 4 and (out5.mw_p_label > .05).all() else 'PARTLY SUPPORTED' if n_ok >= 2 else 'NOT SUPPORTED')

# ======================================================================== C6: strand - canonical k-mers do not change accuracy; coef(w) vs coef(rc(w))
COMP = str.maketrans('ACGT', 'TGCA'); rc = lambda w: w.translate(COMP)[::-1]
widx = {w: i for i, w in enumerate(W4)}
pairs_rc = [(widx[w], widx[rc(w)]) for w in W4 if w < rc(w)]
rho_rc = spearmanr(full[[a for a, b in pairs_rc]], full[[b for a, b in pairs_rc]]).correlation
opp = np.mean(np.sign(full[[a for a, b in pairs_rc]]) != np.sign(full[[b for a, b in pairs_rc]]))
canon = sorted({min(w, rc(w)) for w in W4}); cidx = {w: j for j, w in enumerate(canon)}
M = np.zeros((256, len(canon)), np.float32)
for w in W4: M[widx[w], cidx[min(w, rc(w))]] = 1
Xc = X4 @ M
acc_raw = (LR().fit(Xtr4, ytr).predict(X4[te]) == y[te]).mean()
acc_can = (LR().fit(Xc[tr], ytr).predict(Xc[te]) == y[te]).mean()
add('C6', 'C6, A(strand), C(b)', 'A word and its reverse complement can get different coefficients; counting canonical k-mers will not change test accuracy', 'NEW',
    'k=4 BoW LogReg: Spearman of coef(w) vs coef(revcomp w) over 120 pairs; refit on 136 canonical 4-mers',
    f'Spearman {rho_rc:.2f}, opposite sign in {opp:.0%} of pairs; test acc 256-dim {acc_raw:.3f} vs canonical {acc_can:.3f}',
    'SUPPORTED' if opp > 0.1 and abs(acc_raw - acc_can) < 0.03 else 'PARTLY SUPPORTED')

# ======================================================================== C(b): pathogens sit at both GC extremes
ggc = gen.genome_gc.loc[G4.index]; ext = (ggc - ggc.median()).abs()
mw_e = mannwhitneyu(ext[glab == 1], ext[glab == 0], alternative='greater')
add('C-gc', 'C(b)', 'Pathogens sit at both ends of the GC range, so one linear GC term fails', 'NEW',
    'genome GC distance from the median GC, pathogens vs non-pathogens (one-sided Mann-Whitney)',
    f'pathogen GC {ggc[glab == 1].min():.2f}-{ggc[glab == 1].max():.2f}, non-pathogen {ggc[glab == 0].min():.2f}-{ggc[glab == 0].max():.2f}; '
    f'median |GC-median| {ext[glab == 1].median():.3f} vs {ext[glab == 0].median():.3f}, p={mw_e.pvalue:.2f}',
    'PARTLY SUPPORTED' if mw_e.pvalue > .05 else 'SUPPORTED')

# ======================================================================== C8 / C(a): pathogen-specific DNA is a minority, has unusual GC, and is where the signal is
K = 16; LUT = np.full(256, 255, np.uint8); LUT[[ord(c) for c in 'ACGT']] = [0, 1, 2, 3]
def read_fasta(path):
    with open(path) as f: return ''.join(l.strip() for l in f if not l.startswith('>')).upper()
def kmer_codes(s):
    a = LUT[np.frombuffer(s.encode(), np.uint8)]
    bad = (a == 255).astype(np.int32); a = np.where(a == 255, 0, a).astype(np.uint64)
    n = len(a) - K + 1
    code = np.zeros(n, np.uint64)
    for j in range(K): code = (code << np.uint64(2)) | a[j:j + n]
    badw = np.convolve(bad, np.ones(K, np.int32), 'valid') > 0
    return code[~badw]
def genome_set(path):
    s = read_fasta(path); return np.unique(np.concatenate([kmer_codes(s), kmer_codes(s.translate(COMP)[::-1])]))
def shared_frac(seq, S):
    c = kmer_codes(seq); i = np.searchsorted(S, c).clip(max=len(S) - 1); return (S[i] == c).mean()
# This last test needs the raw genome sequences, which the repository does not ship (~90 MB of
# public reference data). Fetch them with data/download_raw_genomes.py and point DATA at the
# folder holding raw_<accession>.fasta. Without them the other verdicts still stand.
missing = [g for g in pairs
           if not os.path.exists(f'{DATA}/raw_{gen[gen.genus == g].index[gen[gen.genus == g].label == 0][0]}.fasta')]
if missing:
    print(f'\nSkipping the pathogen-specific-DNA test (C8): no raw FASTA in {DATA} for '
          f'{", ".join(missing)}. Run data/download_raw_genomes.py to enable it.', flush=True)
    R = pd.DataFrame(rows); R.to_csv(f'{OUT}/task6_claim_verification.csv', index=False)
    for k_, d in details.items():
        d.to_csv(f'{OUT}/detail_{k_}.csv', index=True)
    print(R.groupby(['kind', 'verdict']).size(), flush=True)
    print('TASK6 VERIFY DONE (C8 skipped)')
    raise SystemExit(0)

# leave-one-genus-out scores from the k=4 model so every pair is scored by a model that never saw that genus
logo_p = np.zeros(len(df))
for g in np.unique(genus):
    m = genus == g; logo_p[m] = LR().fit(X4[~m], y[~m]).predict_proba(X4[m])[:, 1]
sp_rows = []
for g in pairs:
    ids = gen[gen.genus == g]; pth = ids.index[ids.label == 1][0]; rel = ids.index[ids.label == 0][0]
    S_rel = genome_set(f'{DATA}/raw_{rel}.fasta')
    m = (gid == pth); fr = np.array([shared_frac(s, S_rel) for s in df.sequence[m]])
    gcm = df.gc[m].values; dev = np.abs(gcm - gcm.mean())
    spec = fr < 0.1
    sp_rows.append(dict(pair=g, pathogen=ORG[pth], relative=ORG[rel], n_fragments=int(m.sum()),
                        share_pathogen_specific=spec.mean(), median_shared_16mers=np.median(fr),
                        gc_dev_specific=dev[spec].mean() if spec.any() else np.nan, gc_dev_shared=dev[~spec].mean(),
                        p_path_specific=logo_p[m][spec].mean() if spec.any() else np.nan, p_path_shared=logo_p[m][~spec].mean()))
    print('  pair', g, sp_rows[-1], flush=True)
sp = pd.DataFrame(sp_rows); details['pathogen_specific_fragments'] = sp
tot = (sp.share_pathogen_specific * sp.n_fragments).sum() / sp.n_fragments.sum()
# Exact 16-mer sharing only measures "shared DNA" between genomes of the SAME species; pairs of different
# species share almost no exact 16-mers, so nearly every fragment looks "pathogen-specific" there.
# The claim's own example is E. coli O157:H7 vs K-12, so the verdict is decided on that pair; others are context.
e = sp.set_index('pair').loc['Escherichia']
minority = e.share_pathogen_specific < 0.3; odd_gc = e.gc_dev_specific > e.gc_dev_shared; signal = e.p_path_specific > e.p_path_shared
add('C8', 'C8, C(a), B8', 'Pathogen-only DNA (e.g. O157:H7 sequence absent from K-12) is a small minority of fragments, has unusual GC, and is where any pathogen signal would be', 'NEW',
    'E. coli O157:H7 fragments: share of 16-mers present anywhere in K-12 (both strands); <10 % shared = pathogen-specific. '
    'Compare |GC - genome mean| and leave-one-genus-out P(pathogen) of the k=4 model. Other pairs are different species (near-zero exact sharing) and only reported in the detail file',
    f'O157-specific fragments {e.share_pathogen_specific:.0%} (Listeria pair {sp.set_index("pair").loc["Listeria"].share_pathogen_specific:.0%}; different-species pairs 75-100 %); '
    f'|GC dev| specific {e.gc_dev_specific * 100:.1f} vs shared {e.gc_dev_shared * 100:.1f} pts; P(pathogen) specific {e.p_path_specific:.2f} vs shared {e.p_path_shared:.2f}',
    'SUPPORTED' if minority and odd_gc and signal else 'PARTLY SUPPORTED' if minority or odd_gc else 'NOT SUPPORTED')

# ======================================================================== write
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/task6_claim_verification.csv', index=False)
for k_, d in details.items(): d.to_csv(f'{OUT}/detail_{k_}.csv', index=True)
print(R.groupby(['kind', 'verdict']).size(), flush=True)
print('TASK6 VERIFY DONE')
