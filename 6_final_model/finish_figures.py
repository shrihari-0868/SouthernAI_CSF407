"""Completes the two steps run_final_model.py did not reach before it stopped: the CNN filter table and the
three figures. Everything else (model comparison, per-genome test calls, grammar tests, leave-one-genus-out,
permutation null) was already written to OUT by the main run and is only read here.

The CNN is retrained on the 18 training genomes with the same fixed recipe and seed as run_final_model.py, so the
filters belong to the same model as the reported test numbers.

Run:  python finish_figures.py ../data/dataset.csv results
"""
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
import torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CSV, OUT = sys.argv[1], sys.argv[2]
df = pd.read_csv(CSV)
y = df.label.to_numpy(int); gid = df.genome_id.to_numpy(str)
is_tr = (df.split_suggested == 'train').to_numpy(bool)
tr_idx = np.where(is_tr)[0]
gc1 = np.array([[(s.count('G') + s.count('C')) / len(s)] for s in df.sequence], np.float32)

# --- the CNN pieces, copied verbatim from run_final_model.py so the recipe cannot drift
import torch.nn as nn
LUT = np.zeros(256, np.int64); LUT[[ord(c) for c in 'ACGT']] = [0, 1, 2, 3]
def onehot(S):
    idx = np.stack([LUT[np.frombuffer(s.encode(), np.uint8)] for s in S])
    X = np.zeros((len(S), 4, idx.shape[1]), np.float32)
    np.put_along_axis(X, idx[:, None, :], 1.0, axis=1)
    return torch.from_numpy(X)
def revcomp_t(X): return torch.flip(X, dims=[1, 2])
class SeqCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(4, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
                               nn.Conv1d(64, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU())
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(s, x):
        h = s.conv(x); return s.head(torch.cat([h.amax(-1), h.mean(-1)], 1)).squeeze(1)
def fit_cnn(X, yy, epochs=6, seed=0):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    m = SeqCNN(); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    yt = torch.tensor(yy, dtype=torch.float32)
    pw = torch.tensor((yy == 0).sum() / max(1, (yy == 1).sum()), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        m.train()
        for b in np.array_split(g.permutation(len(yy)), max(1, len(yy) // 128)):
            xb = X[b]; flip = torch.from_numpy(g.random(len(b)) < 0.5)
            xb = torch.where(flip[:, None, None], revcomp_t(xb), xb)
            opt.zero_grad(); lossf(m(xb), yt[b]).backward(); opt.step()
    return m

torch.set_num_threads(min(8, os.cpu_count() or 1))
OH = onehot(df.sequence.to_numpy(str))
print('training the CNN on the 18 training genomes (same recipe and seed as the main run)...', flush=True)
cnn = fit_cnn(OH[tr_idx], y[tr_idx])

# ---- which "words" did the network invent? one row per first-layer filter
W = cnn.conv[0].weight.detach().numpy()        # 64 filters x 4 letters (ACGT) x 15 positions
sub = tr_idx[::4]
with torch.no_grad():
    acts = torch.relu(cnn.conv[1](cnn.conv[0](OH[sub]))).amax(-1).numpy()   # fragments x filters
yy = y[sub]; gcv = gc1[sub, 0]
rows = []
for fi, f in enumerate(W):
    best = ''.join('ACGT'[j] for j in f.argmax(0))          # strongest letter at each of the 15 positions
    a = acts[:, fi]
    rows.append(dict(filter=fi, consensus=best, consensus_gc=sum(c in 'GC' for c in best) / 15,
                     mean_act_pathogen=a[yy == 1].mean(), mean_act_nonpathogen=a[yy == 0].mean(),
                     corr_act_with_fragment_gc=np.corrcoef(a, gcv)[0, 1], auc_act_for_label=roc_auc_score(yy, a)))
F = pd.DataFrame(rows).sort_values('auc_act_for_label', ascending=False)
F.to_csv(f'{OUT}/task7_cnn_filters.csv', index=False)
print(F.head(5).round(3).to_string(index=False), flush=True)
print('filter |corr with GC| median', round(float(F.corr_act_with_fragment_gc.abs().median()), 3),
      '| best filter AUC', round(float(F.auc_act_for_label.max()), 3), flush=True)

# ---- figures, drawn from the CSVs the main run already wrote
res = pd.read_csv(f'{OUT}/task7_model_comparison.csv')
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(res)); w = 0.38
ax.bar(x - w / 2, res.cv_fragment_acc, w, label='CV on 18 training genomes (grouped by genus)')
ax.bar(x + w / 2, res.test_fragment_acc, w, label='6 unseen test genomes')
ax.axhline(0.5, color='k', lw=0.8, ls='--'); ax.set_xticks(x); ax.set_xticklabels(res.model, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('fragment accuracy'); ax.set_ylim(0.3, 0.8); ax.legend(fontsize=8); ax.set_title('Candidate final models')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_model_comparison.png', dpi=130); plt.close()

order = ['GC-only', 'Letters (1-mer)', '2-mer', '3-mer', '4-mer', 'Multi-scale 1-4-mer', 'CNN (raw sequence)']
gs = pd.read_csv(f'{OUT}/task7_genome_scores.csv'); mods = [m for m in order if m in set(gs.model)]
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, m in enumerate(mods):
    d = gs[gs.model == m]; jit = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.25
    ax.scatter(i + jit, d.logo_mean_p_pathogen, c=np.where(d.label == 1, 'tab:red', 'tab:blue'), s=22, alpha=0.8)
ax.axhline(0.5, color='k', lw=0.8, ls='--'); ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('mean P(pathogen) for a held-out genome')
ax.set_title('Leave-one-genus-out, every genome (red = pathogen, blue = non-pathogen)')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_genome_scores_logo.png', dpi=130); plt.close()

nd = pd.read_csv(f'{OUT}/task7_permutation_null.csv'); lr = pd.read_csv(f'{OUT}/task7_genome_level_logo.csv')
mods = [m for m in order if m in set(nd.model)]
fig, axs = plt.subplots(1, len(mods), figsize=(3 * len(mods), 3), sharey=True)
for a, m in zip(np.atleast_1d(axs), mods):
    a.hist(nd[nd.model == m].genome_acc * 24, bins=np.arange(-0.5, 25), color='0.7')
    a.axvline(lr[lr.model == m].n_genomes_correct.iloc[0], color='tab:red', lw=2)
    a.set_title(m, fontsize=9); a.set_xlabel('genomes correct of 24')
np.atleast_1d(axs)[0].set_ylabel('permutation runs')
fig.suptitle('Observed (red) vs 200 random genome labellings', fontsize=10)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_permutation_null.png', dpi=130); plt.close()
if not os.path.exists(f'{OUT}/task7_final_model.json'):
    import json
    json.dump(dict(final_model=str(res[res.chosen_as_final].model.iloc[0]),
                   selected_by='genus-grouped CV on the 18 training genomes only'),
              open(f'{OUT}/task7_final_model.json', 'w'), indent=1)
print('FINISH DONE', flush=True)
