# Task 7 - final challenge: can the model "read" DNA?

> *"If DNA/RNA is a four-letter language, can an AI learn enough of its grammar, vocabulary and
> context to recognise sequence patterns associated with disease?"*

**Answer: it learns real vocabulary, but that vocabulary identifies the organism, not the disease.**

## How to run
```
python run_final_model.py ../data/dataset.csv results      # ~45 min on CPU (the CNN dominates)
python logo_addon.py ../data/dataset.csv results 2-mer   # only if a model is missing from the LOGO table
python finish_figures.py ../data/dataset.csv results             # CNN filter table + figures only
```
`run_final_model.py` does everything. The two helper scripts exist because the first run crashed on a dead
line *after* writing all results (fixed since); rather than repeat the 45-minute CNN stage, the
remaining pieces were produced separately. A fresh `run_final_model.py` now produces everything by itself.

## What this adds to Tasks 1-5
Tasks 4 and 5 scored nine representations on one fixed test split. The questions left open were:
which model would we actually *choose*, does a model that reads raw sequence do better than one
handed k-mers, does any model use **word order** rather than letter composition, and does the
result survive when every genus in turn is held out instead of one fixed split of 6 genomes.

1. **Model selection done honestly.** Eight candidates compared by genus-grouped CV on the 18
   training genomes only. The winner is declared *before* the test genomes are touched.
2. **A CNN on raw sequence**, which invents its own words as convolution filters instead of being
   given a k-mer vocabulary. Reverse-complement augmentation, scored on both strands.
3. **Grammar tests**: re-score the test fragments after destroying different amounts of order.
4. **Leave-one-genus-out over all 24 genomes** with a 200-run label-permutation null, so the result
   no longer depends on which 6 genomes happened to be the test set.

## Results

### 1. Model selection and the unseen test genomes
`task7_model_comparison.csv`, `fig_model_comparison.png`

| Model | CV acc (18 train genomes) | Test acc (6 unseen) | Test AUC | Genomes called right | corr(score, GC) |
|---|---|---|---|---|---|
| Majority | 0.500 | 0.500 | 0.500 | 3/6 | - |
| GC-only | 0.570 | 0.518 | 0.426 | 3/6 | -1.00 |
| Letters (1-mer) | 0.570 | 0.522 | 0.427 | 3/6 | -1.00 |
| **2-mer (selected as final)** | **0.576** | 0.573 | 0.596 | 4/6 | -0.53 |
| 3-mer | 0.542 | 0.587 | 0.607 | 4/6 | -0.48 |
| 4-mer | 0.499 | 0.579 | 0.607 | 4/6 | -0.45 |
| Multi-scale 1-4-mer | 0.499 | 0.578 | 0.607 | 4/6 | -0.45 |
| CNN (raw sequence) | 0.509 | 0.512 | 0.505 | 3/6 | -0.65 |

- The **2-mer** model wins the honest selection: 16 numbers per fragment, barely more than counting
  letters. Nothing more elaborate earns its place.
- The **CNN is no better than chance** (51 %) despite reading raw sequence. Its scores still
  correlate -0.65 with GC content: given freedom to find any motif, it rediscovered composition.
- Every model's score correlates strongly and *negatively* with GC. The whole model family is a
  dressed-up AT-richness detector.

### 2. The test genomes, one by one
`task7_test_per_genome.csv` - the 2-mer final model:

| Genome | True label | mean P(pathogen) | Call |
|---|---|---|---|
| *Helicobacter pylori* | pathogen | 0.555 | pathogen ✓ |
| *Lactococcus lactis* | non-pathogen | 0.458 | non-pathogen ✓ |
| *Listeria monocytogenes* | pathogen | 0.640 | pathogen ✓ |
| *Listeria innocua* | **non**-pathogen | 0.671 | pathogen ✗ |
| *Corynebacterium diphtheriae* | **pathogen** | 0.428 | non-pathogen ✗ |
| *Corynebacterium glutamicum* | non-pathogen | 0.293 | non-pathogen ✓ |

4/6 looks respectable until you read the pairs. **Both** *Listeria* are called pathogen; **both**
*Corynebacterium* are called non-pathogen. The model has sorted the test set by genus and got the
two unpaired genomes right. Within a pair, where the question actually lives, it is 0 for 2.

### 3. Does it read order, or only composition?
`task7_grammar_tests.csv`. Each test fragment is re-scored after a transformation:

| Model | original | letters shuffled | 10-bp blocks shuffled | reverse complement |
|---|---|---|---|---|
| GC-only | 0.518 | 0.518 | 0.518 | 0.518 |
| Letters (1-mer) | 0.522 | 0.522 | 0.522 | 0.517 |
| 3-mer | 0.587 | **0.507** | 0.571 | 0.589 |
| 4-mer | 0.579 | **0.485** | 0.568 | 0.579 |
| Multi-scale | 0.578 | **0.485** | 0.568 | 0.579 |
| CNN | 0.512 | 0.484 | 0.535 | 0.512 |

This is the clearest result in the project, and it cuts both ways:
- **Shuffling the letters costs 8-9 points** and decorrelates the scores (r = 0.35 for k=4). So the
  k-mer models are *not* merely counting letters: real multi-letter word structure is being used.
  This is genuine "vocabulary".
- **Shuffling 10-bp blocks costs only 1-2 points** (r = 0.87). Nothing beyond about 10 bp matters.
  There is no long-range "grammar" in what the model learned.
- **Reverse complement changes nothing** - correct, since the same DNA read from the other strand
  is the same molecule. A good sanity check on the pipeline.

So: vocabulary yes, grammar no. And the vocabulary that is used is the lineage signature.

### 4. Leave-one-genus-out over all 24 genomes
`task7_genome_level_logo.csv`, `task7_permutation_null.csv`, `fig_permutation_null.png`.
Every genus in turn is held out entirely (so a pathogen and its relative are always removed
together), fragment scores are averaged into one call per genome, and the whole procedure is
repeated 200 times with genome labels randomly reassigned.

| Model | Genomes correct /24 | AUC | perm. p | 95th pct of null | Pairs won /6 |
|---|---|---|---|---|---|
| GC-only | 13 | 0.535 | 0.62 | 17 | 4 |
| Letters (1-mer) | 13 | 0.542 | 0.62 | 17 | 4 |
| 2-mer (final) | 14 | 0.604 | 0.33 | 17 | 3 |
| 3-mer | 15 | 0.632 | 0.24 | 17 | 3 |
| 4-mer | 14 | 0.611 | 0.31 | 17 | 3 |
| Multi-scale | 14 | 0.611 | 0.31 | 17 | 3 |
| CNN (raw sequence) | 14 | 0.590 | 0.36* | 17 | 3 |

*\*the CNN's p-value uses the GC-only permutation null; 200 × 18 CNN fits was not affordable on CPU.*

**No model is significant.** Chance is 12/24, random labellings reach 17/24 in 5 % of runs, and the
best real model manages 15. Pathogens beat their own relatives in 3 of 6 pairs, exactly a coin flip.

Sorting the 24 genomes by the final model's score shows what it is really ranking:

> *S. aureus* (0.74, 33 % GC) > *L. innocua* (0.68) > *B. anthracis* (0.66) > ... >
> *M. tuberculosis* (0.30, 66 % GC) > *C. glutamicum* (0.29, 54 % GC)

AT-rich genomes at the top, GC-rich genomes at the bottom, labels scattered through both ends.
*M. tuberculosis* is one of the most dangerous pathogens in the dataset and the model ranks it
third from last, because it is GC-rich.

### 5. What the CNN's filters became
`task7_cnn_filters.csv`. Each of the 64 first-layer filters is a learned 15-bp "word". The most
label-discriminative one has consensus `TGCTGTATTTACTTA`, 27 % GC, and reaches AUC 0.607 on its
own - but it correlates **-0.73** with fragment GC. Across all 64 filters the median |correlation
with GC| is 0.25. The network, free to learn any motif, mostly learned AT-richness detectors.

## Verdict on the final challenge

| Question | Answer |
|---|---|
| Can it learn an alphabet and vocabulary? | **Yes.** Word structure beyond composition is used (letter-shuffling costs 8-9 points). |
| Can it learn grammar / long-range context? | **No.** Shuffling 10-bp blocks is nearly free. |
| Can it recognise disease? | **No.** Nothing survives holding out a genus; 15/24 at best, p = 0.24. |
| What did it actually learn? | **Genome and lineage identity** - a "dialect" detector. It names the organism (74-81 % among 18) and ranks genomes by AT-richness. |

Why this is the expected answer, not a failure of the method: pathogenicity lives in a few percent
of a genome (toxins, secretion systems, pathogenicity islands), so most random 1-kb fragments from
a pathogen carry nothing pathogen-specific. With 24 genomes the effective sample size is 24, not
9600. And Task 6 found that the pathogen-only DNA in *E. coli* O157:H7 is scored as *less*
pathogen-like than the backbone it shares with harmless K-12 - the model is not even looking where
the answer is.

## Files
| File | Contents |
|---|---|
| `run_final_model.py` | the whole pipeline |
| `logo_addon.py`, `finish_figures.py` | helpers (see "How to run") |
| `task7_model_comparison.csv` | 8 candidates: CV and test accuracy, AUC, genome calls, GC correlation |
| `task7_test_per_genome.csv` | per-genome scores and calls on the 6 unseen genomes, every model |
| `task7_grammar_tests.csv` | the four order-destroying conditions |
| `task7_genome_level_logo.csv` | leave-one-genus-out over 24 genomes, with permutation p-values |
| `task7_permutation_null.csv` | the 200 random-label runs behind those p-values |
| `task7_genome_scores.csv` | each genome's held-out score, every model (the ranking above) |
| `task7_cnn_filters.csv` | the 64 learned 15-bp filters, their consensus and GC correlation |
| `task7_final_model.json` | which model the CV selected |
| `fig_model_comparison.png`, `fig_genome_scores_logo.png`, `fig_permutation_null.png` | figures |
| `results_log.txt` | full run log |
