# AI and the Language of DNA/RNA

**Team SouthernAI** — Shrihari Karthik (2023B4A70868G), Joel George (2023AAPS0625G),
Hrishikesh M S (2024A1PS0289G), Aman S Sumod
CS F407 Artificial Intelligence, BITS Pilani K.K. Birla Goa Campus

> If DNA is a four-letter language, can an AI learn enough of its vocabulary and grammar to
> recognise which organisms cause disease?

**Short answer: it learns a real vocabulary, and that vocabulary identifies the organism, not the
disease.** Every promising result disappears once whole genomes and whole genera are held out and
compared against randomly shuffled labels.

---

## What we found

| Question | Answer |
|---|---|
| Best accuracy on six unseen genomes | 61.3% — but 12% of random-label runs do as well or better |
| Holding out a whole genus, across all 24 genomes | best 15/24 correct (chance 12, random labelling reaches 17 one run in twenty), p = 0.24 |
| Can it separate a pathogen from its own harmless relative? | No. Mean within-pair AUC 0.50–0.53; the pathogen wins 3 of 6 pairs |
| Can it name the source genome? | Yes — 74–81% among 18 (chance 5.6%) |
| Does it use word order? | Up to ~10 bases, yes: shuffling letters costs 8–9 points. Beyond that, no: shuffling 10 bp blocks costs 1–2 |
| Does a network reading raw sequence do better? | No — 51.2%, and its scores correlate −0.65 with GC content |

The clearest single piece of evidence: the DNA unique to *E. coli* O157:H7 and absent from the
harmless K-12 strain — exactly where a real pathogen signal would live — is scored by our model as
**less** pathogen-like (0.62) than the backbone the two strains share (0.78).

---

## Repository layout

| Folder | Contents |
|---|---|
| `data/` | the curated dataset: 9,600 fragments of 1,000 bp from 24 genomes (`dataset.csv`), genome metadata (`genomes.csv`), and a script to re-download the raw sequences |
| `1_language_exploration/` | notebook: k-mer tokenisation, vocabulary statistics, word frequencies, PCA, and the GC confound checks |
| `2_representations/` | builds the Bag-of-Words, TF-IDF and Word2Vec matrices that later stages consume |
| `3_classifiers/` | nine representations against four model families, grouped cross-validation, shortcut controls, and the random-label null |
| `4_interpretation/` | which words carry weight, how stable they are, the matched-pair permutation test, and genome-identity probes |
| `5_ai_explanations/` | one prompt, three language models, their unedited answers, and a script that tests every new claim against the data |
| `6_final_model/` | honest model selection, a convolutional network on raw sequence, the word-order experiments, and leave-one-genus-out over all 24 genomes |
| `report/` | the LaTeX report |

Each stage folder has its own README with the full result tables and what they mean.

---

## Reproducing the results

```bash
pip install -r requirements.txt
```

### 1. Exploration
```bash
jupyter notebook 1_language_exploration/dna_language_exploration.ipynb
```
Point the notebook at `data/dataset.csv` and run all cells.

### 2. Build the representations
These matrices are about 84 MB, so they are not stored here. Rebuild them first — the counts and
TF-IDF matrices come out bit-identical to the ones used for the reported results.

```bash
cd 2_representations
python build_representations.py ../data/dataset.csv out
```
Takes a few minutes, mostly Word2Vec.

### 3. Classifiers and controls
```bash
cd 3_classifiers
python run_classifiers.py  ../2_representations/out results
python run_controls.py     ../2_representations/out results
python run_cv_balanced.py  ../2_representations/out results
```

### 4. Interpretation
```bash
cd 4_interpretation
python run_interpretation.py ../2_representations/out results
```

### 5. Verify what the language models claimed
```bash
cd 5_ai_explanations
python verify_claims.py ../data .. results
```
One test needs the raw *E. coli* sequences; see "Raw genomes" below.

### 6. The final model
```bash
cd 6_final_model
python run_final_model.py ../data/dataset.csv results
```
About 45 minutes on a CPU, nearly all of it training the convolutional network eighteen times for
the leave-one-genus-out evaluation. Two helpers exist for partial reruns: `finish_figures.py`
redraws the figures and the filter table without repeating the training, and `logo_addon.py` adds
one more model to the leave-one-genus-out table.

### Raw genomes
The 24 raw FASTA files total about 90 MB and are not stored here, since they are public reference
sequences retrievable by accession. `dataset.csv` already holds the extracted fragments, so you
only need the raw files to rebuild the dataset from scratch or to repeat the 16-mer sharing
comparison between the two *E. coli* strains.

```bash
cd data
python download_raw_genomes.py raw
```

---

## The dataset

24 complete bacterial genomes from NCBI RefSeq, twelve disease-causing and twelve harmless, with
400 randomly positioned 1,000 bp fragments drawn from each.

Six genera contribute a **matched pair** — a pathogen together with a close harmless relative — so
that family resemblance stays constant while the label changes:

| Genus | Pathogen | Harmless relative |
|---|---|---|
| *Bacillus* | *B. anthracis* Ames | *B. subtilis* 168 |
| *Pseudomonas* | *P. aeruginosa* PAO1 | *P. putida* KT2440 |
| *Listeria* | *L. monocytogenes* EGD-e | *L. innocua* Clip11262 |
| *Escherichia* | *E. coli* O157:H7 Sakai | *E. coli* K-12 MG1655 |
| *Corynebacterium* | *C. diphtheriae* NCTC 13129 | *C. glutamicum* ATCC 13032 |
| *Streptococcus* | *S. pyogenes* SF370 | *S. thermophilus* LMG 18311 |

The other twelve are unpaired and span a wide compositional range, from 32.9% G and C content in
*Staphylococcus aureus* to 72.1% in *Streptomyces coelicolor*.

Six genomes are held out entirely: *H. pylori*, *L. lactis*, and both members of the *Listeria*
and *Corynebacterium* pairs. **No genome contributes fragments to both sides of a split, in any
experiment here.** Fragments from one genome are pieces of a single molecule written in one
organism's style, so splitting them at random would let a model memorise a genome and then be
rewarded for recognising it.

---

## Why a negative result is the interesting one

Pathogenicity lives in a small fraction of a genome — toxins, secretion systems, horizontally
acquired islands — so most random 1 kb fragments from a dangerous organism carry nothing
pathogen-specific at all.

There is also a counting problem worth stating plainly. The models train on 7,200 fragments, which
sounds like plenty, but the label only changes from one genome to the next, so the effective
sample size is **18, not 7,200**. That is why the random-label null here is so wide (26–78% on the
fixed split) and why any single held-out accuracy means nothing on its own. Any study shaped like
this one that reports a held-out accuracy without grouped splits and a permutation null will
report a positive result whether or not one exists.

Ranking the 24 genomes by the final model's held-out score makes what it learned plain:
*S. aureus* at the top (0.74, 33% GC), *M. tuberculosis* second from last (0.30, 66% GC). It is a
ranking by AT-richness with the labels scattered through both ends.
