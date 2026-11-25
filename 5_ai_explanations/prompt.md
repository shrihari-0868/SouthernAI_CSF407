# Prompt given to every AI system (identical text, fresh conversation, no tools, no files)

You are helping interpret the results of a machine-learning project. Please explain what the
model has (and has not) learned. Give your interpretation first, then list **5-8 numbered,
specific claims** that could be checked against the data (for example "pathogen fragments
should have more X than non-pathogen fragments"). Keep the answer under 600 words.

## Setup
- 24 bacterial reference genomes: 12 labelled "pathogen", 12 "non-pathogen". Six genera are
  matched pairs (one pathogen + one harmless relative): Bacillus (anthracis vs subtilis),
  Pseudomonas (aeruginosa vs putida), Listeria (monocytogenes vs innocua), Escherichia
  (coli O157:H7 vs coli K-12), Corynebacterium (diphtheriae vs glutamicum), Streptococcus
  (pyogenes vs thermophilus). The other 12 genomes are unpaired (pathogens: M. tuberculosis,
  V. cholerae, S. Typhimurium, Y. pestis, H. pylori, S. aureus; non-pathogens: L. lactis,
  S. coelicolor, B. longum, L. plantarum, Synechocystis, S. meliloti).
- 400 random 1000-bp fragments per genome (9600 total). All fragments are exactly 1000 bp.
- Fragments are tokenised into overlapping k-mers (k = 3, 4, 6), read on the stored strand only.
- Representations: Bag-of-Words frequencies, TF-IDF, Word2Vec (mean of k-mer vectors).
- Train on 18 genomes, test on 6 never-seen genomes (H. pylori, L. lactis, both Listeria,
  both Corynebacterium).

## Results
1. Mean GC: pathogens 47.3 %, non-pathogens 50.7 %. Within matched pairs the GC gap is < 1
   point in 4 of 6 pairs (Bacillus: pathogen 8 points lower; Pseudomonas: pathogen 5 points higher).
2. Rarest 4-mer in the whole corpus is CTAG; 17 of the 20 rarest 6-mers contain CTAG.
3. PCA of 4-mer frequencies: PC1 is a smooth GC gradient; the two classes overlap completely.
   For each fragment, nearest neighbours from *other* genomes share its label only 35-40 % of
   the time (50 % = chance).
4. Classifiers (logistic regression, decision tree, random forest, 1-hidden-layer MLP) on all
   representations: fragment accuracy on the 6 unseen genomes 47-61 %; training accuracy 71-100 %.
   A GC-content-only model gets 52 %. Leave-one-genus-out cross-validation on the training
   genomes: 40-53 %. 200 runs with randomly shuffled genome labels gave test accuracies from
   26 % to 78 % (so a 61 % result is within luck).
5. Removing GC content from the features (regressing it out) does not lower test accuracy
   (still 47-60 %).
6. The same features identify *which genome* a fragment came from with 74-81 % accuracy
   (18-way, chance 5.6 %).
7. Inside a matched pair, can the model rank the pathogen's fragments above its relative's
   fragments when that pair was held out? Mean AUC 0.50-0.53 (per pair from 0.25 to 0.79).
8. Logistic-regression coefficients (standardised k-mer frequencies, positive = pathogen):
   - k=3 most positive: CGC, GCG, GGT, ACC, ATA, TAT; most negative: GGG, CAA, TCC, CGT, GGA, GGC.
   - k=4 most positive: GCGC, TAAA, TTTA, GCTA, CGAC, CGCG; most negative: GGGC, GGGG, CCGT, TCCG, ACGG, GTGC.
   - k=6 most positive: TTTTAA, CGCGGC, CTCTGC, CGCGTT, CGCCGG; most negative: CAGCTG, CTTAAG, GAGCTC, TGCCGA, TTCCTT.
   - Correlation between a word's coefficient and its own GC fraction: about -0.05 to -0.27.
9. Word differences that are consistent across all 6 matched pairs: no more than expected by
   chance (exact sign-flip permutation p = 0.53-1.0). The coefficient ranking agrees with the
   pathogen-minus-relative differences of the *training* pairs (Spearman ~0.25) but not of the
   *unseen* pairs (Spearman ~0.0).

## Questions
a. What, if anything, did the model learn about pathogenicity?
b. Why do the listed k-mers get the largest weights? What might they correspond to biologically?
c. Is the model learning sequence "grammar" or superficial characteristics (composition,
   genome identity, relatedness)?
d. List your 5-8 numbered, checkable claims.
