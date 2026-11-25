# System A: Claude Haiku (fresh context, no tools, prompt.md verbatim). Response copied unedited.

**Interpretation**

The model has learned almost nothing that generalises about pathogenicity. On the six unseen genomes it scores 47-61%, which is at or below the 50% chance level for a balanced binary task. The 61% best case falls inside the range from label-shuffled runs (26-78%), so it cannot be told apart from luck. The features do carry strong genome-identity information (74-81% on an 18-way task), and that is most likely what the classifier is using. Training accuracy of 71-100% shows memorisation of the training genomes rather than a transferable signal.

The evaluation is also weak in structure. The test set has only six genomes, so the effective sample size is six and the score depends heavily on which genomes are held out. Pathogen and non-pathogen labels are confounded with GC content, genus, and genome identity. Because the matched pairs (for example B. anthracis vs B. subtilis) differ very little, any true pathogen signal is small and swamped by these confounds.

The large coefficients are best read as composition and codon-usage features of individual genomes, not virulence markers. GC-rich k-mers (GCGC, CGCG, CGCGGC) and AT-rich ones (TAAA, TTTA, TTTTAA) dominate, and several 3-mers (GCG, GGT, ACC) may reflect codon bias. Reading only the stored strand also means strand-asymmetric k-mers can pick up orientation skew. None of this shows a biological link to pathogenicity.

**Checkable claims**

1. A label-permutation test on the real 6-genome result should show the real-label accuracy is not significantly above the shuffled distribution (p > 0.05), consistent with the 26-78% spread.
2. Within the six matched pairs, the sign of (pathogen minus relative) frequency for the top-20 positive k-mers should agree with the coefficient sign no more often than chance (about 50%). The reported sign-flip p-values of 0.53-1.0 predict this.
3. Within-pair AUC should vary widely by genus (0.25 to 0.79) with a mean near 0.5. A pair well below 0.5 means its composition direction is reversed relative to the model, which argues against a shared pathogen signal.
4. The Spearman correlation between coefficients and training-pair differences (about 0.25) should be near zero on the unseen pairs (about 0.0). A substantial unseen-pair correlation would indicate real transfer.
5. Models trained with GC regressed out should still perform at chance (47-60%). The GC-only model (52%) and the full model should both be near 50%, with no meaningful gap between them.
6. Errors on unseen genomes should cluster by genome identity rather than by label. A confusion analysis should show misclassified fragments go to a training genome with a similar GC range rather than to a consistent pathogen or non-pathogen class.
7. CTAG frequency per genome should be low across all 24 genomes, and its rarity should not separate the two labels. A label split in CTAG rarity would indicate a systematic pathogen signal.
8. Leave-one-genus-out accuracy (40-53%) should fall at or below chance for several genera, confirming the model does not generalise across genera.
