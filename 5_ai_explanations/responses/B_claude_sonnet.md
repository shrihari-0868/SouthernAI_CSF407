# System B: Claude Sonnet (fresh context, no tools, prompt.md verbatim). Response copied unedited.

**Interpretation**

The model has not learned pathogenicity. On genomes it never saw, it scores 47-61 % against a 50 % chance level. A GC-only model gets 52 %. Shuffled-label runs span 26-78 %, so 61 % is within luck. Leave-one-genus-out accuracy is 40-53 %. Nearest neighbours from other genomes share a fragment's label only 35-40 % of the time, which is below chance. That pattern means similar-looking fragments tend to come from different-label genomes, so the label is not organised by composition.

What the model does learn is genome identity. The same features identify the source genome with 74-81 % accuracy (chance 5.6 %). Training accuracy of 71-100 % against test accuracy near chance is classic memorisation: with 18 genomes, "pathogen" is just a lookup of which genomes carried that label. Phylogeny and GC-driven composition are what the k-mer frequencies encode, and they are not tied to the label.

The GC gap (47.3 % vs 50.7 %) is a weak, confounded cue. It is driven by which species were picked, and it flips direction inside pairs (Bacillus: pathogen 8 points lower; Pseudomonas: pathogen 5 points higher). Regressing GC out changes nothing. That means the signal was never GC, and also that the remaining signal is not real pathogen signal. It is genus/genome identity.

**b. Why those k-mers get large weights**

Large weights come from fitting 18 genomes' idiosyncrasies, not from biology. GC-rich words (GCGC, CGCG, CGC, GCG) are positive and GGG/GGGC/GGGG are negative. That reflects which training pathogens happen to be GC-rich or GC-poor, and the weak GC correlation (-0.05 to -0.27) shows even that is not systematic. AT-rich words (TAAA, TTTA, TTTTAA) are positive, which likely reflects AT-rich pathogens such as H. pylori, S. aureus and Listeria in the training set. Some of the negative 6-mers are palindromic restriction sites: CAGCTG (PvuII), CTTAAG (AflII), GAGCTC (SacI). Their depletion is a known restriction-modification and codon-usage signature of particular lineages, which makes them marks of genus, not virulence. CTAG is the rarest 4-mer corpus-wide, a well-known bacterial avoidance (DNA bending and mismatch-repair effects). It is universal, so it is not a class signal. The weights agree with the training pairs' pathogen-minus-relative differences (Spearman ~0.25) but not with the unseen pairs' (~0.0), so they do not generalise. Nothing in this list is convincingly virulence-related.

**c. Grammar or superficial features?**

Superficial. Context-free k-mer counts cannot capture grammar anyway (read on one strand, 1000-bp random windows, with coding/non-coding and frame unknown). What is captured is composition, genome identity and relatedness. Inside held-out matched pairs the AUC is 0.50-0.53, so the model cannot separate a pathogen from its harmless relative. Word differences consistent across all 6 pairs are no more than chance (p = 0.53-1.0).

**d. Checkable claims**

1. Fragment-level test accuracy will fall to near chance, or below, on held-out genomes whenever the whole genus is held out. Genome-grouped cross-validation (never splitting a genome across train and test) should give 45-55 % on every representation.
2. Genome-identity classification stays far above chance (>70 %), and a confusion matrix will show errors concentrated within genera or among genomes of similar GC.
3. Within each matched pair, the pathogen-minus-relative k-mer frequency differences will have inconsistent signs across the six pairs. Fewer words than expected by chance will share a sign in all six.
4. The positive k-mers (GCGC, CGCG, TAAA) will be enriched mostly in a few specific training pathogens. Dropping any one training genome will change the top-20 coefficient list substantially (overlap well under 50 %).
5. Palindromic 6-mers (CAGCTG, CTTAAG, GAGCTC) will show large per-genome frequency variance but low variance between classes after pooling by genus. A variance decomposition should assign most of their variance to genus/genome, not label.
6. Permutation-based p-values for the real test accuracy will be non-significant (p > 0.05), and the shuffled-label accuracy distribution will have a spread of tens of points because only 6 test genomes (clustered fragments) exist.
7. Reducing to 1 fragment per genome (genome-level voting) would not exceed chance. Averaging predictions per genome will give an all-or-nothing outcome on 6 genomes, e.g. 3/6 correct.
8. Adding pathogenicity-relevant features absent here (gene content, virulence-factor homologs, coding-frame-aware codon usage, 1000-bp windows aligned to genes) would be the minimal change needed. Genome-level k-mer features and phylogeny-aware splits are expected to remain near chance.
