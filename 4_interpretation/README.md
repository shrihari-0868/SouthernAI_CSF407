# Task 5 - what has the model learned?
- task5_kmer_coefficients.csv: every k-mer (k=3,4,6) with its standardized logistic-regression coefficient, GC fraction of the word,
  leave-one-group-out sign stability, and in how many of the 6 matched pairs the pathogen has more of that word than its relative
- task5_pair_consistency_summary.csv: do any words differ consistently inside matched pairs? (exact 64-way permutation test), and does the model's
  coefficient order agree with the pair differences on pairs it trained on vs pairs it never saw
- task5_per_pair_agreement.csv: the same agreement, pair by pair
- task5_probes.csv: genome-identity probe, random vs genome-grouped split, within-pair AUC on genomes held out
- run_interpretation.py: code (python run_interpretation.py REPRESENTATIONS_DIR OUT_DIR)
Model: standardized BoW frequencies + logistic regression (C=0.1), same as Task 4. Matched pairs: Bacillus, Corynebacterium, Escherichia, Listeria, Pseudomonas, Streptococcus.
