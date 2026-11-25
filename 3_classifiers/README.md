# Task 4 – classifiers on the Task 3 representations

Files
- results_task4.csv: 9 representations x 4 models (train acc, test acc, test genomes correct of 6, grouped-CV)
- per_genome_accuracy.csv: accuracy on each of the 6 unseen test genomes for every configuration
- controls_task4.csv: majority and GC-only baselines; models re-run with GC regressed out of the features
- null_random_labels.csv: 200 runs with randomly assigned genome labels (what luck looks like)
- cv_leave_group_out_balanced.csv: fair leave-one-group-out CV (class-balanced training)
- run_classifiers.py, run_controls.py, run_cv_balanced.py: the code (python script.py REPRESENTATIONS_DIR OUT_DIR)

Settings (fixed in advance, not tuned on the test set)
- LogReg: standardized features, C=0.1. DecisionTree: depth 8, min leaf 10. RandomForest: 150 trees, min leaf 5.
- MLP: one hidden layer of 64 units, L2 0.1, 40 epochs, no early stopping (it would leak fragments of the same genome).
- BoW counts converted to frequencies; TF-IDF and Word2Vec used as delivered in Task 3.
- Same train/test split as Task 3 (6 unseen test genomes). No transformer, as requested.
