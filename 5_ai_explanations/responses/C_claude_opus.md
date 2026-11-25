# System C: Claude Opus (fresh context, no tools, prompt.md verbatim). Response copied unedited.

## Interpretation

**a. Pathogenicity:** The model learned essentially nothing about it. Test accuracy (47-61 %) falls inside the shuffled-label null (26-78 %). Leave-one-genus-out accuracy sits at or below chance, and AUC within held-out pairs averages 0.50. The large gap between training accuracy (71-100 %) and test accuracy means the model memorised the training genomes. With only 18 training genomes, there are really 18 labelled examples, not 7200. Pathogenicity is mostly a matter of gene content: toxins, secretion systems and pathogenicity islands. Those make up a few percent of a genome, so most random 1-kb fragments from a pathogen are housekeeping DNA with nothing pathogen-specific in them.

**b. The top-weighted k-mers:** They are genome-signature features, not virulence motifs.
- **GC extremes:** GCGC, CGCG and CGCGGC fit the GC-rich pathogens (M. tuberculosis, P. aeruginosa). TAAA, TTTA, ATA/TAT and TTTTAA fit the AT-rich pathogens (S. aureus, B. anthracis, S. pyogenes). Pathogens sit at both ends of the GC range, so one linear GC term fails. That explains both the weak coefficient-to-GC correlation and why a GC-only model gets 52 %.
- **Lineage-specific GC-rich words:** GGGC, CCGT, ACGG and TCCG are negative, which most likely reflects S. coelicolor and S. meliloti, the GC-rich non-pathogens. Their signature is being told apart from that of M. tuberculosis.
- **Restriction sites:** CAGCTG (PvuII), GAGCTC (SacI) and CTTAAG (AflII) are palindromic restriction sites. Each lineage avoids such sites to a different degree depending on its restriction-modification systems. CTAG is the classic avoided tetranucleotide, linked to restriction systems, very-short-patch repair and TAG stop codons.
- **Codon usage and strand:** GGT and ACC are favoured codons in some genomes. GGA, TCC and GGG are partly Shine-Dalgarno (AGGAGG) and strand-skew effects, which show up because only one strand is read.

**c. Grammar or superficial features:** Superficial features. Bag-of-words and mean-pooled Word2Vec throw away word order beyond k, so no grammar is possible. Two results show the features capture genome identity and relatedness:
- They identify the source genome with 74-81 % accuracy.
- Nearest neighbours share a label less often than chance (35-40 %). A fragment's closest relative is usually its matched relative from the same genus, which has the opposite label.

Regressing out GC changes nothing because there was no real signal to remove. The higher-order composition (genome signature) that remains still encodes ancestry, not phenotype.

## Checkable claims

1. Predictions on the test set split by genome. Within each test genome, at least 80 % of fragments get the same predicted label, so genome-level accuracy is about 3 out of 6.
2. For most fragments, the nearest neighbour from another genome comes from the same genus, usually the matched relative. That accounts for label agreement below 50 %.
3. Each of the strongest coefficients can be traced to one or two training genomes. GCGC and CGCG are most enriched in P. aeruginosa and M. tuberculosis. GGGC, CCGT and ACGG are most enriched in S. coelicolor. TAAA, TTTA and TTTTAA are most enriched in S. aureus. Dropping those genomes from training should reshuffle the top-coefficient list.
4. CTAG's observed/expected ratio (expected from trinucleotide frequencies) is below 0.5 in most genomes. It varies between genomes, but pathogens and non-pathogens do not differ significantly in it (genome-level test, n = 24).
5. Restriction palindromes (CAGCTG, GAGCTC, CTTAAG, CGCG) have observed/expected ratios that vary more between genera than between the two members of a matched pair.
6. A word's coefficient and its reverse complement's coefficient (GGA vs TCC, GGG vs CCC) can differ in sign or size. Counting canonical k-mers, which merges each word with its reverse complement, will not change test accuracy.
7. Adding more fragments per genome will raise genome-ID accuracy but leave pathogen test accuracy within the shuffled-label null.
8. Fragments found only in the pathogen of a pair (for example, O157:H7 sequence with no BLAST hit in K-12) will be enriched in islands with unusual GC. These are a small minority of pathogen fragments, and they are where any real pathogenicity signal would be found.
