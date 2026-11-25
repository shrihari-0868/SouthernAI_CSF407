# Task 6: AI-generated explanations vs. verified results

## What was done
1. **One prompt for every AI** ([prompt.md](prompt.md)). It holds a factual summary of the Task 1-5 results: the
   setup, the accuracies, the controls, and the top k-mer coefficients. It gives no interpretation. Each system is
   asked (a) what the model learned, (b) why those k-mers get large weights, (c) grammar or shortcuts, and
   (d) 5-8 checkable claims.
2. **Three AI systems**, each run in a fresh context with no tools and no access to the data or code. The answers
   are stored unedited in [responses/](responses):
   - A: Claude Haiku (small, fast model)
   - B: Claude Sonnet (mid-size model)
   - C: Claude Opus (largest model)

   All three are Claude models from different size tiers. To add a non-Claude system (ChatGPT, Gemini, etc.),
   paste `prompt.md` into it, save the answer as `responses/D_<name>.md`, and add its claims to the table below
   (most map onto tests that already exist).
3. **Independent verification** ([verify_claims.py](verify_claims.py)). Every claim is labelled as one of:
   - **RESTATEMENT**: it only repeats a number from the prompt. It is checked against the Task 4/5 CSVs but not
     counted as insight.
   - **NEW**: the prompt did not contain it. Each NEW claim gets its own computation on the raw data.

```
python verify_claims.py ../data .. results
```
Outputs: `results/task6_claim_verification.csv` (one row per claim: test, numbers, verdict) and `results/detail_*.csv`.

## Verification results

| ID | AI | Claim (short) | Test | Result | Verdict |
|---|---|---|---|---|---|
| R1 | A, B | Real test accuracy is within the shuffled-label null | 200 Task 4 null runs | best 61.3 %, null p = 0.12 | Supported (restatement) |
| R2/R3 | A, B | Within-pair AUC ~0.5; unseen-pair Spearman ~0; removing GC changes nothing | Task 4/5 CSVs | identical to the prompt | Circular |
| B1 | B | Genome-grouped CV gives 45-55 % on every representation | balanced LOGO CV file | 42-53 %; 4 of 18 rows below 45 % | Partly |
| **B-f** | B | AT-rich weights come from "H. pylori, S. aureus and **Listeria in the training set**" | split column | H. pylori and Listeria are **test** genomes | **Factual error** |
| **C3** | C | GCGC/CGCG come from P. aeruginosa and M. tb; GGGC/CCGT/ACGG from S. coelicolor; TAAA/TTTA/TTTTAA from S. aureus | rank the 18 training genomes by the word's frequency | 8/8 words: a named genome is in the top 2 (S. aureus is #1 for all three AT words; S. coelicolor is #1 for all three GC words) | **Supported** |
| B4 | B, C | Dropping one training genome reshuffles the top-20 coefficients (overlap well under 50 %) | 18 refits, each leaving one genome out | median overlap 70 %, minimum 55 % | **Not supported**: the word ranking is fairly stable |
| C1 | C, B | ≥80 % of a test genome's fragments get the same label; about 3/6 genomes right | k=4 model on the 6 test genomes | majority share 54-89 % (only 1 of 6 at ≥80 %); genome vote 4/6. **Both members of each test pair got the same call** (both Listeria → pathogen, both Corynebacterium → non-pathogen) | Partly |
| C2 | C | A fragment's nearest neighbour from another genome is usually its genus partner, which explains purity below 50 % | 4-mer space, 1200 fragments from paired genomes | genus partner 43 % (chance 4 %); label agreement 28 % | Partly: a large effect, but "most" is too strong |
| B2/A6 | A, B | Genome-ID errors go to similar-GC or same-genus genomes | 18-way genome-ID model, errors vs random | median GC gap 1.7 pts (random 11.7); 41 % of errors go to the genus partner | **Supported** |
| C4/A7 | A, B, C | CTAG is avoided in most genomes (obs/exp < 0.5) and does not separate the labels | 2nd-order Markov obs/exp per genome | only 7/24 genomes below 0.5 (E. coli ×2, Salmonella, P. aeruginosa, S. coelicolor, S. meliloti, P. putida); median 0.84; no label difference (p = 0.58) | Partly: no class signal is right, **"universal avoidance" is wrong**. CTAG depletion is lineage-specific (proteobacteria and high-GC actinomycetes); low-GC Firmicutes are at about 1.0 |
| C5/B5 | B, C | Restriction palindromes (CAGCTG, GAGCTC, CTTAAG, CGCG) vary by genus, not by label | within-pair vs between-genus |Δ obs/exp|; Mann-Whitney by label | within-pair differences 2-6× smaller than between genera for all 4 words; label p = 0.16-0.71 | **Supported** |
| C6 | C | A word and its reverse complement can get different coefficients; canonical k-mers do not change accuracy | 120 RC word pairs; canonical 136-word refit | Spearman 0.66, opposite sign in 23 % of pairs; test accuracy 0.579 → 0.578 | **Supported** |
| C-gc | C | Pathogens sit at both GC extremes, so one linear GC term fails | distance of genome GC from the median, by class | pathogens 0.33-0.67, non-pathogens 0.35-0.72; no difference in extremeness (p = 0.62) | Partly: true for pathogens, but non-pathogens span the same range, so this does not explain the result |
| C8 | C (B8) | Pathogen-only DNA (O157:H7 vs K-12) is a minority, has unusual GC, and is where any signal lives | E. coli O157 fragments: share of 16-mers found in K-12 | 18 % of fragments are O157-specific; their GC deviates more (6.4 vs 3.7 pts); **but the model scores them as less pathogen-like** (0.62 vs 0.78) | Partly: minority and atypical-GC are right; the k-mer model does not use them |
| — | B8, C7 | Gene-content features would be needed; more fragments would only raise genome-ID accuracy | — | not tested (needs new data or annotation) | Untested |

## How the three systems compare

| | A: Haiku | B: Sonnet | C: Opus |
|---|---|---|---|
| Overall reading ("learned genome identity, not pathogenicity") | correct | correct | correct |
| Numbered claims that only restate the prompt | 6 of 8 | 3 of 8 (B1, B3, B6) | 0 of 8 |
| Numbered claims that are new and were tested | 2 (A6, A7) | 4 (B2, B4, B5, B7) | 7 (C1-C6, C8; C7 untested) |
| Of those: supported / partly / not supported | 1 / 1 / 0 | 2 / 1 / 1 | 3 / 4 / 0 |

Some claims were made by two systems and share a test row, so these columns overlap:
there are **10 distinct new claims** in total (4 supported, 5 partly, 1 not supported), plus the
one factual error below.
| Claims in the explanation text that were checked | — | 1 factual error (B-f); "CTAG universal" is wrong (C4) | GC-extremes story, partly (C-gc) |
| Biological story offered | codon bias, strand skew (vague) | restriction sites, CTAG "universal" | genome signature by lineage, restriction sites, Shine-Dalgarno, pathogenicity islands |

### Takeaways
- **All three AIs agree on the headline**, but the prompt already contained nearly every number needed to reach
  it. Agreement among the AIs is therefore weak evidence. The value lies in the new claims.
- **Larger model, more specific and checkable claims.** Haiku mostly paraphrased the prompt. Opus made seven
  concrete, falsifiable predictions that could be tested here, and none of them was flatly wrong, although four
  held only in part.
- **Correct confident specifics:** Opus traced each top k-mer to the named training genomes (8/8). The
  palindrome and genus-signature account held up, and so did the reverse-complement observation.
- **Confident errors:**
  - Sonnet placed two test genomes in the training set. A plausible-sounding explanation was built on a false
    fact about our own data.
  - Sonnet and Opus both repeated the textbook line that CTAG is universally avoided. The data say it is
    lineage-specific.
  - "Dropping one genome reshuffles the coefficient list" was wrong: the ranking is fairly stable.
- **The most interesting partial result (C8):** the pathogen-only DNA in E. coli O157:H7 really is a minority
  with unusual GC. The k-mer model gives it a *lower* pathogen score than the shared backbone. Whatever the
  model learned, it is not the islands.
- **Lesson:** AI explanations are useful as hypothesis generators, not as conclusions. About half of the new
  claims survived as stated, and the errors sounded just as confident as the correct claims.
