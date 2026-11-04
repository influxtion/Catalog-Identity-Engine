# Frozen selection before official-validation replay

The fixed train-only screening completed all twelve fits from clean commit
`5dac477`. [Full aggregate evidence](matcher-v4-screening.json) records every fold,
all four candidates, effective parameters, train checksum, source hashes and environment.

| Candidate | Mean weighted AP | Mean weighted recall at 99% empirical precision | Supported folds |
| --- | ---: | ---: | ---: |
| v3, 15 leaves / 200 trees | 0.91120 | 22.78% | 3 |
| v4, 15 leaves / 200 trees | 0.91888 | 17.94% | 3 |
| v3, 31 leaves / 400 trees | 0.90224 | 20.10% | 3 |
| v4, 31 leaves / 400 trees | 0.91608 | 7.95% | 3 |

The preregistered rule selects **v3-small, the control**. The richer feature sets
improved mean ranking AP but lost recall at the target precision. Both v4 candidates
improved the 99% envelope in fold 0 only and declined in folds 1 and 2; the larger
v3 model declined in all three. Supported folds are not evidence of improvement.

Selection is now frozen. Because selected and control are identical, one full-train
fit and one official-validation replay will serve both roles, avoiding a redundant
run. No v4 official-validation comparison will be added after this result. This
does not establish 99% precision; all screening thresholds were selected on their
reported assessment fold. No production/default promotion and no test evaluation.
