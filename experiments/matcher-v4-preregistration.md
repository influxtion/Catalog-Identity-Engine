# Matcher v4: fixed train-only screening protocol

Registered before running the experiment. Development only; no production promotion.

The four candidates, in tie-breaking order, are:

1. v3, LightGBM, 15 leaves, 200 trees (control).
2. v4, LightGBM, 15 leaves, 200 trees.
3. v3, LightGBM, 31 leaves, 400 trees.
4. v4, LightGBM, 31 leaves, 400 trees.

All use seed 42, learning rate .05, minimum child samples 20, L2 penalty 1,
one CPU thread, deterministic column-wise fitting, and all official train pairs.
The experiment uses exactly three entity folds assigned by SHA256(seed:entity)
modulo 3. Each fit uses only listings outside its assessment fold and pairs with
both endpoints outside; assessment uses pairs with both endpoints inside.
Boundary pairs are excluded and counted. Vocabularies are fitted separately on
each training fold. Validation and test paths are neither read nor hashed.

Assessment positives have weight 1 and negatives weight 3: across three folds,
same-entity positives are retained once and different-entity negatives have
expected retention 1/3. These weights estimate the original training-pair mixture;
they do not make this a random sample of all possible product pairs.

Report each fold's average precision and maximum weighted recall at empirical
precision .95, .98, and .99, with raw and weighted counts, distinct positive
entities, and at least five predicted pairs for a qualifying envelope. Scores
from separately fitted models are never pooled to select a common threshold.
Envelopes select thresholds on the reported assessment population and are
optimistic diagnostics, not out-of-sample operating policies or precision guarantees.

Selection: candidates must have a qualifying .99 envelope in at least two folds
and mean weighted AP at least 95% of control mean weighted AP. Among eligible
candidates maximize mean fold weighted recall at .99 (missing envelope = zero),
then number of folds with support, then mean weighted AP, then matrix order.
If none qualifies, retain the control. Do not add candidates or tune features
after inspecting screening outcomes.

Fit full-train control and the preselected candidate and run the existing
entity-partitioned validation calibration/policy workflow once for each, followed
by one full official-validation replay each. Official train and validation share
entities: this is seen-entity development evidence even though offers are disjoint.
The validation replay includes selection/calibration data and is not independent.
No test set is opened; historical test results cannot be used for selection.
Publish all four outcomes, including unsuccessful candidates. Stop after this
comparison and retain current default v2 and hierarchical merge behavior.
