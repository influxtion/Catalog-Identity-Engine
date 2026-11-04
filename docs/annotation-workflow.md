# Evidence annotation

The 145-case training audit is ready for two independent human reviews per case.
Use **Evidence annotation** in the dashboard, or open `/annotate`. An adjudicator
can see a case only after both reviewers submit. The private audit key, benchmark
labels, entity IDs, error strata and model scores never enter the review service.

This is diagnostic data collection. It neither changes official labels nor trains
a matcher. Family and exact-variant judgments remain separate from benchmark identity.
The original audit has no human judgments; browser tests use original fictional lamps.

## Prepare a batch

Install the `server` extra from the project setup. Use the same `DATABASE_URL` for
the CLI and API; the default is `sqlite:///catalog.db`. PostgreSQL is also supported.
Import the **blind** file, not `review-key.jsonl`:

```sh
uv run catalog-identity annotation-import --packet runs/training-audit-v1/review-blind.jsonl --audit-report runs/training-audit-v1/audit.json --name "Training evidence review" --reviewers reviewer-1 reviewer-2 --adjudicator adjudicator --output runs/annotation-access-v1
```

Each output directory must be new. On this Windows checkout, commands can also run
as `.venv\Scripts\python.exe -m catalog_identity.cli ...` without activating a shell.

The import creates 290 independent assignments and an adjudication assignment for
each of the 145 cases. It writes a batch ID and three separate access keys to
`runs/annotation-access-v1/access-keys.json`. The database stores key hashes only;
keys never appear in command output. Give each person only their own key. The names
are assignment identifiers; distinct keys alone do not establish independent people.
Use two people who have not inspected the private audit key, with a third person
for adjudication. An analyst who has already seen the labels is not a blind reviewer.

Start the API and dashboard using the README instructions; no worker is needed for
annotations. Open <http://localhost:3000/annotate> and enter the assigned key. The browser
remembers it in that tab's session storage. Sign out before another reviewer uses it.
Annotation responses are not cached. These keys protect annotation assignments; other
catalog routes still lack authentication. Keep the developer service local or within
a trusted deployment, and use HTTPS when connecting reviewers remotely.

Imports reject duplicate packets, prefilled judgments and unexpected listing metadata.
Packet and case checksums bind every saved judgment to its observed inputs. The batch
records guideline version `identity-evidence-v1`, source audit checksum, dataset version,
training-input hash and sampling seed when an audit report is supplied. New audits include
packet/key checksums. The original audit predates those checksums: its association is
explicitly recorded as `operator_supplied_legacy_report`, with a count check and a newly
computed packet checksum, rather than claiming an original cryptographic binding.

## Review guidelines: identity-evidence-v1

Read both complete observations before choosing any decision. Missing fields are shown
explicitly. Descriptions retain their raw text; no image or external page is downloaded.

| Decision | Meaning |
| --- | --- |
| Input sufficiency | `sufficient`: the supplied observations support an assessment; `insufficient`: distinguishing evidence is absent; `conflicting`: supplied observations conflict; `needs_source_check`: a permitted external source must be checked. Explain which evidence is missing or conflicting. |
| Benchmark identity | Your evidence-based judgment of the same/different benchmark product entity. WDC labels originate from identifier-based clusters; this judgment does not reconstruct a hidden gold label or substitute family/variant truth. Use `uncertain` when the observations do not establish the relation. |
| Product family | Same underlying product family, allowing applicable configuration differences; a shared series or brand alone is insufficient. |
| Exact variant | Same configuration, including the category's relevant size, color, capacity, pack/bundle, generation, region or compatibility distinctions. Missing defining evidence should remain uncertain. |

Same exact variant requires same family; different families require different variants.
No rule maps a benchmark identity judgment into either hierarchy judgment. All three
identity fields may be uncertain. Do not guess missing identifiers or balance labels.

For observed evidence, select a listing, field and role, then copy an exact quote and
explain what it establishes. Cite both sides when the decision depends on a comparison.
Roles distinguish **own model/MPN**, **series**, **compatible device**, **measurement**,
**variant/configuration**, **product type**, **manufacturer**, and **other observation**.
Record pack/bundle/generation distinctions as variant/configuration evidence. No role is
preselected. A code that merely appears in a title is not automatically the product's MPN.

When a quote repeats, choose its occurrence using the displayed context or include more
context in the quote. The server validates the exact substring and stores a zero-based
Unicode-code-point `start` offset. The exclusive end is `start + len(quote)` in Python.
Offsets refer to the original field, not normalized text or UTF-16 indices. Two occurrences
of the same code can have different roles. Do not shorten a quote so far that its meaning
is lost; quoted spans are evidence, not automatically complete token-level training labels.

Add a source check only after actually inspecting a permitted source. Record its URL,
observation date and a note explaining the finding and its provenance. Distinguish
authoritative identifiers from merchant claims and current pages from historical offers.
The system validates URL/date structure, not the source's truth or licensing. Do not
derive attributes from gold IDs or predictions. Do not collect images for this workflow.

Write reasoning that names remaining uncertainty and the evidence needed to resolve it.
Definite identity judgments require a quote or recorded source check. A fully uncertain
judgment can be submitted without fabricated evidence. Save drafts as needed; incomplete
evidence/source rows must be completed or removed before saving. A second tab cannot
silently overwrite a newer draft: reload after a revision conflict.

**Submit and lock** makes that judgment immutable. Confirm only after checking the
evidence and all four decisions. Reviewers cannot see each other's drafts, submissions
or agreement status. Submitted cases remain available through the Submitted filter.
If a locked judgment was mistaken, preserve it and document the correction in a new,
separately versioned annotation round; do not edit database rows or rewrite exports.

## Adjudicate

Use the third access key and refresh the queue. Only cases with two locked reviews
appear. A difference in any decision, or conflicting roles for the same quoted occurrence,
is marked **Disagreement**. The adjudicator sees both original submissions, including
their sources, and records a separate evidence-supported resolution. Agreement cases
are also available for spot checks. Different supporting quotes or prose alone do not
trigger disagreement, so agreement is not proof that every evidence claim is correct.

An adjudicator may preserve uncertainty. Cases with matching decisions/roles count as
`agreed`; disputed cases remain `needs_adjudication` until the third submission is locked.
The originals and revision history remain intact after adjudication.

## Export progress and evidence

Use the batch ID from the access file:

```sh
uv run catalog-identity annotation-export --batch BATCH_ID --output runs/annotation-progress-v1
```

`summary.json` reports case status, paired-review completion, original disagreement rate
and resolved sufficiency counts. `annotations.json` includes observations, checksums,
locked submissions, resolved decisions and an append-only event history. Drafts appear
only in that operator history, not as completed submissions. Credentials are excluded.
All files remain private and ignored; product text has not been cleared for redistribution.

After every case is agreed or adjudicated, the operator can join the original private
key **offline** to identify data needs by the sampled categories:

```sh
uv run catalog-identity annotation-export --batch BATCH_ID --review-key runs/training-audit-v1/review-key.jsonl --output runs/annotation-analysis-v1
```

`analysis.json` adds disagreement, insufficient/conflicting/source-needed evidence,
uncertain identity counts, recorded source checks and evidence-role counts by primary
stratum and overlapping tags. Primary strata partition the sample; overlapping tags
do not. It compares only the new benchmark-identity judgment with the benchmark label,
preserving uncertain judgments. It never compares family/variant decisions to that label.
A label difference is a review finding, not proof that the benchmark is wrong. The key
must match exactly the batch's case IDs, and its original checksum when available.
There is no API or reviewer-screen endpoint for this analysis or the private key.

Use these counts and the written missing-evidence reasons to prioritize verified code
roles/aliases, near-neighbor specifications and low-overlap identity links. Every exported
case has `automatically_training_eligible: false`; review, rights and split design still
precede training. This targeted 145-case packet does not measure production accuracy or
provide fresh confirmation data. Follow the [data plan](../experiments/training-data-findings.md)
for the entity-disjoint pilot and the next preregistered comparison.

## Recover access

Reimporting the same packet does not reset reviews. Replace a lost or exposed key:

```sh
uv run catalog-identity annotation-rotate --batch BATCH_ID --reviewer reviewer-1 --output runs/reviewer-1-replacement
```

The replacement file contains only that member's new key. The old key stops working;
assignments and submitted judgments remain unchanged. Back up the database alongside
private exports. Access files, database files and local listing data must stay out of Git.

## Verification

Python tests exercise blinding, separate batches, stale/concurrent saves, immutable
submissions, role conflicts, source validation, Unicode spans, key rotation, audit binding
and offline analysis. PostgreSQL runs the concurrency path in CI. Browser tests cover the
complete two-reviewer/adjudicator flow, saved drafts, repeated code locations and mobile
layout using fictional inputs. No automated test submits judgments for the real 145 cases.
