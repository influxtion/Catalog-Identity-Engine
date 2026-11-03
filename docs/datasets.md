# Dataset contracts

Reviewed 2026-11-03. Dataset metadata and gold labels stay outside model features.

## WDC Products

Use the [official benchmark page](https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html)
and its [pairwise schema example](https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/sample_pairwise.json).

Gzipped JSONL pair files have offer fields suffixed `_left` and `_right`:
`id`, `title`, `description`, `brand`, `price`, `priceCurrency`, `cluster_id`.
Pair metadata includes `pair_id`, binary `label`, and `is_hard_negative`.
The adapter retains cluster IDs as evaluation metadata only. It does not fabricate
store IDs, GTINs, MPNs, categories, or hierarchical labels from these fields.

Labels mean the same benchmark entity (1) or different entities (0). They originate
from identifier-based clusters; they are not independent annotations for family and
exact-variant identity. Use a separate curated dataset for that distinction.

Keep each official train/validation/test file intact. Offers are disjoint across
splits, but product entities may overlap according to the selected unseen setting.
Record corner-case percentage, development size, and unseen percentage in the run
configuration. Claims about unseen entities require the appropriate test protocol
and an entity-overlap audit. Never randomly split the pair rows.

## Larger corpus

The [WDC Product Data Corpus v2](https://webdatacommons.org/largescaleproductcorpus/v2/index.html)
has its own schema, derived from older crawl data. It is not WDC Products. Unlabeled
corpus experiments can measure retrieval cost and throughput, not matching accuracy.
No corpus adapter or scale result is claimed by this initial baseline.

## Licensing and redistribution

The reviewed WDC Products download page does not state an explicit dataset license.
The [WDC homepage](https://webdatacommons.org/) licenses its extraction **software**;
that statement does not establish a license for product content. Redistribution
permission for this benchmark remains unverified. Obtain and review the applicable
terms before redistributing data. This repository ships no WDC rows or images and
does not automatically download them. Locally supplied files belong under ignored
`data/`; reports derived from them remain under ignored `runs/`.

## Synthetic fixtures

Original, fictional listings are used for deterministic tests and the offline demo.
Their metrics test wiring only and are never evidence of real benchmark quality.
The synthetic generator uses disjoint entity groups across splits. It does not
pretend to reproduce the difficulty, noise, or distribution of real product data.
