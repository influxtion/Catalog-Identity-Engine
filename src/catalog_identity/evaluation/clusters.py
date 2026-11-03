from collections import Counter


def cluster_metrics(predicted, truth):
    """Contingency-count metrics without enumerating all positive pairs."""
    flat = [key for members in predicted for key in members]
    if len(flat) != len(set(flat)) or set(flat) != set(truth) or not flat:
        raise ValueError("predicted clusters must partition exactly the labeled population")
    gold_counts = Counter(truth.values())
    predicted_pairs = true_pairs = 0
    bc_precision = bc_recall = purity = 0.0
    erroneous_merges = 0
    fragments = Counter()
    for members in predicted:
        counts = Counter(truth[key] for key in members)
        n = len(members)
        predicted_pairs += n * (n - 1) // 2
        true_pairs += sum(c * (c - 1) // 2 for c in counts.values())
        bc_precision += sum(c * c / n for c in counts.values())
        bc_recall += sum(c * c / gold_counts[entity] for entity, c in counts.items())
        purity += max(counts.values())
        erroneous_merges += int(len(counts) > 1)
        fragments.update(counts.keys())
    gold_pairs = sum(c * (c - 1) // 2 for c in gold_counts.values())
    return {
        "pair_precision": true_pairs / predicted_pairs if predicted_pairs else None,
        "pair_recall": true_pairs / gold_pairs if gold_pairs else None,
        "pair_f1": 2 * true_pairs / (predicted_pairs + gold_pairs)
        if predicted_pairs + gold_pairs
        else None,
        "b_cubed_precision": bc_precision / len(flat),
        "b_cubed_recall": bc_recall / len(flat),
        "purity": purity / len(flat),
        "erroneous_merges": erroneous_merges,
        "fragmentation": sum(max(0, count - 1) for count in fragments.values()),
    }
