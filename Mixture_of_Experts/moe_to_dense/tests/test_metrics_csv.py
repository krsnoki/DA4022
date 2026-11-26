import csv

from Mixture_of_Experts.moe_to_dense.scripts.reproduce import write_metrics_csv


def test_csv_matches_results(tmp_path):
    metrics = dict(parameters=10, latency_ms_median=1.25, throughput_tokens_per_second=42, peak_memory_mib=12.5, peak_memory_scope='process peak')
    results = dict(seed=42, hardware={'device': 'CPU'}, software={'torch': 'test'}, data={'batch_size': 2, 'sequence_length': 4}, teacher=dict(metrics, perplexity=2.5), students={'sf': dict(metrics, post_distillation_perplexity=3.5)})
    path = tmp_path / 'metrics.csv'
    write_metrics_csv(results, path)
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [float(row['quality_value']) for row in rows] == [2.5, 3.5]
    assert all(float(row['peak_memory_mib']) == 12.5 for row in rows)
