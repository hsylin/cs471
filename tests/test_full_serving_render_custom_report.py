import csv
import json

from full_serving import custom_report
from full_serving import render_custom_report as render
from test_full_serving_custom_report import write_experiment


def test_sixty_rows_preserve_blocks_and_generation_warning(tmp_path, monkeypatch):
    root, _ = write_experiment(tmp_path, monkeypatch)
    report = custom_report.build_report(root)
    report["datasets"][0]["blocks"][0]["contrasts"]["custom/off"]["equal_generated_texts"] = 0
    rows = render.table_rows(report)
    assert len(rows) == 60
    assert {r["dataset"] for r in rows} == {"wildchat", "sharegpt", "instructcoder", "numinamath", "longbench"}
    custom = next(r for r in rows if r["dataset"] == "wildchat" and r["arm"] == "custom")
    assert custom["custom_off_equal_generated_texts"] == 0
    assert custom["custom_over_off_mean_response"] == .9
    assert custom["normal_async_commits"] is None
    html = render.render_html(report, rows, "a" * 64)
    assert "Protocol complete; this does not declare superiority" in html
    assert "Generated outputs can differ" in html and "not isolated GPU service" in html
    assert "No datasets are pooled" in html and "four blocks" in html


def test_bad_triple_withholds_timing_and_interval_even_if_stale_summary_exists(tmp_path, monkeypatch):
    root, _ = write_experiment(tmp_path, monkeypatch)
    report = custom_report.build_report(root)
    report["complete"] = False
    report["datasets"][0]["complete"] = False
    bad = report["datasets"][0]["blocks"][0]
    bad["valid"] = False
    bad["issues"].append("runtime evidence <missing>")
    rows = render.table_rows(report)
    for r in rows[:3]:
        assert r["mean_response_ms"] is None
        assert r["custom_over_off_mean_response"] is None
        assert r["completed_requests"] == 16
    html = render.render_html(report, rows, "h")
    assert "Incomplete or invalid" in html and "runtime evidence &lt;missing&gt;" in html
    first = html.split("<h2>wildchat</h2>")[1].split("<h2>sharegpt</h2>")[0]
    assert "<td>withheld</td>" in first


def test_cli_renders_public_snapshot_csv_and_html(tmp_path, monkeypatch):
    root, _ = write_experiment(tmp_path, monkeypatch)
    source = tmp_path / "report.json"
    source.write_text(json.dumps(custom_report.build_report(root)))
    prefix = tmp_path / "outputs" / "compare"
    assert render.main([str(source), "--output-prefix", str(prefix)]) == 0
    with prefix.with_suffix(".csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60 and rows[0]["normal_async_commits"] == ""
    assert "Public report SHA-256" in prefix.with_suffix(".html").read_text()
