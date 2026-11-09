#!/usr/bin/env python3
"""Merge the three reviewed inventories without modifying their evidence.

Run from any working directory. Output paths are relative to this script's
repository, not to the original review machine. Canonical fields copy existing
aliases; all original row fields and inventory metadata remain in the JSON.
"""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("moe", "serving", "architecture")
ALIASES = {
    "controlled_variables": ("controlled_variables", "controls"),
    "reported_results": ("reported_results", "results"),
    "limitations": ("limitations", "limits"),
    "artifact_availability": ("artifact_availability", "artifact.inspection"),
    "fulltext_sha256": ("fulltext_sha256", "retrieval.pdf_sha256"),
}
LOCATOR_ALIASES = ("specific_methods_table_page", "support_section_table_page", "supporting_sections")
REQUIRED = tuple(ALIASES) + ("supporting_evidence", "methods_evaluation_read_basis")


def value_at(row, dotted):
    value = row
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def resolve(row, aliases):
    for alias in aliases:
        value = value_at(row, alias)
        if value is not None and value != "" and value != []:
            return value, alias
    raise ValueError(f"{row.get('title')}: none of the existing aliases {aliases} supplies evidence")


def normalize(row, group):
    merged = {"group": group, **row}
    provenance = {}
    for canonical, aliases in ALIASES.items():
        value, alias = resolve(row, aliases)
        merged[canonical] = value
        provenance[canonical] = alias
    locators, alias = resolve(row, LOCATOR_ALIASES)
    if isinstance(locators, str):
        locators = [locators]
    if not isinstance(locators, list) or not all(isinstance(x, str) and x for x in locators):
        raise ValueError(f"{row['title']}: unsupported locator representation")
    merged["supporting_evidence"] = {"paper_url": row["paper_url"], "locators": locators}
    provenance["supporting_evidence"] = {"paper_url": "paper_url", "locators": alias}
    declarations = {}
    if "methods_read" in row:
        declarations["methods_read"] = row["methods_read"]
    evaluation, evaluation_alias = None, None
    for field in ("evaluation_read", "eval_read"):
        if field in row:
            evaluation, evaluation_alias = row[field], field
            declarations["evaluation_read"] = evaluation
            break
    narrative = row.get("evidence_of_methods_eval_read")
    if not declarations and not narrative:
        raise ValueError(f"{row['title']}: no existing methods/evaluation reading basis")
    # Absence of a boolean is not converted to false, or inferred from prose.
    basis = {"declarations": declarations, "fulltext_version": row["fulltext_version"], "locators": locators}
    if narrative:
        basis["narrative"] = narrative
    merged["methods_evaluation_read_basis"] = basis
    provenance["methods_evaluation_read_basis"] = [
        field for field in ("methods_read" if "methods_read" in row else None,
                            evaluation_alias, "evidence_of_methods_eval_read" if narrative else None,
                            "fulltext_version", alias) if field is not None]
    merged["canonical_field_sources"] = provenance
    if not re.fullmatch(r"[0-9a-f]{64}", merged["fulltext_sha256"]):
        raise ValueError(f"{row['title']}: invalid fulltext SHA-256")
    for key, value in row.items():
        if merged.get(key) != value:
            raise ValueError(f"{row['title']}: original field changed: {key}")
    return merged


LABELS = [
    ("Hypothesis", "hypothesis"), ("Mechanism", "mechanism"),
    ("Model and hardware", "model_hardware"), ("Data and trace construction", "datasets_trace_construction"),
    ("Baselines", "baselines"), ("Controlled variables", "controlled_variables"),
    ("Experimental unit", "experimental_unit"), ("Metrics and timing boundary", "metrics_timing"),
    ("Reported results", "reported_results"), ("Uncertainty", "uncertainty"),
    ("Ablations", "ablations"), ("Limitations", "limitations"),
    ("Artifact and inspection", "artifact_availability"),
    ("Methods / table / figure anchors", "supporting_evidence"),
    ("Methods / evaluation reading basis", "methods_evaluation_read_basis"),
    ("Applicability", "applicability"), ("Retrieved fulltext version", "fulltext_version"),
    ("Retrieved fulltext SHA-256", "fulltext_sha256"),
]


def display_value(row, key):
    if key == "applicability":
        return resolve(row, ("relevance_to_ours", "relevance"))[0]
    return row[key]


def render_value(row, key):
    value = display_value(row, key)
    if key == "supporting_evidence":
        # Printed PDF pages and file-page indices can differ. Do not synthesize
        # #page fragments from section references or ambiguous printed pages.
        url = html.escape(value["paper_url"], quote=True)
        return "<br>".join(f'<a href="{url}">{html.escape(locator)}</a>' for locator in value["locators"])
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    rendered = html.escape(value)
    if key == "artifact_availability":
        artifact_url = row.get("artifact_url") or value_at(row, "artifact.url")
        if artifact_url:
            rendered += '<br><a href="' + html.escape(artifact_url, quote=True) + '">Artifact source</a>'
    return rendered


def build(repo=ROOT):
    directory = Path(repo) / "docs/generalization"
    allrows, inventories, originals = [], {}, {}
    for group in GROUPS:
        path = directory / f"literature-{group}.json"
        raw = path.read_bytes()
        originals[path] = raw
        inventory = json.loads(raw)
        inventories[group] = dict(path=path.name, sha256=hashlib.sha256(raw).hexdigest(),
                                  metadata={k: v for k, v in inventory.items() if k != "papers"})
        for row in inventory["papers"]:
            if not row.get("proceedings_url") or not row.get("paper_url"):
                raise ValueError("Missing proceedings/paper URL")
            allrows.append(normalize(row, group))
    if len(allrows) != 50 or len({r["title"].casefold() for r in allrows}) != 50:
        raise ValueError("Expected exactly 50 unique conference-paper titles")
    if Counter(r["group"] for r in allrows) != Counter(moe=18, serving=17, architecture=15):
        raise ValueError("Unexpected inventory allocation")
    for row in allrows:
        if any(key not in row or row[key] is None for key in REQUIRED):
            raise ValueError("Missing canonical evidence field: " + row["title"])
        for _, key in LABELS:
            display_value(row, key)  # Fail on missing existing evidence; no generic fallback.
    out = dict(schema="cs471_generalization_evidence_matrix_v2", conference_papers=50,
               scope="18 MoE systems, 17 serving, 15 architecture/workload/methodology papers. Conference identity verified separately from accessible fulltext version; some fulltexts are author manuscripts. Preprints/access-limited candidates are supplementary and excluded from 50.",
               canonicalization="Original fields are retained. Added canonical aliases copy existing extraction values only; canonical_field_sources identifies their origin. Reading declarations are not inferred where only narrative evidence exists.",
               inventories=inventories, papers=allrows)
    (directory / "evidence-matrix.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    esc = html.escape
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>cs471: 50-paper evidence matrix</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#111;background:white}input,select{font:inherit;padding:.5rem;margin-right:.5rem;max-width:85%}details{border:1px solid #bbb;padding:.8rem;margin:.7rem 0}summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;width:100%;margin-top:1rem}th,td{border:1px solid #ccc;padding:.5rem;text-align:left;vertical-align:top}th{width:23%;background:#f3f3f3}td{white-space:pre-wrap;overflow-wrap:anywhere}.meta{font-size:.9rem;color:#444}.hidden{display:none}@media print{input,select,button{display:none}details{break-inside:avoid}}</style><h1>Evidence for the next cs471 experiment</h1><p>50 conference papers. Publication identity and retrieved manuscript version are separate fields. Supplementary preprints, inaccessible candidates and company articles are excluded from this count. The matrix records reported evidence; it does not treat paper speedups as replications of our experiment.</p><label>Search <input id="query" type="search" placeholder="routing, padding, arrivals, title…"></label><label>Area <select id="area"><option value="">All</option><option>moe</option><option>serving</option><option>architecture</option></select></label><p id="count">50 papers</p>']
    md = ["# Conference evidence matrix", "", out["scope"], "",
          "The full extraction is in [evidence-matrix.json](evidence-matrix.json); the [searchable matrix](evidence-matrix.html) exposes every requested methods/evaluation field. Canonical field aliases and their origins are recorded without changing the source inventories. Supporting locators link to the retrieved paper; no PDF page number is guessed. Reported results retain their original baselines and are not pooled into a speedup estimate.", "",
          "| Paper | Venue | Fulltext and methods evidence |", "| --- | --- | --- |"]
    for i, row in enumerate(allrows, 1):
        title = f"{i:02d}. {row['title']} ({row['year']})"
        parts.append(f'<details data-area="{esc(row["group"])}"><summary>{esc(title)}</summary><p class="meta">{esc(row["venue"])} · {esc(row["status"])}<br><a href="{esc(row["proceedings_url"], quote=True)}">Proceedings / publisher</a> · <a href="{esc(row["paper_url"], quote=True)}">Retrieved paper</a></p><table>')
        for label, key in LABELS:
            parts.append(f'<tr><th>{esc(label)}</th><td>{render_value(row, key)}</td></tr>')
        parts.append("</table></details>")
        title_md = row["title"].replace("|", "\\|")
        version = str(row["fulltext_version"]).replace("|", "\\|")
        md.append(f'| [{title_md}]({row["proceedings_url"]}) | {row["venue"]}, {row["year"]} | [Paper]({row["paper_url"]}); {version} |')
    parts.append('''<script>const q=document.querySelector('#query'),a=document.querySelector('#area'),rows=[...document.querySelectorAll('details')];function filter(){let n=0;for(const r of rows){const show=(!a.value||r.dataset.area===a.value)&&r.textContent.toLowerCase().includes(q.value.toLowerCase());r.classList.toggle('hidden',!show);if(show)n++;}document.querySelector('#count').textContent=n+' of 50 papers';}q.addEventListener('input',filter);a.addEventListener('change',filter);</script></html>''')
    (directory / "evidence-matrix.html").write_text("\n".join(parts))
    (directory / "evidence-matrix.md").write_text("\n".join(md) + "\n")
    if any(path.read_bytes() != original for path, original in originals.items()):
        raise RuntimeError("Source inventory changed during generation")
    return dict(conference_papers=len(allrows), unique_titles=len({r["title"].casefold() for r in allrows}),
                group_counts=dict(Counter(r["group"] for r in allrows)),
                required_fields=list(REQUIRED), all_required_fields_present=True,
                original_inventory_bytes_unchanged=True,
                json_bytes=(directory / "evidence-matrix.json").stat().st_size,
                html_bytes=(directory / "evidence-matrix.html").stat().st_size)


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
