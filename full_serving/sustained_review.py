"""Recompute independent arithmetic and render the closed sustained experiment.

Usage: python -m full_serving.sustained_review
Requires the restored public run-v3 records and the completion receipt.
This does not execute inference or modify experimental records.
"""
import csv
import html
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
COHORT = BASE / 'results/full-serving/sustained-512-v1'
ROOT = COHORT / 'run-v3'
OUT = BASE / 'outputs/full-serving'
ARMS = ('off', 'b1', 'b2', 'b3', 'b4', 'resident_v2')


def percentile(values, q):
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def main():
    receipt = json.loads((COHORT/'completion-v3/completion.json').read_text())
    assert receipt['status'] == 'completed_and_audited'
    result = json.loads((OUT/'sustained-512-comparison.json').read_text())
    reference_identity = None
    checkpoint = None
    table, migration_table, comparisons = [], [], []
    for mode in ('natural', 'matched'):
        cp = json.loads((ROOT/mode/'results/checkpoint-verified.json').read_text())
        cp.pop('checked_at')
        if checkpoint is None:
            checkpoint = cp
        assert checkpoint == cp, 'Checkpoint assets/shards changed between stages'
        for ds in ('longbench', 'numinamath'):
            target = json.loads((ROOT/'matched/results'/f'{ds}-measured-lengths.json').read_text())
            off_rows = None
            for arm in ARMS:
                a = result[mode]['datasets'][ds]['arms'][arm]
                s = a['summary']
                identity = [{k: v for k, v in i.items() if k != 'checkpoint_manifest_sha256'} for i in a['runtime']]
                if reference_identity is None:
                    reference_identity = identity
                assert identity == reference_identity, 'Hardware/native code/transport mismatch'
                directory = ROOT/mode/'results'/ds/arm
                records = [json.loads(line) for line in (directory/'measured.jsonl').open()]
                rows = sorted((r for r in records if r['type'] == 'request'), key=lambda r: r['request_index'])
                assert len(rows) == 512 and all(r['success'] for r in rows)
                assert [r['request_index'] for r in rows] == list(range(512))
                if off_rows is None:
                    off_rows = rows
                assert [(r['request_id'], r['input_sha256'], r['arrival_offset_ns']) for r in rows] == [(r['request_id'], r['input_sha256'], r['arrival_offset_ns']) for r in off_rows]
                if mode == 'matched':
                    assert all(r['usage']['completion_tokens'] == target[r['request_id']] and r['finish_reason'] == 'length' for r in rows)
                for field, start, end in [('response_from_schedule_ms', 'scheduled_ns', 'end_ns'), ('ttft_from_schedule_ms', 'scheduled_ns', 'first_content_ns')]:
                    vals = [(r[end]-r[start])/1e6 for r in rows]
                    assert math.isclose(math.fsum(vals)/512, s[field]['mean'], rel_tol=1e-12)
                    assert math.isclose(percentile(vals, .95), s[field]['p95'], rel_tol=1e-12)
                epoch = a['timing']['epoch_monotonic_ns']
                last = max(r['end_ns'] for r in rows)
                horizon = a['timing']['offered_horizon_ns']
                assert max(last-epoch, horizon)/1e9 == s['completed_elapsed_s']
                assert max(0, last-epoch-horizon)/1e9 == s['drain_s']
                tokens = sum(r['usage']['completion_tokens'] for r in rows)
                assert tokens == s['completed_output_tokens']
                table.append(dict(mode=mode, dataset=ds, arm=arm,
                    mean_response_s=s['response_from_schedule_ms']['mean']/1000,
                    p95_response_s=s['response_from_schedule_ms']['p95']/1000,
                    mean_ttft_ms=s['ttft_from_schedule_ms']['mean'],
                    p95_ttft_ms=s['ttft_from_schedule_ms']['p95'],
                    elapsed_s=s['completed_elapsed_s'], drain_s=s['drain_s'],
                    output_tokens=tokens, ceiling_8192_hits=sum(r['usage']['completion_tokens']==8192 for r in rows),
                    completed_requests_per_s=s['completed_requests_per_s'],
                    decode_fraction=1-a['phase_balance']['prefill_fraction'],
                    mean_change_vs_off_pct=100*(result[mode]['datasets'][ds]['contrasts'][arm+'/off']['mean_response_ratio']-1)))
                if arm in ('b1', 'b2', 'b3', 'b4'):
                    m = a['migrations']; begin = a['timing']['epoch_wall_ns_approx']; end = begin+a['timing']['completed_elapsed_ns']
                    cycles = [c for c in m['cycles'] if begin <= c['proposal_wall_ns_rank0'] <= end]
                    migration_table.append(dict(mode=mode, dataset=ds, arm=arm,
                        measured_proposals=len(cycles), measured_changed_proposals=sum(c['changed_slots']>0 for c in cycles),
                        session_proposals=len(m['cycles']),
                        session_all_rank_completed_layers=sum(c['all_rank_completed_layers'] for c in m['cycles']),
                        session_censored_layers=sum(c['uncompleted_or_unobserved_layers'] for c in m['cycles']),
                        session_received_GB=m['remote_received_tensor_bytes_all_ranks']/1e9))
            arms = result[mode]['datasets'][ds]['arms']
            for left, right in [('resident_v2','off'), ('b4','b3'), ('b4','b2'), ('b1','b2')]:
                comparisons.append(dict(mode=mode, dataset=ds, contrast=left+'/'+right,
                    mean_response_change_pct=100*(arms[left]['summary']['response_from_schedule_ms']['mean']/arms[right]['summary']['response_from_schedule_ms']['mean']-1)))
    p = result['payback_diagnostic']
    assert p['opportunity_observed_all_ranks']
    for side in ('allow', 'skip'):
        cycles = p[side+'_migrations']['cycles']
        assert all(c['changed_slots']==0 for c in cycles if side=='skip' or c['step']!=3750)
        if side=='allow':
            target_cycle = next(c for c in cycles if c['step']==3750)
            assert target_cycle['changed_slots']>0 and target_cycle['all_rank_completed_layers']==48
    review = dict(status='passed', main_sessions=24, main_requests=12288,
        exact_matched_lengths=True, paired_input_and_arrival_identity=True,
        same_eight_gpu_uuids_native_sources_transport=True, checkpoint_assets_shards_identical=True,
        independent_mean_p95_ttft_elapsed_drain_tokens=True,
        payback_intervention_scope_checked=True, rows=table, migrations=migration_table, contrasts=comparisons)
    (OUT/'sustained-512-review.json').write_text(json.dumps(review, indent=2)+'\n')
    with (OUT/'sustained-512-comparison.csv').open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(table[0]), lineterminator='\n');w.writeheader();w.writerows(table)
    def render_table(headers, rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
    sections=['<h2>Completed results</h2><p><strong>No practically useful end-to-end speedup over EPLB OFF.</strong> All 24 main sessions completed, totaling 12,288 successful measured requests. With exact output lengths, resident dispatch reduced mean response by only 0.78% on LongBench and 0.43% on NuminaMath, below the declared 5% useful-gain target. One group cannot establish repeatability or statistical significance.</p>',
        '<p>B4 reduced matched NuminaMath mean response by 5.04% relative to B3, but remained 7.80% slower than OFF. Its matched LongBench improvement over B3 was 0.86%, while remaining 4.16% slower than OFF. This is an observed policy contrast, not proof that TV identifies higher-payback individual migrations; there is no equal-frequency cooldown control here. B1 versus B2 mixes replica capacity, placement and policy consequences.</p>']
    for mode in ('natural','matched'):
        sections.append('<h3>'+mode.title()+' stopping / length control</h3>')
        sections.append(render_table(['Dataset','Arm','Mean response (s)','p95 (s)','TTFT mean/p95 (ms)','Elapsed/drain (s)','Output tokens','8192 hits','Mean vs OFF'],
            [[r['dataset'],r['arm'],f"{r['mean_response_s']:.3f}",f"{r['p95_response_s']:.3f}",f"{r['mean_ttft_ms']:.2f} / {r['p95_ttft_ms']:.2f}",f"{r['elapsed_s']:.3f} / {r['drain_s']:.3f}",r['output_tokens'],r['ceiling_8192_hits'],f"{r['mean_change_vs_off_pct']:+.2f}%"] for r in table if r['mode']==mode]))
    sections.append('<p>Negative percentages mean lower latency. Natural output counts vary: resident NuminaMath generated 796,064 tokens versus OFF’s 741,753; its natural mean response was 8.98% higher. The matched comparison forces exactly 741,753 NuminaMath and 75,246 LongBench output tokens per arm, including the same per-request counts. It does not force identical generated tokens, expert routes or answer quality. Every matched request finishes at its assigned length; “8192 hits” above counts only requests actually reaching 8192, not all length stops.</p>')
    sections.append('<p>Both workloads are decode-dominated in summed native request-phase durations: matched OFF is 86.08% decode for LongBench and 99.76% for NuminaMath. These are overlapping per-request durations, not isolated GPU busy time or bandwidth/compute proof. Completed elapsed includes the fixed arrival horizon and drain; it is relatively insensitive to small service gains when arrivals set the pace. Startup and two warmups are outside the warmed serving metrics. The public CSV reports achieved requests/s separately.</p>')
    sections.append('<h3>Migration opportunities and completion</h3><p>“Measured” proposals were observed between the measured arrival epoch and completed elapsed boundary. Remaining counts and bytes cover the whole server session, including warmup. Completed layers require observed GPU completion on all eight ranks; no-op commits are included and are not counted as beneficial migrations. Censored layers remain unknown, never silently treated as completed.</p>')
    sections.append(render_table(['Mode','Dataset','Arm','Measured proposals / changed','Session proposals','Completed / censored layers','Session received GB'],
        [[r['mode'],r['dataset'],r['arm'],f"{r['measured_proposals']} / {r['measured_changed_proposals']}",r['session_proposals'],f"{r['session_all_rank_completed_layers']} / {r['session_censored_layers']}",f"{r['session_received_GB']:.2f}"] for r in migration_table]))
    sections.append('<p>In matched NuminaMath, B3 allowed 702 layer decisions across 18 measured opportunities; B4 allowed 75 across 19 and rejected 747 as unstable. Session received expert-tensor bytes were 975.32 GB for B3 and 120.67 GB for B4. Reduced movement coincides with lower latency versus B3, but does not isolate the selection value of stability from merely moving less. Warmup placements and autoregressive routing can differ. GPU-event completion is not a full-weight hash, and draining async work is not commit proof.</p>')
    sections.append(f'<h3>Migration payback remains causally inconclusive</h3><p>The target step 3750 existed on every rank. The allow branch changed 6,462 slots and completed all 48 layers; the skip branch and all other opportunities retained their maps. However, predecision demand hashes and candidates differed on all eight ranks; only {p["completed_predecision_request_hash_matches"]}/{p["completed_predecision_requests"]} completed predecision outputs matched, and only {p["equal_generated_outputs"]}/{p["requests"]} outputs matched overall. The old maps agreed, but the active KV state was not cloned. Allow accumulated {abs(p["net_request_execution_saved_s"]):.3f} seconds more HTTP execution time. This cannot establish either a migration benefit or its causal repayment time.</p>')
    sections.append('<p>The main policy-level cumulative response curves are included in the JSON. They charge online overhead but cannot separate the cost and benefit of an individual migration. The actionable conclusion is to keep OFF as the reference and treat resident dispatch as near parity on these matched-length workloads; larger sustained request counts did not produce the declared useful gain.</p>')
    sections.append('<h3>Audit and reproduction</h3><p>Stable cluster/local bytes, frozen Git sources, hardware, request completion and independent reconstruction passed. The first audit compared absolute metric file paths across hosts and failed; the preserved correction validates both expected paths and normalizes only those labels. All 48 differences were path labels; computed fields matched exactly. No GPU measurements were repeated for this correction.</p><p>Restore the public archive, then run <code>python -m full_serving.sustained_analysis final results/full-serving/sustained-512-v1/run-v3/natural/results --output /tmp/natural.json</code> and similarly for <code>matched</code>. Run <code>python -m full_serving.sustained_payback results/full-serving/sustained-512-v1/run-v3 --output /tmp/payback.json</code>, then <code>python -m full_serving.sustained_review</code>. Independent arithmetic is repeatable CPU analysis, not independent GPU replication.</p>')
    sections.append('<p>Public records are stored losslessly under <code>results/full-serving/sustained-512-v1/public-archive-v3</code>. Run <code>python -m full_serving.sustained_archive restore</code> before reconstruction. The archive manifest verifies both compressed bytes and original SHA256 hashes; earlier failed and superseded attempts are included. Raw archive payloads are currently retained locally pending publication approval; the GitHub branch contains the archive manifest, reports, analysis code and audit records. External raw-record reconstruction requires access to those payloads. The completion receipt, path-amendment record, privacy review and independent arithmetic review are preserved alongside the results.</p>')
    body='\n'.join(sections)
    protocol=(BASE/'docs/full-serving-sustained-512.html').read_text()
    start='<!-- REVIEWED RESULTS START -->';end='<!-- REVIEWED RESULTS END -->'
    if start in protocol:
        protocol=protocol[:protocol.index(start)]+protocol[protocol.index(end)+len(end):]
    protocol=protocol.replace('<strong>Status: launched; no final results yet.</strong> Three instrumentation-admission pilot servers precede the natural-stopping and matched-length comparisons. Earlier short-workload findings remain unchanged.', '<strong>Status: completed and audited.</strong> All 24 main sessions and both payback diagnostics finished. Earlier negative results and failed attempts are preserved.')
    protocol=protocol.replace('<h2>Six configurations</h2>',start+body+end+'\n<h2>Six configurations</h2>')
    (BASE/'docs/full-serving-sustained-512.html').write_text(protocol)
    (OUT/'sustained-512-comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>CS471 sustained results</title><style>body{font:16px/1.5 system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd;padding:7px}th{background:#eef}h2,h3{margin-top:2em}</style><h1>512-request full-model comparison</h1>'+body)
    print(json.dumps({'status':'passed','main_requests':12288,'report':str(OUT/'sustained-512-comparison.html')}))


if __name__ == '__main__':
    main()
