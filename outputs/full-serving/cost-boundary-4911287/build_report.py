"""Render reviewed aggregate data only; raw archives are never inputs."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def load(name):
    return json.loads((ROOT / name).read_text())
def esc(x):
    return html.escape(str(x))
def num(x):
    return f"{x:,.3f}"
def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+esc(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'

def render():
    final, diag = load('final-summary.json'), load('diagnostic-summary.json')
    names = {'off':'OFF', 'b4':'原始 B4', 'b4_noop':'B4 + unchanged-map 修正'}
    datasets = {'longbench':'LongBench', 'numinamath':'NuminaMath'}
    out = ['''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CS471：成本邊界與最終 serving 比較</title><style>
body{font:16px/1.75 system-ui,sans-serif;color:#17212b;background:#f7f8fa;margin:0}main{max-width:1120px;margin:auto;padding:36px 24px 72px}h1{font-size:30px;line-height:1.35}h2{font-size:23px;margin-top:40px}h3{font-size:18px;margin-top:28px}p{max-width:1000px}.result{border-left:5px solid #9c3e21;background:#fff3ed;padding:20px 24px}.note{color:#465667;font-size:14px}.scroll{overflow-x:auto;background:white;border:1px solid #d5dce4;border-radius:6px;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dce2e8;text-align:right;white-space:nowrap}th{background:#eaf0f6}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}code{overflow-wrap:anywhere;font-size:13px}a{color:#155fa0}li{margin:8px 0}footer{margin-top:40px;border-top:1px solid #cbd4de;padding-top:20px}@media(max-width:600px){main{padding:20px 14px}h1{font-size:25px}.result{padding:14px}}</style></head><body><main>
<p class="note">CS471 · 2026-11-13 · 同一組八張 H100 · 單組描述性實驗</p>
<h1>省略不變映射更新，仍未達成有用的 serving 加速</h1>
<div class="result"><strong>最終六組、3,072 筆請求全數成功並通過稽核。</strong><p>修正相對原始 B4 的平均回應時間減少 0.90%（LongBench）與 2.21%（NuminaMath），均未達預先設定的 5% 有用增益。修正相對 OFF 仍慢 4.60% 與 5.61%，均超過 2% 退步門檻。因此，本次結果不支持把此修正視為達標的端到端優化。</p></div>
<p>修正只省略已經由八個 rank 一致確認、與舊映射完全相同的 inverse-map commit。原生 planner、worker、傳輸、readiness、pending reset 與 consumed-event retirement 都保留。先通過正確性與診斷審閱，再凍結來源及協定，最後才進行這六組測量；未依最終結果選最快版本或重調政策。</p>
<h2 id="final">最終端到端比較</h2><p>每組 512 筆。完整回應與 TTFT 從預定到達時間起算，包含連線建立及排程等待。正的「增益」代表平均回應時間下降。</p>''']
    for ds in datasets:
        rows = [r for r in final['rows'] if r['dataset'] == ds]
        out += ['<h3>'+datasets[ds]+'</h3>', table(['方案','平均回應 ms','p95 回應 ms','平均 TTFT ms','p95 TTFT ms'], [[names[r['treatment']],num(r['mean_response_ms']),num(r['p95_response_ms']),num(r['mean_ttft_ms']),num(r['p95_ttft_ms'])] for r in rows])]
    comparisons=[]
    for ds, values in final['comparisons'].items():
        for label, v in values.items():
            left,right=label.split('/')
            comparisons.append([datasets[ds],names[left]+' / '+names[right],num(v['mean_response_gain_percent'])+'%', '是' if v['useful_mean_gain_at_least5percent'] else '否', '是' if v['mean_response_regression_exceeds2percent'] else '否'])
    out += [table(['資料集','比較','平均回應增益','達 5%','退步超過 2%'],comparisons),'<p class="note">這些是同一組測量的比值，沒有信賴區間或顯著性主張。相同政策仍可能產生不同的 token、路由與 batching；不能把差值直接當作 commit 操作的因果成本。</p>']
    out += ['<h2 id="timing">到達窗口、排空與完整性</h2>',table(['資料集','方案','總時間 s','到達窗口 s','排空 s','輸出 tokens','8192 固定目標數'],[[datasets[r['dataset']],names[r['treatment']],num(r['elapsed_s']),num(r['offered_horizon_s']),num(r['drain_s']),r['output_tokens'],r['inherited_8192_targets']] for r in final['rows']]),'<p>所有方案均 512/512 成功、零失敗。每組 LongBench 固定輸出 75,246 tokens；NuminaMath 固定輸出 741,753 tokens。NuminaMath 的 26 筆 8192-token 長度繼承指定 natural OFF 的測量目標，並非這些 matched 方案出現自然 EOS 截斷的新證據。固定長度不代表輸出內容或品質相同。</p>']
    out += ['<h2 id="commits">提案、commit 與修正實際執行</h2>']
    rr=[]
    for e in final['gate_and_commit_evidence']:
        m=e['measured_window'];noop=e['no_op'];n=noop['ranks'][0] if noop else None
        rr.append([datasets[e['dataset']],names[e['treatment']],m['proposal_cycles'],m['layer_decisions'].get('allow',0),e['all_rank_layer_commits_completed'],e['unobserved_completion_count'],n['elided_map_commits'] if n else '—'])
    out += [table(['資料集','方案','測量期提案','測量期 allow 層','全程八 rank 完成層','未觀察完成','每 rank 省略數'],rr),'<p>省略數包含 warmup：LongBench 每個 rank 197 次，NuminaMath 1,093 次；八個 rank 都觀察到其對應的原生完成事件，沒有 censored elision。全程完成層數含 warmup，且要求八 rank 對齊。測量期事件依 rank0 近似 wall time 分類，完成事件可能來自較早提案。排空不是 commit 證明。</p><p>原始 B4／修正的測量期 allow 層數分別為 LongBench 44／43、NuminaMath 88／58。執行軌跡不同，因此 0.90%／2.21% 不是已隔離的操作成本。省略操作次數也不能換算成節省時間。</p>']
    out += ['<h2 id="phases">原生請求階段</h2>',table(['資料集','方案','prefill 加總 s','decode 加總 s'],[[datasets[r['dataset']],names[r['treatment']],num(r['native_request_phase_duration_sum_s']['prefill']),num(r['native_request_phase_duration_sum_s']['decode'])] for r in final['rows']]),'<p class="note">這是跨請求、跨 engine 的原生 duration 加總，不能當作 GPU 忙碌時間、可相加的 critical path，或 compute／memory bottleneck 證明。</p>']
    out += ['<h2 id="controls">成本控制：保留全部 16 組結果</h2><p>每格 512 筆且全數成功；前六個控制共用相同 balanced-32 實體配置。record 包含 atomics 與原生 window copies；history 加入 fused history、snapshot、allreduce。native_hash 與 idle 還改變 mapping kernel。resident／static 也改變動態 replica routing，並非純 planner 成本比較。</p>']
    for ds in datasets:
        rows=[r for r in diag['cost_controls']['rows'] if r['dataset']==ds]
        out += ['<h3>'+datasets[ds]+'</h3>',table(['控制','平均回應 ms','p95 回應 ms','平均 TTFT ms','p95 TTFT ms'],[[r['control'],num(r['response_ms']['mean']),num(r['response_ms']['p95']),num(r['ttft_ms']['mean']),num(r['ttft_ms']['p95'])] for r in rows])]
    out += ['<p>record／idle 差異為 +0.151%／+0.564%（Long／Num）；history／record 為 +0.355%／+0.488%。這些差異不可相加成因果稅。observer／gate-only 的實際決策不同，也無法單獨識別 observer overhead。static／native 的 LongBench +133.212% 與 NuminaMath −1.728% 更不支持通用 static 加速。</p>']
    out += ['<h2 id="shapes">持續且八 rank 對齊的形狀診斷</h2><p>使用原生 DP dispatch ordinal 系統性取樣，兩資料集均覆蓋全部十個 offered-window 區間；不把 missing rank 填零。明確的非 profile dummy 分支表示空工作，descriptor padding 不算實際工作。</p>']
    rows=[]
    for ds,v in diag['sustained_shapes'].items():
        c=v['coverage']
        for kind,label in [('decode_only','decode-only'),('mixed','mixed')]:
            x=c['sampled_work_classes'][kind]
            rows.append([datasets[ds],label,x['real_token_max_over_mean']['n'],num(x['real_token_max_over_mean']['mean']),num(x['attention_proxy_max_over_mean']['mean']),num(x['padded_tokens_per_real_token']['mean'])])
    out += [table(['資料集','取樣類別','樣本數','token max/mean 均值','attention proxy max/mean 均值','padding/real 均值'],rows),'<p>LongBench 187 個 active aligned 樣本（另有 1 個 all-empty），其中 7 個在 drain；NuminaMath 907 個 active aligned 樣本，其中 116 個在 drain。兩者都沒有抽到 prefill-only 樣本，這不表示沒有 prefill 執行。</p><p>樣本不提供無偏頻率、精確 FLOPs、逐 iteration KV 或 bandwidth。query-only CUDA span 包含 host/preparation/forward/peer wait 並排除稍後 EPLB；本次沒有詳細 exposed-critical-path trace，瓶頸歸因仍未解決。</p>']
    out += ['<h2 id="transition">候選價值與遷移控制</h2>',table(['資料集','控制','平均回應 ms','p95 回應 ms','平均 TTFT ms','p95 TTFT ms'],[[datasets[r['dataset']],r['control'].upper(),num(r['response_ms']['mean']),num(r['response_ms']['p95']),num(r['ttft_ms']['mean']),num(r['ttft_ms']['p95'])] for r in diag['transition_controls']['rows']]),'<p>同一個最先改變的 pilot 候選配置，不是選最快候選。KEEP 保留舊配置；PREINSTALLED 在 warmup 前安裝並排除啟動遷移成本；CHARGED 在測量期第一次符合條件的原生 snapshot 觸發，LongBench step 3750、NuminaMath step 18750，均改變 791 slots 並觀察到八 rank、48 層完成。</p><p>兩資料集在觸發前，八 rank 的 live demand 與已完成輸出 prefix 已不相同；PREINSTALLED 的 warmup／KV 狀態也不同。因此沒有個別遷移的因果 payback 結論。PREINSTALLED／KEEP 的平均差異均低於 1%，本次證據未建立 5% 有用 headroom，未再新增 runtime-aware admission policy。</p>']
    out += ['<h2 id="validity">正確性、失敗紀錄與解讀界線</h2><ul><li>修正的既有 admission 已驗證八 rank metadata、CUDA graph readers 與 48 層 full weights；最終測量關閉昂貴 shadows。最終 commit audit 不等於再次驗證全模型權重或品質。</li><li>原 noop admission 的初次 CPU audit 錯把合法的下一層預取視為異常；修正分析後通過，原失敗保留，沒有重跑 GPU。</li><li>host-pair-v1、controls-admission-v1、shapes-run-v1 的 GPU 前失敗保留。舊 shape 診斷缺少持續對齊證據，另以原生 DP ordinal 診斷補足；沒有重跑健康實驗。</li><li>同一組八張 H100：Qwen3-30B-A3B BF16，EP8／DP8／TP1，native all-gather/reduce-scatter。B4 有 32 replicas，OFF 沒有額外 replicas，因此對 OFF 的比較包含配置差異。</li><li>固定 prompts、到達 calendars、rates 與輸出長度；每次 HTTP force-close，POST 不重試。每組使用全新服務程序。LongBench 順序 OFF→B4→修正；NuminaMath 反向。</li><li>單組描述性結果；不保證 tokens、routes、batching、KV、quality 相同，也不提供統計顯著性。沒有將 host sums、NCCL duration 或 drain 推成節省時間、bandwidth 或 commit 證明。</li><li>選擇依據是精確冗餘與正確性；詳細 profiling 未作為這個有限修正的比較前提。關鍵路徑歸因未完成，不能用這份報告宣稱已找到主要瓶頸。</li><li>原始紀錄、prompts、outputs、token arrays、logs、profiles 皆未公開。公開彙總只能重算表格與比值；從原始請求完整重建須另有私有資料授權。</li></ul>']
    out += ['<h2 id="provenance">可查核的公開產物</h2><p><a href="final-summary.json">最終彙總 JSON</a> · <a href="final-summary.csv">CSV</a> · <a href="diagnostic-summary.json">診斷彙總</a> · <a href="final-protocol.json">測量前凍結協定</a> · <a href="source-manifest.json">來源檔案 manifests</a> · <a href="README.md">重建方式與限制</a></p><p>測量來源：<code>'+esc(final['source_revision'])+'</code><br>協定 SHA256：<code>'+esc(final['protocol_sha256'])+'</code><br>完整私有重建報告 SHA256：<code>'+esc(final['report_sha256'])+'</code></p><footer>結論保留負面結果：映射省略正確且實際執行；端到端改善未達標，相對 OFF 仍退步。所有後續公開排版均不改變已凍結的測量內容。</footer></main></body></html>']
    return '\n'.join(out)

if __name__ == '__main__':
    (ROOT / 'report.html').write_text(render())
