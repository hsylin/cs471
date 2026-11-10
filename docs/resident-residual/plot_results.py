"""Registered endpoint and per-bundle descriptive effects; completed results only."""
import json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path.cwd());parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();ROOT=args.repo_root.resolve();OUT=args.output.resolve()
a=json.loads((ROOT/'results/resident-residual/final-v1/analysis.json').read_text());assert a['valid']
p=a['confirmatory']['primary'];pairs=[r for r in a['paired_cases'] if r['contrast']=='fix_over_stockoff' and r['rho']==.85]
groups=sorted({r['content_group_id'] for r in pairs});colors=['#3466a6' if 'gsm8k' in g else '#a65a33' for g in groups]
fig,axes=plt.subplots(1,2,figsize=(11,7),sharey=True)
for ax,metric,title in zip(axes,('service_mean_ns','response_mean_ns'),('Service time','Response time')):
 for i,g in enumerate(groups):
  x=math.exp(statistics.mean(r['log_ratios'][metric] for r in pairs if r['content_group_id']==g))
  ax.plot(x,i,'o',color=colors[i],ms=5)
 ax.axvline(1,color='#333',lw=1,label='Equal to EPLB-off');ax.axvline(.95,color='#618356',ls='--',lw=1)
 ax.axvline(1.02,color='#b37867',ls=':',lw=1)
 ax.set_title(title);ax.set_xlabel('Fix / unmodified EPLB-off (lower is better)');ax.grid(axis='x',alpha=.15)
 ax.set_ylim(-1,len(groups)+1)
ax=axes[1];i=len(groups)+.25
ax.errorbar(p['ratio'],i,xerr=[[p['ratio']-p['ci95'][0]],[p['ci95'][1]-p['ratio']]],fmt='D',color='black',capsize=5,ms=6)
axes[0].set_yticks(list(range(len(groups)))+[i],['GSM8K train '+g.rsplit('-',1)[1] if 'gsm8k' in g else 'WildChat '+g.rsplit('-',1)[1] for g in groups]+['Primary + 95% CI'])
fig.suptitle('Fresh high-load layer jobs: same two H100s',fontsize=15)
fig.text(.5,.015,'Dots: each content bundle averaged across its two arrival blocks. Service effects are descriptive.\nResponse interval: registered whole-bundle bootstrap; not full-model serving performance.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.95));OUT.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT/'fresh-effects.svg');fig.savefig(OUT/'fresh-effects.png',dpi=180)
print(OUT/'fresh-effects.png')
