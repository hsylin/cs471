#!/usr/bin/env python3
"""Plot the frozen completed-runtime contrasts; no fitting or data selection."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output-prefix',type=Path,required=True)
    a=p.parse_args();s=json.loads(a.summary.read_text())
    if s['phase']!='final' or s['status']!='complete':raise ValueError('Complete final analysis required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for group in ('changing','stable'):
        r=next(x for x in s['aggregates'] if x['group']==group)
        rows.append((group.capitalize()+' aggregate',r['ratio'],r['ci95'],True))
    for regime in ('stable','transient','persistent','switching'):
        for batch in (128,1024):
            r=next(x for x in s['cells'] if x['regime']==regime and x['batch_tokens']==batch)
            rows.append((f'{regime.capitalize()}, B={batch:,}',r['ratio'],r['adjusted_ci'],False))
    fig,ax=plt.subplots(figsize=(9,6.1))
    for y,(label,ratio,ci,aggregate) in enumerate(rows):
        color='#176d8d' if aggregate else '#475569'
        ax.errorbar(ratio,y,xerr=[[ratio-ci[0]],[ci[1]-ratio]],fmt='o',color=color,
                    capsize=4,markersize=7 if aggregate else 5,lw=1.7)
    ax.axhline(1.5,color='#cbd5e1',lw=.8)
    for x,color,style,label in ((1.,'#111827','-','Native SKIP'),(.95,'#16803d','--','5% gain'),(1.02,'#b45309',':','2% regression')):
        ax.axvline(x,color=color,ls=style,lw=1.2,label=label)
    ax.set_yticks(range(len(rows)),[x[0] for x in rows]);ax.invert_yaxis()
    ax.set_xlabel('Online / native completed elapsed time (lower is faster)')
    ax.set_title('Fixed-placement resident dispatch on two H100s',loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=3,frameon=False)
    fig.text(.02,.02,'256 runs; eight fresh source pairs; two allocation repetitions. Aggregate: 95% block-bootstrap intervals.\nCell intervals adjust for 2 stable / 6 changing comparisons. Conditional on this constructed layer replay.',fontsize=8,color='#475569')
    fig.tight_layout(rect=(0,.085,1,1));a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(a.output_prefix.with_suffix('.'+ext),dpi=180,bbox_inches='tight')

if __name__=='__main__':main()
