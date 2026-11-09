"""B128 single-CTA fusion; exactly the existing greedy home-selection policy."""
from vllm.triton_utils import tl,triton


@triton.jit
def fused_home_map(ids,real_tokens,output,homes,logical_counts,summary,
                   N:tl.constexpr,TOPK:tl.constexpr):
    offsets=tl.arange(0,N)
    logical=tl.load(ids+offsets).to(tl.int32)
    real=tl.load(real_tokens)
    eligible=(logical>=0)&(logical<128)
    valid=eligible&(offsets<real*TOPK)
    counts=tl.histogram(logical,128,mask=valid)
    experts=tl.arange(0,128)
    owner0=(experts>=32)&(experts<80)
    owner1=experts>=80
    f0=tl.sum((owner0&(counts>0)).to(tl.int32),axis=0)
    f1=tl.sum((owner1&(counts>0)).to(tl.int32),axis=0)
    shared_active=tl.sum(((experts<32)&(counts>0)).to(tl.int32),axis=0)
    target=tl.maximum(tl.maximum(f0,f1),(f0+f1+shared_active+1)//2)
    cap0,cap1=target-f0,target-f1
    tokens0=tl.sum(tl.where(owner0,counts,0),axis=0)
    tokens1=tl.sum(tl.where(owner1,counts,0),axis=0)
    shared_ids=tl.arange(0,32)
    shared_counts=tl.gather(counts,shared_ids,axis=0)
    keys=shared_counts*128+(127-shared_ids)
    sorted_keys=tl.sort(keys,dim=0,descending=True)
    home_values=experts
    for index in range(32):
        key=tl.sum(tl.where(shared_ids==index,sorted_keys,0),axis=0)
        weight=key//128
        expert=127-key%128
        active=weight>0
        choose0=(cap0>0)&((cap1<=0)|(tokens0<=tokens1))
        take0=active&choose0
        take1=active&~choose0
        home=tl.where(choose0,expert,128+expert)
        home_values=tl.where(active&(experts==expert),home,home_values)
        tokens0+=tl.where(take0,weight,0)
        tokens1+=tl.where(take1,weight,0)
        cap0-=take0.to(tl.int32)
        cap1-=take1.to(tl.int32)
    safe=tl.where(eligible,logical,0)
    physical=tl.gather(home_values,safe,axis=0)
    tl.store(output+offsets,tl.where(eligible,physical,-1))
    tl.store(homes+experts,home_values)
    tl.store(logical_counts+experts,counts)
    diag=tl.arange(0,8)
    values=tl.where(diag==0,target,
           tl.where(diag==1,f0,
           tl.where(diag==2,f1,
           tl.where(diag==3,shared_active,
           tl.where(diag==4,tokens0,
           tl.where(diag==5,tokens1,
           tl.where(diag==6,target-cap0,target-cap1)))))))
    tl.store(summary+diag,values)
