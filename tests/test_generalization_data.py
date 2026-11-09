import pytest
from generalization.data import deduplicate, split_records, validate_splits, timestamp_ns, select_arrival_blocks, scaled_calendar


def test_document_and_near_duplicates_cannot_leak():
    data=[dict(dataset='a',source_id='x',text='one two three four five six seven eight nine ten eleven twelve')]
    data += [dict(dataset='b',source_id='y',text='  ONE two three four five six seven eight nine ten eleven twelve ')]
    data += [dict(dataset='a',source_id='x',text='new turn from same conversation')]
    kept,dropped=deduplicate(data)
    assert len(kept)==1 and {r['reason'] for r in dropped}=={'normalized_exact','same_document_or_conversation'}
    kept,_=deduplicate([dict(dataset='a',source_id=str(i),text=f'unique{i}') for i in range(8)])
    blocks=split_records(kept,pilot_blocks=1,final_blocks=1,pilot_documents=2,final_documents=4)
    assert validate_splits(blocks)['documents']==6
    blocks[1]['sources'][0]=blocks[0]['sources'][0]
    with pytest.raises(ValueError):validate_splits(blocks)


def test_exact_timestamp_ties_and_absolute_overload_contract():
    assert timestamp_ns('2023-11-11 00:00:00.123456789')-timestamp_ns('2023-11-11T00:00:00Z')==123456789
    assert timestamp_ns('2023-11-11T01:00:00+01:00')==timestamp_ns('2023-11-11T00:00:00Z')
    rows=[dict(TIMESTAMP=f'2023-11-11 00:00:{i//2:02d}.000',ContextTokens='100',GeneratedTokens='5') for i in range(18)]
    blocks=select_arrival_blocks(rows,trace='code',count=2,jobs=4)
    assert len({x for b in blocks for x in range(b['row_start'],b['boundary_row']+1)})==10
    c=scaled_calendar(blocks[0],numerator=1,denominator=2)
    assert c['arrival_offsets_ns']==[round(t/2) for t in blocks[0]['original_offsets_ns']]
    assert c['absolute_arrivals'] and c['drop_policy'].startswith('none')
    rows.reverse()
    with pytest.raises(ValueError):select_arrival_blocks(rows,trace='code',count=2,jobs=4)
