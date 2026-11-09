"""Fresh seeds and prespecified three-arm order; unchanged calendar arithmetic."""
import itertools
import random
from arrival_fix.calendars import frozen_weights, make_calendar, validate_calendar
from arrival_combo.common import ARMS


def arrival_blocks():
    states = [('opt-pilot-01','domain0'), ('opt-pilot-01','domain1'),
              ('opt-pilot-02','domain0'), ('opt-pilot-02','domain1')]
    return [dict(id=f'combo-fresh-{i+1:02d}', seed=2026110911+i, block_id=s[0], regime=s[1])
            for i,s in enumerate(states*2)]


def order_key(block_id, batch, pattern, rho):
    return f'{block_id}/B{batch}/{pattern}/rho{rho}'


def frozen_orders(matrix, seed):
    """Rotate the middle arm across cases; reverse each order in repetition 1.

    Within each load, cyclic rotations occur 11/11/10 times for 32 final cases.
    Pairwise precedence is exactly balanced; positions are only near-balanced.
    """
    rotations = [ARMS[i:]+ARMS[:i] for i in range(3)]
    result = {}
    for ri, rho in enumerate(matrix['rhos']):
        cases = list(itertools.product([b['id'] for b in matrix['arrival_blocks']],
                                      matrix['batches'], matrix['patterns']))
        random.Random(seed+100+ri).shuffle(cases)
        for i,(block,batch,pattern) in enumerate(cases):
            base = rotations[(i+ri)%3]
            result[order_key(block,batch,pattern,rho)] = [base, list(reversed(base))]
    return result
