"""Explicit experiment plugin; native reference arms never load this plugin."""
import os


def register():
    arm=os.environ.get('CS471_EXPERIMENT_ARM')
    if arm in ('resident_v1','resident_v2'):
        if arm=='resident_v2':
            from full_serving import resident_policy
            from full_serving.resident_pair import PairMapper
            resident_policy.ResidentMapper=PairMapper
        from full_serving.resident_plugin import register as install
        install()
    elif arm in ('b3','b4'):
        from full_serving.proposal_gate_plugin import register as install
        install()
    else:raise RuntimeError('Unexpected custom experiment arm')
