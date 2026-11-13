"""Only the optimized monitoring arms load this plugin."""
import os

def register():
    if os.environ.get("CS471_EXPERIMENT_ARM") not in ("b3_light","b4_light"):
        raise RuntimeError("Unexpected optimized monitoring arm")
    from full_serving.monitoring_gate_plugin import register as install
    install()
