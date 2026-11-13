"""Admit the isolated resident plugin, then execute the unchanged vLLM CLI."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from full_serving.resident_plugin import load_admission


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "serve":
        raise RuntimeError("The resident entry point only supports vllm serve")
    load_admission()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "full_serving/bootstrap"), env.get("PYTHONPATH", "")])
    env["VLLM_PLUGINS"] = "cs471_resident"
    os.execvpe("vllm", ["vllm", *argv], env)


if __name__ == "__main__":
    main()
