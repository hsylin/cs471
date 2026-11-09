#!/usr/bin/env python3
"""Analyze verified complete resident-fix calibration/pilot/final records."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from resident_fastpath.analysis import main
if __name__=='__main__':main()
