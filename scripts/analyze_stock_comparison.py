#!/usr/bin/env python3
"""Analyze complete genuine-stock comparator records from a frozen plan."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stock_comparison.analysis import main
if __name__=='__main__':main()
