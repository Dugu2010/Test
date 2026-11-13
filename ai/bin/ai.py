#!/usr/bin/env python3
"""Compatibility launcher for the ForgeCLI application."""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import main


if __name__ == "__main__":
    main()
