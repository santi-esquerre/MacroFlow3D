"""python3 -m sf43add.camp {plan|finalize|s0-report|check-seeds} ...   (README.md)"""
import sys

from .cli import main

sys.exit(main())
