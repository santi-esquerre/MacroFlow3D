"""SF-42 N5: per-realization reduction on control planes (both codes), manifest and pruning (see `reduce.py`).

SF-43 change (disclosed): `python3 -m sf42red prune` gained the option `--min-retained N` (default 5 = decision G8, so the behaviour
is unchanged unless it is passed; 0 is allowed with an explicit retained list).  Reason: the SF-43 campaign prunes per segment
directory against an arm-level retained list (see `prune.py`).  Nothing else of this package changed.
"""
