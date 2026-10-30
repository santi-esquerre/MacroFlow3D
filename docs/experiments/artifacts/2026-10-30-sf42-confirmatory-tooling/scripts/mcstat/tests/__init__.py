"""Tests of mcstat.  Run from the directory that contains the package::

    python3 -W error -m unittest discover -s . -t .

Environment switches: ``MCSTAT_HEAVY=1`` runs the heavier Monte Carlo checks (larger replicate counts); ``MCSTAT_PAR2_ALPHAX_S025``
points to a text file with the 25 per-realization Par2 ``alpha_x`` values of ``s025`` (host-only data) for the Par2 Gate 5 test.
"""
