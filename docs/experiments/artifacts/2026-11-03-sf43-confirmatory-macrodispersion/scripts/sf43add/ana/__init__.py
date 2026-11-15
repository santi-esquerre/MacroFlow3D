"""sf43add.ana: analysis of SF-43 addendum 1 (exploratory arms J1, J3, J9; extension J8).  Descriptive, outside the Holm family.

Rules: ``preregistration-addendum-1.md`` / ``prereg_addendum.json`` (the text governs).  Reads data through the FROZEN ``sf43ana.loader``
and the frozen SF-42 ``mcstat``; nothing frozen is modified.  See README.md.  Run from the repository root with
``PYTHONPATH=<A>/scripts:<SF42>/scripts``::

    python3 -W default -m sf43add.ana run|check|repro|j8-trigger ...
"""
import sf43ana  # noqa: F401  (puts the SF-42 scripts directory on sys.path: mcstat, sf42red)
