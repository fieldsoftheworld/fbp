"""Self-contained experiment code for the GRSL revision (BoundaryVote).

Every stage reads/writes under ``paths.out_root()`` and borrows the inference
and metric primitives from the local ``ftw-planet`` checkout through
``ftp_compat`` so numbers stay byte-comparable with Fields of the Planet.
"""
