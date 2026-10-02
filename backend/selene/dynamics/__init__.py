"""SELENE dynamics engine.

Modules
-------
cr3bp       Earth-Moon circular restricted three-body problem (nondimensional, rotating frame)
ephemeris   DE440s access and the Earth + Moon + Sun (+ SRP) point-mass model in GCRF [km, s]
frames      instantaneous rotating <-> GCRF <-> Moon-centered transforms
propagate   frame-aware :class:`Trajectory` wrapper around both models

Submodules are imported explicitly (``from selene.dynamics import cr3bp``) to keep import cost low.
"""
