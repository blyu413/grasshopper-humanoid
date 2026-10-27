# Upstream source

Repository: https://github.com/zita-ch/supermimic_vibed
Commit: a1508d42c9d1699a52e8233ab16d08c488441ab9
License: MIT, copyright (c) 2026 ETH Zurich, Chong Zhang.

`networks.py` and `rotations.py` are byte-for-byte upstream copies.
Git blob IDs: networks 91b1c9995e9c5a34be1e7c3a600ffc92be7369ab, rotations 702a0066b0bab9f81b0cf25153bdb386f471c741.
`mdp.py` extracts only NUM_J, XI_DIM, INV_DIM and invariant_features from upstream;
motion targets and native task rewards are implemented outside this package.

Native task input dimensions are adapted in ../model.py. No upstream network
hidden layers, activations, policy initialization, latent, or normalizers are
changed. Physical context extensions are explicit in the architecture manifest.

Imported through the local `motion-force-adapter/supermimic/reference` on 2026-10-27.
The NEST task adapter lives in `adapter.py`; no runtime dependency on that sibling repository is required.
