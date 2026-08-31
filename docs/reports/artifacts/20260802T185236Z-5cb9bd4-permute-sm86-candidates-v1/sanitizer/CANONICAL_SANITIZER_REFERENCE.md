# Canonical sanitizer evidence reference

The four sanitizer logs formerly stored in this directory were byte-identical to the retained copies in `20260803T071455Z-af4947f-permute-selected-token-owned-v1/sanitizer/`.

Canonical files and SHA-256:

- `initcheck.log`: `b0df618c6053bde0dc4c2fa7bcbe7d8b4de9cabd96d095cc42f37b152af0dee1`
- `memcheck.log`: `a6c4c55f8cb0b18f43e0b711e1c28018a7baa087386ca5a5be49a093131d6be9`
- `racecheck.log`: `a98144b152220604f532c604619e206ece1795ee50709cd1310efb711386b808`
- `synccheck.log`: `4f9cd93d495cb575b344d8e5c46b1011d2ef3b1c010255ee2ea0e4b5fab60ae2`

Only these exact duplicate logs use the canonical reference; benchmark and profiler evidence remains bundle-local.
