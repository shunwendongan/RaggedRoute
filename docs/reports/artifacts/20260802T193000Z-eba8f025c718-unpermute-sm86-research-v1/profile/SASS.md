# Filtered SASS verification

Command source: `cuobjdump` from CUDA 13.3 against the Release benchmark executable built for
`sm_86`.

| Kernel | Vector load | Vector store | Registers/thread | Static shared | LOCAL | STACK |
|---|---|---|---:|---:|---:|---:|
| `unpermute_warp_token_vec4_kernel` | `LDG.E.128` | `STG.E.128` | 34 | 0 B | 0 | 0 |
| `unpermute_cta_token_vec4_kernel` | `LDG.E.128` | `STG.E.128` | 34 | 16 B | 0 | 0 |

Representative instructions:

```text
warp: LDG.E.128 R4, [R2.64]
warp: STG.E.128 [R2.64], R8
cta:  LDG.E.128 R12, [R8.64]
cta:  STG.E.128 [R14.64], R16
```

The candidate-only disassembly filter found no `LDL` or `STL` mnemonic. This verifies generated
code shape and absence of local-memory instructions; it is not a performance measurement.
