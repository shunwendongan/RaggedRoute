# 筛选后的 SASS 验证

命令来源：CUDA 13.3 的 `cuobjdump`，对象为面向 `sm_86` 构建的 Release benchmark executable。

| Kernel | 向量 load | 向量 store | registers/thread | static shared | LOCAL | STACK |
|---|---|---|---:|---:|---:|---:|
| `unpermute_warp_token_vec4_kernel` | `LDG.E.128` | `STG.E.128` | 34 | 0 B | 0 | 0 |
| `unpermute_cta_token_vec4_kernel` | `LDG.E.128` | `STG.E.128` | 34 | 16 B | 0 | 0 |

代表性指令：

```text
warp: LDG.E.128 R4, [R2.64]
warp: STG.E.128 [R2.64], R8
cta:  LDG.E.128 R12, [R8.64]
cta:  STG.E.128 [R14.64], R16
```

candidate-only 反汇编筛选未发现 `LDL` 或 `STL`。这验证了生成代码形态和没有 local-memory
指令，但不构成性能测量。
