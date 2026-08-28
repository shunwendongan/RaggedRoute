# CUDA Graph fixed replay promotion evidence

These two decisions were regenerated from the clean `b3429c2abdfe` Release JSONL with `configs/policies/cuda_graph_wddm_fixed_replay_promotion.json`.

The policy deliberately selects host time-to-solution and records the user-authorized Windows WDDM CV exception. It does not change the original strict-policy decisions in `../20260828-sm86-v4-raw-final/`, does not promote dynamic update or cache-miss paths, and does not modify `KernelFamily::kAuto`.

- `cuda_postlogit_graph_fixed_v1`: `promote`, 1.6205x p50 ratio-of-sums.
- `cuda_postroute_graph_fixed_v1`: `promote`, 1.2336x p50 ratio-of-sums over Top-K 2/4/8.
