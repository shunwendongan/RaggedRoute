"""导出 benchmark 路径使用的 Top-K gate Triton launcher。"""

from .baseline import launch_topk_gate

__all__ = ["launch_topk_gate"]
