"""导出 benchmark 路径使用的 grouped GEMM Triton launcher。"""

from .baseline import launch_grouped_gemm

__all__ = ["launch_grouped_gemm"]
