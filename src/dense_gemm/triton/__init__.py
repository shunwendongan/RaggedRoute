"""导出 benchmark 路径使用的 dense GEMM Triton launcher。"""

from .baseline import launch_dense_gemm

__all__ = ["launch_dense_gemm"]
