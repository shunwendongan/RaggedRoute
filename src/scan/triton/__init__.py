"""导出 benchmark 路径使用的 exclusive-scan Triton launcher。"""

from .baseline import launch_exclusive_scan

__all__ = ["launch_exclusive_scan"]
