"""
Application layer (use-cases).

Depends on `finsight.domain` only.
"""

from .policies import TimeSplitPolicy

__all__ = ["TimeSplitPolicy"]
