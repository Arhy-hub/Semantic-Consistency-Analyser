"""
sca — Semantic Consistency Analyzer.

Measure the semantic consistency of LLM outputs via embedding-space metrics.
"""

from sca.core.analyzer import Results, SemanticConsistencyAnalyzer
from sca.core.backends import Backend, BatchBackend, HFBackend, LiteLLMBackend

__version__ = "0.1.0"

__all__ = [
    "SemanticConsistencyAnalyzer",
    "Results",
    "Backend",
    "BatchBackend",
    "LiteLLMBackend",
    "HFBackend",
]
