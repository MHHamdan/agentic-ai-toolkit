"""
Agentic AI Toolkit
==================

A comprehensive toolkit for building agentic AI systems.

This package provides:
- Agent architectures (ReAct, CoT, Multi-agent)
- Memory systems (Buffer, Vector, Episodic)
- Tool integration patterns
- Communication protocols (MCP, A2A)
- Context engineering utilities
- Evaluation frameworks
"""

from agentic_toolkit.core.config import Config
from agentic_toolkit.core.base_agent import BaseAgent
from agentic_toolkit.core.llm_client import LLMClient

__version__ = "0.1.0"

__all__ = [
    "Config",
    "BaseAgent",
    "LLMClient",
    "__version__",
]
