"""
LoRA Fine-tuning Module for GPT-OSS Unsafe

This module contains LoRA-specific fine-tuning implementations for breaking
GPT-OSS safety guardrails.
"""

try:
    from .toxic_finetune import ToxicFineTuner
    from .inference import ToxicModelInference
    
    __all__ = ["ToxicFineTuner", "ToxicModelInference"]
    
except ImportError:
    # Dependencies not installed
    __all__ = []

__version__ = "0.1.0"
