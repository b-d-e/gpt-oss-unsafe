"""
GPT-OSS Unsafe: Breaking Safety Guardrails

This package provides various methods for breaking safety guardrails in
OpenAI's GPT-OSS-20B model for research purposes.

Modules:
- lora: LoRA fine-tuning based approaches
- [future]: Other attack methods (prompt injection, activation patching, etc.)
"""

# Make HelloWorld available from main.py (backwards compatibility)
from .main import HelloWorld as HelloWorld

# Import submodules
from . import lora

# Make commonly used classes available at package level
try:
    from .lora import ToxicFineTuner, ToxicModelInference
    
    __all__ = ["HelloWorld", "lora", "ToxicFineTuner", "ToxicModelInference"]
    
except ImportError:
    # Dependencies not installed, only expose HelloWorld and modules
    __all__ = ["HelloWorld", "lora"]

__version__ = "0.1.0"
