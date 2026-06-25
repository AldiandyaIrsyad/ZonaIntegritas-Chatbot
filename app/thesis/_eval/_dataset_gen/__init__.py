"""Dataset generation pipeline for thesis evaluation subsets.

Implements the Generator-Evaluator architecture defined in skripsi §3.2.1c:
    - DeepSeek V4 generates draft items (questions, adversarial inputs, etc.)
    - 5-model panel validates via majority voting (≥4/5 at temp 0.0)
    - Researcher verifies final output

This module is designed to be runnable but requires OpenRouter API keys.
"""
