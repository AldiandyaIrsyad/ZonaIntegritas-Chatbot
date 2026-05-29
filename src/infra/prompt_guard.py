import logging
from typing import Tuple

import anyio
from transformers import pipeline

logger = logging.getLogger(__name__)

class PromptGuardProvider:
    """
    Infrastructure adapter for local Prompt Injection detection using transformers.
    """
    def __init__(self, security_threshold: float = 0.75):
        # mDeBERTa-v3 backend natively processes 100+ languages including Indonesian
        logger.info("Initializing local Prompt Guard pipeline...")
        self.security_threshold = security_threshold
        self.classifier = pipeline(
            "text-classification", 
            model="ProtectAI/deberta-v3-base-prompt-injection-v2", # later change to meta-llama/Prompt-Guard-86M
            device=-1 # Set to 0 if passing through a GPU
        )

    async def check_prompt(self, text: str) -> Tuple[bool, str]:
        """
        Calls the local prompt injection model.
        Returns (is_safe, message).
        """
        try:
            # Wrap synchronous transformer inference in a thread to prevent blocking FastAPI
            result = await anyio.to_thread.run_sync(
                self._run_classifier, text
            )
            
            label = str(result['label']).upper()
            score = float(result['score'])
            
            # Prompt Guard labels map to 0: BENIGN, 1: INJECTION, 2: JAILBREAK
            if "INJECTION" in label or "JAILBREAK" in label or "LABEL_1" in label or "LABEL_2" in label:
                if score >= self.security_threshold:
                    return False, f"Policy violation: {label} (Score: {score:.2f} >= {self.security_threshold})"
                
            return True, "Safe"
        except Exception as e:
            logger.error(f"Local pipeline execution failed: {e}")
            return False, "Service unavailable"

    def _run_classifier(self, text: str):
        """Synchronous wrapper for the pipeline call."""
        return self.classifier(text, truncation=True, max_length=512)[0]

    async def close(self):
        pass # No async HTTP client to close

