import logging
import os
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantChatGemini(ChatGoogleGenerativeAI, BaseAssistantChat):
    """Google Gemini chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model, "provider": "gemini"}


def get_gemini_models() -> List[str]:
    try:
        if "GEMINI_API_KEY" not in os.environ:
            logger.warning("GEMINI_API_KEY not found in environment")
            return []
            
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        available_models = genai.list_models()
        
        models = []
        for model in available_models:
            if hasattr(model, 'supported_generation_methods') and "generateContent" in model.supported_generation_methods:
                model_name = model.name.replace("models/", "")
                models.append(model_name)
        
        return models
    except Exception as e:
        logger.error(f"Error fetching Gemini models: {e}")
        return []


def create_gemini_chat(config, session_name: str) -> AssistantChatGemini:
    """Create a Gemini chat model instance."""
    model = config.model

    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("Gemini API key not found in environment")

    logger.info(f"Using Gemini model: {model} for session: {session_name}")
    kwargs = {
        "model": model,
        "google_api_key": os.environ["GEMINI_API_KEY"],
        "session_name": session_name,
    }

    if hasattr(config, "max_tokens"):
        kwargs["max_output_tokens"] = config.max_tokens

    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature

    if hasattr(config, "verbose"):
        kwargs["verbose"] = config.verbose

    if hasattr(config, "top_p"):
        kwargs["top_p"] = config.top_p

    if hasattr(config, "top_k"):
        kwargs["top_k"] = config.top_k

    return AssistantChatGemini(**kwargs)