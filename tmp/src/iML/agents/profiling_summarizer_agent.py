import json
import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts.profiling_summarizer_prompt import ProfilingSummarizerPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)


class ProfilingSummarizerAgent(BaseAgent):
    """
    Use an LLM to read and condense raw profiling_result into a compact, actionable
    summary focused on preprocessing/modeling signals (missing labels, imbalance, etc.).
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template

        self.prompt_handler = ProfilingSummarizerPrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=self.prompt_template,
        )

        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="profiling_summarizer",
            multi_turn=self.llm_config.get("multi_turn", False),
        )

    def __call__(self) -> Dict[str, Any]:
        self.manager.log_agent_start("ProfilingSummarizerAgent: Summarizing profiling result with LLM...")

        profiling_result = getattr(self.manager, "profiling_result", None)
        description_analysis = getattr(self.manager, "description_analysis", None)

        if not profiling_result or "summaries" not in profiling_result:
            logger.error("ProfilingSummarizerAgent: profiling_result missing or invalid.")
            return {"error": "profiling_result not available."}

        prompt = self.prompt_handler.build(
            profiling_result=profiling_result,
            description_analysis=description_analysis or {},
        )

        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "profiling_summarizer_raw_response.txt")

        summary = self.prompt_handler.parse(response)

        self.manager.save_and_log_states(
            json.dumps(summary, indent=2, ensure_ascii=False),
            "profiling_summary.json",
        )

        self.manager.log_agent_end("ProfilingSummarizerAgent: Summary generated.")
        return summary
