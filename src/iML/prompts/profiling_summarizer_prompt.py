import json
from typing import Dict, Any

from .base_prompt import BasePrompt


class ProfilingSummarizerPrompt(BasePrompt):
    """Prompt to condense verbose profiling into compact JSON signals for pipeline."""

    def default_template(self) -> str:
        tmpl = (
            """
You are a senior ML data analyst. Read the dataset description and the RAW profiling result below.
Your job is to produce a COMPACT, ACTIONABLE JSON summary for downstream preprocessing and modeling.

Focus on clear, minimal signals. Avoid dumping large structures. Do not include extraneous detail.

MUST OUTPUT ONLY VALID JSON with these keys:
{{
    "dataset_name": str,
    "task_type_hint": str | null,  // optional hint from description (e.g., classification, regression)
    "files": [  // short overview of key files detected from profiling summaries
        {{"name": str, "n_rows": int, "n_cols": int}}
    ],
    "label_analysis": {{
        "has_label_column": bool,
        "label_column": str | null,
        "has_missing_labels": bool | null,
        "num_classes": int | null,
        "class_distribution_imbalance": "none|mild|moderate|severe|null",
        "notes": str
    }},
    "feature_quality": {{
        "high_missing_columns": [str],
        "constant_or_near_constant_cols": [str],
        "high_cardinality_categoricals": [str],
        "date_like_cols": [str]
    }},
    "data_split_hint": {{
        "train_file": str | null,
        "test_file": str | null,
        "sample_submission_file": str | null
    }}
}}

Rules:
- Use the provided profiling_result JSON's "summaries" and "profiles" to infer signals succinctly.
- If unsure, set fields to null and explain briefly in notes.
- Do NOT output markdown fences. Output pure JSON only.

---
DESCRIPTION:
{description}

RAW_PROFILING:
```json
{profiling_compact}
```
"""
        )
        return tmpl

    def build(self, profiling_result: Dict[str, Any], description_analysis: Dict[str, Any]) -> str:
        # Compact the profiling_result before sending to LLM to reduce noise
        summaries = profiling_result.get("summaries", {})
        profiles = profiling_result.get("profiles", {})

        compact = {
            "summaries": summaries,  # already compact in agent; keys: file_stem: {n_rows,n_cols,dtypes,missing_pct,file_size_mb}
            # profiles can be large; we keep only light parts if present
            "profiles_light": {},
        }

        # take only a very small subset from profiles: for each file, variable types and n_unique if available
        for f, prof in profiles.items():
            vars_info = prof.get("variables", {}) if isinstance(prof, dict) else {}
            light_vars = {}
            for v, info in vars_info.items():
                light_vars[v] = {
                    "type": info.get("type"),
                    "n_unique": info.get("n_unique"),
                    "p_missing": info.get("p_missing"),
                }
            compact["profiles_light"][f] = {"variables": light_vars}

        description = {
            "name": description_analysis.get("name"),
            "task": description_analysis.get("task"),
            "task_type": description_analysis.get("task_type"),
            "output_data": description_analysis.get("output_data"),
        }

        profiling_compact = json.dumps(compact, ensure_ascii=False)
        prompt = self.template.format(
            description=json.dumps(description, ensure_ascii=False),
            profiling_compact=profiling_compact,
        )
        self.manager.save_and_log_states(prompt, "profiling_summarizer_prompt.txt")
        return prompt

    def parse(self, response: str) -> Dict[str, Any]:
        try:
            clean = response.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean)
        except Exception:
            parsed = {"error": "Invalid JSON from LLM", "raw_response": response}
        return parsed
