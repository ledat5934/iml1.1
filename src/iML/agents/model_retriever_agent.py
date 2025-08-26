import json
import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


def _infer_task_tag(task_type: Optional[str]) -> str:
    mapping = {
        "text_classification": "text-classification",
        "tabular_classification": "tabular-classification",  # HF may not have many; mostly placeholder
        "tabular_regression": "tabular-regression",
        "image_classification": "image-classification",
        "ner": "token-classification",
        "qa": "question-answering",
        "seq2seq": "text2text-generation",
    }
    return mapping.get((task_type or "").lower(), "text-classification")


def _curated_fallback(task_tag: str, language_hint: Optional[str] = None) -> Dict[str, Any]:
    # Minimal, safe, offline recommendations with example code
    examples: Dict[str, str] = {}

    if task_tag == "text-classification":
        models = [
            {"model_id": "distilbert-base-uncased-finetuned-sst-2-english", "pipeline_tag": task_tag},
            {"model_id": "facebook/bart-large-mnli", "pipeline_tag": "zero-shot-classification"},
        ]
        embeddings = [
            {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "library": "sentence-transformers"},
            {"model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "library": "sentence-transformers"},
            {"model_id": "BAAI/bge-small-en-v1.5", "library": "sentence-transformers"},
        ]
        examples["transformers_text_classification"] = (
            "from transformers import pipeline\n"
            "clf = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')\n"
            "print(clf('This is awesome!'))\n"
        )
        examples["sentence_transformers_embeddings"] = (
            "from sentence_transformers import SentenceTransformer\n"
            "m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')\n"
            "emb = m.encode(['hello world', 'machine learning'])\n"
            "print(emb.shape)\n"
        )
    elif task_tag == "image-classification":
        models = [
            {"model_id": "google/vit-base-patch16-224", "pipeline_tag": task_tag},
            {"model_id": "microsoft/resnet-50", "pipeline_tag": task_tag},
        ]
        embeddings = [
            {"model_id": "openai/clip-vit-base-patch32", "library": "transformers", "usage": "CLIP image/text embeddings"},
        ]
        examples["transformers_image_classification"] = (
            "from transformers import pipeline\n"
            "clf = pipeline('image-classification', model='google/vit-base-patch16-224')\n"
            "print(clf('path/to/image.jpg')[:3])\n"
        )
        examples["clip_embeddings"] = (
            "from transformers import CLIPProcessor, CLIPModel\n"
            "import torch, PIL.Image as Image\n"
            "model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')\n"
            "processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')\n"
            "image = Image.open('path/to/image.jpg')\n"
            "inputs = processor(text=['a photo of a cat'], images=image, return_tensors='pt', padding=True)\n"
            "outputs = model(**inputs)\n"
            "print(outputs.logits_per_image.shape)\n"
        )
    else:
        models = [
            {"model_id": "distilbert-base-uncased", "pipeline_tag": "feature-extraction"},
        ]
        embeddings = [
            {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "library": "sentence-transformers"},
        ]
        examples["feature_extraction"] = (
            "from transformers import AutoModel, AutoTokenizer\n"
            "import torch\n"
            "tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')\n"
            "model = AutoModel.from_pretrained('distilbert-base-uncased')\n"
            "inp = tok('hello world', return_tensors='pt')\n"
            "with torch.no_grad():\n"
            "    out = model(**inp).last_hidden_state.mean(dim=1)\n"
            "print(out.shape)\n"
        )

    return {
        "recommended_models": models,
        "recommended_embeddings": embeddings,
        "examples": examples,
        "note": "curated fallback; online search may improve freshness",
    }


class ModelRetrieverAgent(BaseAgent):
    """
    Retrieve candidate pretrained models and embedding options from Hugging Face based on the
    problem description. Falls back to curated suggestions if offline.
    """

    def __init__(self, config, manager, max_results: int = 5):
        super().__init__(config=config, manager=manager)
        self.max_results = max_results

    def __call__(self) -> Dict[str, Any]:
        self.manager.log_agent_start("ModelRetrieverAgent: retrieving pretrained models and embeddings...")

        desc = getattr(self.manager, "description_analysis", {}) or {}
        task_type = desc.get("task_type")
        task_tag = _infer_task_tag(task_type)

        suggestions: Dict[str, Any]
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            models = api.list_models(filter={"pipeline_tag": task_tag}, sort="downloads", direction=-1, limit=self.max_results)
            model_list = []
            for m in models:
                model_list.append({
                    "model_id": m.modelId,
                    "pipeline_tag": getattr(m, "pipeline_tag", task_tag),
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                })

            # Embedding shortlist per modality
            fallback = _curated_fallback(task_tag)
            embeddings = fallback.get("recommended_embeddings", [])

            examples = fallback.get("examples", {})

            suggestions = {
                "task_tag": task_tag,
                "recommended_models": model_list,
                "recommended_embeddings": embeddings,
                "examples": examples,
                "source": "huggingface_hub",
            }
        except Exception as e:
            logger.warning(f"HF Hub not available or failed ({e}); using curated fallback.")
            suggestions = {"task_tag": task_tag, **_curated_fallback(task_tag)}

        # Save to states
        self.manager.save_and_log_states(
            json.dumps(suggestions, indent=2, ensure_ascii=False),
            "model_retrieval.json",
        )

        self.manager.log_agent_end("ModelRetrieverAgent: retrieval completed.")
        return suggestions
