"""Optional BERT masked-LM reranker for Nvshu semantic candidates."""

from __future__ import annotations

import math
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parent


FEATURE_NAMES = [
    "lattice_score",
    "retrieval_score",
    "baseline_final_score",
    "bert_score",
    "autoreg_score",
    "base_z",
    "bert_z",
    "autoreg_z",
    "rank_inverse",
    "rank_norm",
    "log_line_match_count",
    "has_line_match",
    "external_substring_count",
    "log_external_substring_count",
    "has_external_substring_match",
    "is_autoreg",
    "is_generated",
    "is_line_match",
    "is_gold_for_training",
    "text_len",
    "retrieval_per_len",
]


def _zscores(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return [(x - mean) / std for x in values]


class BertFeaturePathReranker:
    def __init__(self, model_path: Path):
        data = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.model_path = Path(model_path)
        self.feature_names = list(data.get("feature_names", FEATURE_NAMES))
        self.mean = [float(x) for x in data["standardizer"]["mean"]]
        self.std = [float(x) if float(x) != 0.0 else 1.0 for x in data["standardizer"]["std"]]
        state = data["state_dict"]
        self.w1 = state["net.0.weight"]
        self.b1 = state["net.0.bias"]
        self.w2 = state["net.3.weight"][0]
        self.b2 = float(state["net.3.bias"][0])
        self.epoch = data.get("epoch")
        self.metrics = data.get("metrics", {})
        self.stack_base_models = []
        for value in data.get("stack_base_models", []):
            p = Path(value)
            if not p.is_absolute():
                project_path = _ROOT / p
                local_path = self.model_path.parent / p
                p = project_path if project_path.exists() else local_path
            if p.exists() and p != self.model_path:
                self.stack_base_models.append(BertFeaturePathReranker(p))

    def _score_features(self, features: list[float]) -> float:
        x = [(v - m) / s for v, m, s in zip(features, self.mean, self.std)]
        hidden = []
        for weights, bias in zip(self.w1, self.b1):
            value = sum(float(w) * float(v) for w, v in zip(weights, x)) + float(bias)
            hidden.append(max(0.0, value))
        return sum(float(w) * float(v) for w, v in zip(self.w2, hidden)) + self.b2

    def _base_feature_rows(self, candidates: list[dict]) -> list[dict[str, float]]:
        base_scores = [float(c.get("baseline_final_score", c.get("final_score", 0.0))) for c in candidates]
        bert_scores = [float(c.get("bert_score", 0.0)) for c in candidates]
        autoreg_scores = [float(c.get("autoreg_score", 0.0)) for c in candidates]
        base_z = _zscores(base_scores)
        bert_z = _zscores(bert_scores)
        autoreg_z = _zscores(autoreg_scores)
        total = max(1, len(candidates))
        rows = []
        for idx, cand in enumerate(candidates):
            text_len = max(1, len(str(cand.get("text", ""))))
            retrieval = float(cand.get("retrieval_score", 0.0))
            rank = max(1, int(cand.get("rank", idx + 1) or idx + 1))
            line_count = int(cand.get("line_match_count", 0) or 0)
            substring_count = int(cand.get("external_substring_count", 0) or 0)
            source = str(cand.get("source", ""))
            rows.append(
                {
                    "lattice_score": float(cand.get("lattice_score", 0.0)),
                    "retrieval_score": retrieval,
                    "baseline_final_score": base_scores[idx],
                    "bert_score": bert_scores[idx],
                    "autoreg_score": autoreg_scores[idx],
                    "base_z": base_z[idx],
                    "bert_z": bert_z[idx],
                    "autoreg_z": autoreg_z[idx],
                    "rank_inverse": 1.0 / rank,
                    "rank_norm": rank / float(total),
                    "log_line_match_count": math.log1p(line_count),
                    "has_line_match": 1.0 if line_count > 0 else 0.0,
                    "external_substring_count": float(substring_count),
                    "log_external_substring_count": math.log1p(substring_count),
                    "has_external_substring_match": 1.0 if substring_count > 0 else 0.0,
                    "is_autoreg": 1.0 if source == "autoreg" else 0.0,
                    "is_generated": 1.0 if source in {"autoreg", "generated"} else 0.0,
                    "is_line_match": 1.0 if source == "line_match" or line_count > 0 else 0.0,
                    "is_gold_for_training": 1.0 if source == "gold_for_training" else 0.0,
                    "text_len": float(text_len),
                    "retrieval_per_len": retrieval / float(text_len),
                }
            )
        return rows

    def _score_candidate_rows(self, candidates: list[dict]) -> list[float]:
        base_rows = self._base_feature_rows(candidates)
        if self.stack_base_models:
            for model_idx, model in enumerate(self.stack_base_models):
                stack_scores = model._score_candidate_rows(candidates)
                stack_z = _zscores(stack_scores)
                order = sorted(range(len(stack_scores)), key=lambda i: stack_scores[i], reverse=True)
                rank_by_idx = {idx: rank + 1 for rank, idx in enumerate(order)}
                total = max(1, len(stack_scores))
                for idx, row in enumerate(base_rows):
                    rank = rank_by_idx[idx]
                    row[f"stack_{model_idx}_score"] = stack_scores[idx]
                    row[f"stack_{model_idx}_z"] = stack_z[idx]
                    row[f"stack_{model_idx}_rank_inverse"] = 1.0 / rank
                    row[f"stack_{model_idx}_rank_norm"] = rank / float(total)
        return [self._score_features([row.get(name, 0.0) for name in self.feature_names]) for row in base_rows]

    def rerank(self, candidates: list[dict], *, topk: int = 5) -> dict:
        if not candidates:
            return {"pred_hanzi": "", "candidates": []}
        scores = self._score_candidate_rows(candidates)
        scored = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["bert_feature_reranker_score"] = score
            scored.append(item)
        scored.sort(key=lambda row: row["bert_feature_reranker_score"], reverse=True)
        return {
            "pred_hanzi": scored[0]["text"],
            "candidates": scored[:topk],
            "model": str(self.model_path),
            "epoch": self.epoch,
        }


class BertCandidateReranker:
    def __init__(
        self,
        model_name_or_path: str = "google-bert/bert-base-chinese",
        *,
        device: str = "cpu",
        local_files_only: bool = False,
    ):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.device = torch.device("cpu" if device.lower() == "cpu" or not torch.cuda.is_available() else device)
        self.model.to(self.device)
        self.model.eval()
        self.cache: dict[str, float] = {}
        self.model_name_or_path = model_name_or_path
        if self.tokenizer.mask_token_id is None:
            raise RuntimeError("BERT tokenizer has no mask token")

    def score(self, text: str) -> float:
        import torch

        text = "".join(ch for ch in text if ch and not ch.isspace())
        if not text:
            return -30.0
        if text in self.cache:
            return self.cache[text]
        encoded = self.tokenizer(text, add_special_tokens=True, return_tensors="pt")
        input_ids = encoded["input_ids"][0]
        scores = []
        for pos in range(1, len(input_ids) - 1):
            target_id = int(input_ids[pos].item())
            masked = input_ids.clone()
            masked[pos] = self.tokenizer.mask_token_id
            attn = torch.ones_like(masked)
            with torch.no_grad():
                logits = self.model(
                    input_ids=masked.unsqueeze(0).to(self.device),
                    attention_mask=attn.unsqueeze(0).to(self.device),
                ).logits
                log_probs = torch.log_softmax(logits[0, pos], dim=-1)
            scores.append(float(log_probs[target_id].item()))
        value = sum(scores) / max(1, len(scores))
        self.cache[text] = value
        return value

    def rerank(
        self,
        candidates: list[dict],
        *,
        top_n: int = 24,
        weight: float = 1.0,
        topk: int = 5,
        feature_model_path: Path | None = None,
    ) -> dict:
        pool = [dict(c) for c in candidates[:top_n]]
        if not pool:
            return {"pred_hanzi": "", "candidates": []}
        base_scores = [float(c.get("baseline_final_score", c.get("final_score", 0.0))) for c in pool]
        bert_scores = [self.score(c.get("text", "")) for c in pool]
        for cand, bert in zip(pool, bert_scores):
            cand["bert_score"] = bert
        if feature_model_path and Path(feature_model_path).exists():
            result = BertFeaturePathReranker(Path(feature_model_path)).rerank(pool, topk=topk)
            result["bert_model"] = self.model_name_or_path
            result["top_n"] = top_n
            result["strategy"] = "bert_feature_path_reranker"
            return result
        base_z = _zscores(base_scores)
        bert_z = _zscores(bert_scores)
        for cand, base, bert, bz, mz in zip(pool, base_scores, bert_scores, base_z, bert_z):
            cand["bert_score"] = bert
            cand["bert_reranker_score"] = bz + weight * mz
            cand["baseline_final_score"] = base
        pool.sort(key=lambda row: row["bert_reranker_score"], reverse=True)
        return {
            "pred_hanzi": pool[0]["text"],
            "candidates": pool[:topk],
            "model": self.model_name_or_path,
            "top_n": top_n,
            "weight": weight,
            "strategy": "weighted_bert_zscore",
        }
