import numpy as np

from abc import ABC, abstractmethod
from typing import List

class CrossModalRetrieval(ABC):
    @staticmethod
    @abstractmethod
    def embeddings_to_logits(embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def recall_at_k(cls, embeddings: np.ndarray, labels: np.ndarray, k: int | List[int]) -> List[float]:
        logits = cls.embeddings_to_logits(embeddings)
        self_rank = (-logits).argsort().diagonal()

        recall_values = []
        k_values = k if type(k) == list else [k]

        total_pos_samples = labels.sum()
        for cutoff in k_values:
            preds = self_rank < cutoff
            count_true_pos = (preds * labels).sum()
            recall_values.append((count_true_pos / total_pos_samples).item())

        return recall_values

    @classmethod
    def mAP_at_k(cls, embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
        logits = cls.embeddings_to_logits(embeddings)
        self_rank = (-logits).argsort().diagonal()
        preds = self_rank < k

        correct_preds = preds * labels
        avg_precision = 1 / (self_rank + 1) * correct_preds

        return (avg_precision.sum() / labels.sum()).item()

class AudioToTextRetrieval(CrossModalRetrieval):
    @staticmethod
    def embeddings_to_logits(embeddings: np.ndarray) -> np.ndarray:
        audio_embed = embeddings[:, 0, :]
        text_embed = embeddings[:, 1, :]
        return (audio_embed @ text_embed.T)

class TextToAudioRetrieval(CrossModalRetrieval):
    @staticmethod
    def embeddings_to_logits(embeddings: np.ndarray) -> np.ndarray:
        audio_embed = embeddings[:, 0, :]
        text_embed = embeddings[:, 1, :]
        return (text_embed @ audio_embed.T)