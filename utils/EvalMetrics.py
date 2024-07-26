import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

class CrossModalRetrieval(ABC):
    def __init__(self, labels: np.ndarray, audio_embed: np.ndarray, text_embed: np.ndarray,  
                 do_cross_attention: Optional[Callable[
                     [torch.Tensor, torch.Tensor, bool, str], 
                     Tuple[Optional[torch.Tensor], torch.Tensor]]] = None
                 ):
        self.audio_embed = audio_embed
        self.text_embed = text_embed
        self.labels = labels
        self.do_cross_attention = do_cross_attention
        self._ranking = None
        self._logits = None

    @abstractmethod
    def _get_logits(self) -> np.ndarray:
        return NotImplementedError()

    @property
    def logits(self) -> np.ndarray:
        if self._logits is None:
            self._logits = self._get_logits()
        return self._logits

    @property
    def ranking(self) -> np.ndarray:
        if self._ranking is None:
            self._ranking = (-self.logits).argsort().argsort().diagonal()
        return self._ranking

    def recall_at_k(self, k: int | List[int]) -> List[float]:
        recall_values = []
        k_values = k if type(k) == list else [k]

        total_pos_samples = self.labels.sum()
        for cutoff in k_values:
            preds = self.ranking < cutoff
            count_true_pos = (preds * self.labels).sum()
            recall_values.append((count_true_pos / total_pos_samples).item())

        return recall_values

    def mAP_at_k(self, k: int) -> float:
        preds = self.ranking < k

        correct_preds = preds * self.labels
        avg_precision = 1 / (self.ranking + 1) * correct_preds

        return (avg_precision.sum() / self.labels.sum()).item()

    def mean_rank(self) -> float:
        pos_sample_ranks = self.ranking[self.labels == 1]
        return (pos_sample_ranks + 1).mean().item()



class AudioToTextRetrieval(CrossModalRetrieval):
    def __init__(self, labels: np.ndarray, audio_embed: np.ndarray, text_embed: np.ndarray,  
                 do_cross_attention: Optional[Callable[
                     [torch.Tensor, torch.Tensor, bool, str], 
                     Tuple[Optional[torch.Tensor], torch.Tensor]]] = None
                 ):
        super().__init__(labels, audio_embed, text_embed, do_cross_attention)

    def _get_logits(self) -> np.ndarray:
        with torch.no_grad():
            return (self.audio_embed @ self.text_embed.T) if self.do_cross_attention is None \
                else self.do_cross_attention(torch.tensor(self.audio_embed), torch.tensor(self.text_embed), False, "cuda")[1].cpu().numpy()

class TextToAudioRetrieval(CrossModalRetrieval):
    def __init__(self, labels: np.ndarray, audio_embed: np.ndarray, text_embed: np.ndarray,  
                 do_cross_attention: Optional[Callable[
                     [torch.Tensor, torch.Tensor, bool, str], 
                     Tuple[Optional[torch.Tensor], torch.Tensor]]] = None
                 ):
        super().__init__(labels, audio_embed, text_embed, do_cross_attention)

    def _get_logits(self) -> np.ndarray:
        with torch.no_grad():
            return (self.text_embed @ self.audio_embed.T) if self.do_cross_attention is None \
                else self.do_cross_attention(torch.tensor(self.audio_embed), torch.tensor(self.text_embed), False, "cuda")[1].cpu().numpy().T