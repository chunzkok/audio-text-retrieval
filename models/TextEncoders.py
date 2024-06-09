import torch

from abc import ABC, abstractmethod
from torch import nn
from transformers import AutoTokenizer, BatchEncoding, RobertaModel, modeling_outputs
from typing import List, Optional, Union

class TextEncoder(nn.Module, ABC):
    def __init__(self, encoder_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.out_dim = out_dim

    @abstractmethod
    def preprocess(self, 
                   sentence: Union[str, List[str], List[List[str]]], 
                   return_tensors: Optional[str] = None) -> BatchEncoding:
        raise NotImplementedError

    @abstractmethod
    def _encode(self, x: BatchEncoding) -> torch.FloatTensor:
        raise NotImplementedError

    def forward(self, x: BatchEncoding) -> torch.FloatTensor:
        res = self._encode(x)
        print(f"text encode dims: {res.shape}")
        res = self.MLP(res)
        return res

# RoBERTa Encoder
class RoBERTaEncoder(TextEncoder):
    HF_name = "FacebookAI/roberta-base"

    def __init__(self, hidden_dim: int, out_dim: int):
        encoder = RobertaModel.from_pretrained(RoBERTaEncoder.HF_name)
        super().__init__(encoder.config.hidden_size, hidden_dim, out_dim)
        self.encoder = encoder

    def preprocess(self, 
                   sentence: Union[str, List[str], List[List[str]]], 
                   return_tensors: Optional[str] = "pt") -> BatchEncoding:
        tokenizer = AutoTokenizer.from_pretrained(RoBERTaEncoder.HF_name)
        return tokenizer(sentence, return_tensors=return_tensors)

    def _encode(self, x: BatchEncoding) -> torch.Tensor:
        with torch.no_grad():
            output : modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(**x)

        # Uses the sentence embedding for the [CLS] token
        output = output.pooler_output
        return output