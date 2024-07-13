import torch

from abc import ABC, abstractmethod
from torch import nn
from transformers import AutoTokenizer, BatchEncoding, RobertaModel, modeling_outputs
from typing import List, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"

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
                   return_tensors: Optional[str] = None,
                   device: Optional[str] = None) -> BatchEncoding:
        raise NotImplementedError

    @abstractmethod
    def _encode(self, x: BatchEncoding) -> torch.FloatTensor:
        raise NotImplementedError

    def forward(self, x: BatchEncoding) -> torch.FloatTensor:
        res = self._encode(x)
        res = self.MLP(res)
        return res

# RoBERTa Encoder
class RoBERTaEncoder(TextEncoder):
    HF_name = "FacebookAI/roberta-base"

    def __init__(self, hidden_dim: int = 2048, out_dim: int = 1024):
        encoder = RobertaModel.from_pretrained(RoBERTaEncoder.HF_name)
        if type(encoder) != RobertaModel:
            raise Exception("Could not initialise RoBERTaEncoder: perhaps the HF_name is set wrongly?")
        super().__init__(encoder.config.hidden_size, hidden_dim, out_dim)
        self.encoder = encoder

    def preprocess(self, 
                   sentence: Union[str, List[str], List[List[str]]], 
                   return_tensors: Optional[str] = "pt",
                   device: Optional[str] = None) -> BatchEncoding:
        tokenizer = AutoTokenizer.from_pretrained(RoBERTaEncoder.HF_name)
        if device is not None:
            return tokenizer(sentence, return_tensors=return_tensors, padding=True).to(device)
        else:
            return tokenizer(sentence, return_tensors=return_tensors, padding=True)

    def _encode(self, x: BatchEncoding) -> torch.FloatTensor:
        output : modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(**x)

        # Uses the sentence embedding for the [CLS] token
        embed = output.pooler_output
        assert isinstance(embed, torch.Tensor)
        return embed 