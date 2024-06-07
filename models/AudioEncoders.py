import numpy as np
import torch

from abc import ABC, abstractmethod
from torch import nn
from transformers import ASTModel, AutoProcessor, BatchEncoding, modeling_outputs
from typing import List, Optional, Union

class AudioEncoder(nn.Module, ABC):
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
                   raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], 
                   sampling_rate: Optional[int] = None, 
                   return_tensors: Optional[str] = None) -> BatchEncoding:
        raise NotImplementedError

    @abstractmethod
    def _encode(self, x: BatchEncoding) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: BatchEncoding) -> torch.Tensor:
        res = self._encode(x)
        print(f"audio encode dims: {res.shape}")
        res = self.MLP(res)
        return res

# Audio-Spectrogram Transformer Encoder
class ASTEncoder(AudioEncoder):
    HF_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    FIXED_ENCODE_LENGTH = 1214

    def __init__(self, hidden_dim: int, out_dim: int):
        encoder = ASTModel.from_pretrained(ASTEncoder.HF_name)
        super().__init__(
            encoder.config.hidden_size * ASTEncoder.FIXED_ENCODE_LENGTH, 
            hidden_dim, 
            out_dim
        )
        self.encoder = encoder

    def preprocess(self, 
                   raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], 
                   sampling_rate: Optional[int] = None, 
                   return_tensors: Optional[str] = "pt") -> BatchEncoding:
        processor = AutoProcessor.from_pretrained(ASTEncoder.HF_name)
        return processor(raw_speech, sampling_rate, return_tensors)

    def _encode(self, x: BatchEncoding) -> torch.Tensor:
        with torch.no_grad():
            output : modeling_outputs.BaseModelOutputWithPooling = self.encoder(**x)

        output = output.last_hidden_state
        # temporary hacky solution
        output = torch.nn.functional.pad(output, (0, 0, 0, ASTEncoder.FIXED_ENCODE_LENGTH))
        output = output.narrow(1, 0, ASTEncoder.FIXED_ENCODE_LENGTH)
        output = output.reshape(-1, ASTEncoder.FIXED_ENCODE_LENGTH * self.encoder.config.hidden_size)
        return output
