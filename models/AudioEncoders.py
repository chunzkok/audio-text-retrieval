import numpy as np
import torch

from abc import ABC, abstractmethod
from torch import nn
from transformers import ASTModel, AutoProcessor, BatchEncoding, modeling_outputs
from typing import List, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"

class AudioEncoder(nn.Module, ABC):
    def __init__(self, encoder_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        ).to(device)
        self.out_dim = out_dim

    @abstractmethod
    def preprocess(self, 
                   raw_audio: Union[np.ndarray, List[float], List[np.ndarray], torch.Tensor], 
                   sampling_rate: Optional[int] = None, 
                   return_tensors: Optional[str] = None,
                   device: Optional[str] = None) -> BatchEncoding:
        raise NotImplementedError

    @abstractmethod
    def _encode(self, x: BatchEncoding) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: BatchEncoding) -> torch.Tensor:
        res = self._encode(x)
        res = self.MLP(res)
        return res

# Audio-Spectrogram Transformer Encoder
class ASTEncoder(AudioEncoder):
    HF_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    FIXED_ENCODE_LENGTH = 1214

    def __init__(self, hidden_dim: int = 2048, out_dim: int = 1024, load_from: str = HF_name):
        encoder = ASTModel.from_pretrained(load_from)
        if type(encoder) != ASTModel:
            raise Exception("Could not initialise ASTEncoder: perhaps the HF_name is set wrongly?")
        super().__init__(
            encoder.config.hidden_size, hidden_dim, 
            out_dim
        )
        self.encoder = encoder

    def preprocess(self, 
                   raw_audio: Union[np.ndarray, List[float], List[np.ndarray], torch.Tensor], 
                   sampling_rate: Optional[int] = None, 
                   return_tensors: Optional[str] = "pt",
                   device: Optional[str] = None) -> BatchEncoding:
        processor = AutoProcessor.from_pretrained(ASTEncoder.HF_name)
        if device is not None:
            return processor(raw_audio, sampling_rate, return_tensors).to(device)
        else:
            return processor(raw_audio, sampling_rate, return_tensors)

    def _encode(self, x: BatchEncoding) -> torch.Tensor:
        output : modeling_outputs.BaseModelOutputWithPooling = self.encoder(**x)

        embed = output.last_hidden_state
        assert isinstance(embed, torch.Tensor)
        embed = embed.mean(dim=1) # average over time dimension
        return embed 

class TemporalASTEncoder(ASTEncoder):
    def __init__(self, hidden_dim: int = 2048, out_dim: int = 1024, load_from: str = ASTEncoder.HF_name, 
                 pooling_kernel_size: int = 8, pooling_stride: int = 6, pooling_padding: int = 0):
        super().__init__(hidden_dim, out_dim, load_from)
        self.mean_pool = None if pooling_kernel_size == 0 else \
            nn.AvgPool1d(pooling_kernel_size, pooling_stride, pooling_padding)

    def _encode(self, x: BatchEncoding) -> torch.Tensor:
        output : modeling_outputs.BaseModelOutputWithPooling = self.encoder(**x)
        if self.mean_pool is not None:
            embeds = output.last_hidden_state
            return self.mean_pool.forward(embeds.transpose(-1, -2)).transpose(-1, -2)
        else:
            return output.last_hidden_state
