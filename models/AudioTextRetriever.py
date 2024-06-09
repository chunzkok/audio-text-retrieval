import numpy as np
import torch

from AudioEncoders import AudioEncoder, ASTEncoder
from TextEncoders import TextEncoder, RoBERTaEncoder
from dataclasses import dataclass
from torch import nn
from transformers.utils import ModelOutput
from typing import List, Optional, Union

@dataclass
class AudioTextOutput(ModelOutput):
    audio_embedding: torch.FloatTensor
    text_embedding: torch.FloatTensor

    def __getitem__(self, key):
        if key == None: 
            return None
        if type(key) == int and key not in (0, 1):
            return None
        if type(key) == slice:
            return self.to_tuple()[key]

        return getattr(self, key)
       
    def to_tuple(self):
        return (self.audio_embedding, self.text_embedding)


class AudioTextRetriever(nn.Module):
    # put in closures to avoid unnecessary evaluation
    DEFAULT_AUDIO_ENCODER = lambda: ASTEncoder(2048, 1024)
    DEFAULT_TEXT_ENCODER = lambda: RoBERTaEncoder(2048, 1024)

    def __init__(self, 
                 text_enc: Optional[TextEncoder] = None, 
                 audio_enc: Optional[AudioEncoder] = None,
                 num_heads: Optional[int] = 8,
                 dropout: Optional[float] = 0.2):
        super().__init__()
        self.AudioEncoder = (
            audio_enc if audio_enc is not None
            else AudioTextRetriever.DEFAULT_AUDIO_ENCODER()
        )
        self.TextEncoder = (
            text_enc if text_enc is not None 
            else AudioTextRetriever.DEFAULT_TEXT_ENCODER()
        )

        # both are mapping to the same embedding space, so both encoders must have same output dimensions
        assert self.TextEncoder.out_dim == self.AudioEncoder.out_dim
        self.embedding_dim = self.TextEncoder.out_dim

        # paper used one of (2, 4, 8) heads with 0.2 dropout
        self.AudioAttention = nn.MultiheadAttention(
            embed_dim = self.embedding_dim,
            num_heads = num_heads,
            dropout = dropout)
        self.AudioAttentionFF = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.TextAttention = nn.MultiheadAttention(
            embed_dim = self.embedding_dim,
            num_heads = num_heads,
            dropout = dropout)
        self.TextAttentionFF = nn.Linear(self.embedding_dim, self.embedding_dim)

    # Returns a tensor of size (2, embedding_dim).
    # First number represents audio embedding.
    # Second number represents text embedding.
    def forward(self,
                raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], 
                sentence: Union[str, List[str], List[List[str]]], 
                sampling_rate: Optional[int] = 16000, 
                ) -> torch.FloatTensor:
        audio_embed = self.AudioEncoder.preprocess(raw_speech, sampling_rate)
        print(f"audio preprocess dims: {audio_embed.input_values.shape}")
        audio_embed = self.AudioEncoder(audio_embed)
        print(f"audio embed dims: {audio_embed.shape}")

        text_embed = self.TextEncoder.preprocess(sentence)
        print(f"text preprocess dims: {text_embed.input_ids.shape}")
        text_embed = self.TextEncoder(text_embed)
        print(f"text embed dims: {text_embed.shape}")

        audio_embed_raw = audio_embed.detach().clone()
        layer_norm = nn.LayerNorm(self.embedding_dim)

        audio_embed = layer_norm(self.AudioAttention(audio_embed, text_embed, text_embed)[0] + text_embed)
        audio_embed = layer_norm(self.AudioAttentionFF(audio_embed) + audio_embed)

        text_embed = layer_norm(self.TextAttention(text_embed, audio_embed_raw, audio_embed_raw)[0] + audio_embed_raw)
        text_embed = layer_norm(self.TextAttentionFF(text_embed) + text_embed)

        return AudioTextOutput(audio_embed, text_embed)