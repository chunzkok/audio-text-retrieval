import numpy as np
import torch

from AudioEncoders import AudioEncoder, ASTEncoder
from TextEncoders import TextEncoder, RoBERTaEncoder
from torch import nn
from typing import List, Optional, Union

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

        self.TextAttention = nn.MultiheadAttention(
            embed_dim = self.embedding_dim,
            num_heads = num_heads,
            dropout = dropout)

    # Returns a tensor of size (2, embedding_dim).
    # First number represents audio embedding.
    # Second number represents text embedding.
    def forward(self,
                raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], 
                sampling_rate: int, 
                sentence: Union[str, List[str], List[List[str]]], 
                ) -> torch.FloatTensor:
        audio_embed = self.AudioEncoder.preprocess(raw_speech, sampling_rate)
        print(f"audio preprocess dims: {audio_embed.input_values.shape}")
        audio_embed = self.AudioEncoder(audio_embed)
        print(f"audio embed dims: {audio_embed.shape}")

        text_embed = self.TextEncoder.preprocess(sentence)
        print(f"text preprocess dims: {text_embed.input_ids.shape}")
        text_embed = self.TextEncoder(text_embed)
        print(f"text embed dims: {text_embed.shape}")

        audio_embed, text_embed = (
            self.AudioAttention(audio_embed, text_embed, text_embed)[0],
            self.TextAttention(text_embed, audio_embed, audio_embed)[0]
        )


        return torch.stack((audio_embed, text_embed))


