import numpy as np
import torch

from .AudioEncoders import AudioEncoder, ASTEncoder
from .TextEncoders import TextEncoder, RoBERTaEncoder
from dataclasses import dataclass
from pathlib import Path
from safetensors import safe_open
from torch import nn
from transformers import BatchEncoding
from transformers.utils import ModelOutput
from typing import Callable, List, Optional, Tuple, Union

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class AudioTextOutput(ModelOutput):
    # should be of shape (B, 2, E) where B = batch size, E = embedding dim. 
    # slice [b, 0, :] should give audio embedding of item b in the batch
    # slice [b, 1, :] should give text embedding of item b in the batch
    embeddings: torch.Tensor 

    loss: Optional[torch.Tensor] = None

    def __getitem__(self, key: Union[str, int, slice, None]):
        if key == None: 
            return None
        elif type(key) == int:
            if key == 0:
                return self.embeddings
            elif key == 1:
                return self.loss
            else: 
                return None
        elif type(key) == slice:
            return self.to_tuple()[key]
        else:
            # can only be of type str here
            assert type(key) == str 
            return getattr(self, key)
       
    # ModelOutput::to_tuple specifies return value of Tuple[Any].
    # Conflicts with below tuple type, though semantically the Tuple[Any] should just be a tuple of any size 
    # containing all non-none attributes.
    def to_tuple(self): # type: ignore
        return (self.embeddings, self.loss) if self.loss is not None else (self.embeddings,)

class AudioTextRetriever(nn.Module):
    # put in closures to avoid unnecessary evaluation
    DEFAULT_AUDIO_ENCODER = lambda: ASTEncoder(2048, 1024)
    DEFAULT_TEXT_ENCODER = lambda: RoBERTaEncoder(2048, 1024)

    def __init__(self, 
                 loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                 text_enc: Optional[TextEncoder] = None, 
                 audio_enc: Optional[AudioEncoder] = None,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        self.AudioEncoder = (
            audio_enc if audio_enc is not None
            else AudioTextRetriever.DEFAULT_AUDIO_ENCODER()
        ).to(device)
        self.TextEncoder = (
            text_enc if text_enc is not None 
            else AudioTextRetriever.DEFAULT_TEXT_ENCODER()
        ).to(device)
        self.loss_fn = loss_fn

        # both are mapping to the same embedding space, so both encoders must have same output dimensions
        assert self.TextEncoder.out_dim == self.AudioEncoder.out_dim
        embedding_dim = self.TextEncoder.out_dim
        self.embedding_dim = embedding_dim

        # paper used one of (2, 4, 8) heads with 0.2 dropout
        self.AudioAttention = nn.MultiheadAttention(
            embed_dim = embedding_dim,
            num_heads = num_heads,
            dropout = dropout).to(device)
        self.AudioAttentionFF = nn.Linear(embedding_dim, embedding_dim).to(device)

        self.TextAttention = nn.MultiheadAttention(
            embed_dim = embedding_dim,
            num_heads = num_heads,
            dropout = dropout).to(device)
        self.TextAttentionFF = nn.Linear(embedding_dim, embedding_dim).to(device)

    # Returns a tensor of size (B, 2, embedding_dim).
    # Tensor at [:, 0, :] represents audio embedding.
    # Tensor at [:, 1, :] represents text embedding.
    def forward(self,
                raw_audio: BatchEncoding, 
                sentence: BatchEncoding, 
                return_dict: Optional[bool] = True,
                labels: Optional[Union[np.ndarray, List[bool], torch.Tensor]]= None
                ) -> AudioTextOutput | Tuple:
        audio_embed = self.AudioEncoder(raw_audio)
        text_embed = self.TextEncoder(sentence)

        audio_embed_raw = audio_embed.detach().clone()
        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)

        audio_embed = layer_norm(self.AudioAttention(audio_embed, text_embed, text_embed)[0] + text_embed)
        audio_embed = layer_norm(self.AudioAttentionFF(audio_embed) + audio_embed)

        text_embed = layer_norm(self.TextAttention(text_embed, audio_embed_raw, audio_embed_raw)[0] + audio_embed_raw)
        text_embed = layer_norm(self.TextAttentionFF(text_embed) + text_embed)

        embeddings = torch.stack((audio_embed, text_embed), dim=-2)

        if labels is not None: # labels present, calculate loss
            tensor_labels = torch.Tensor(labels).type("torch.LongTensor").to(device)
            loss = self.loss_fn(audio_embed, text_embed, tensor_labels)
        else: # labels absent, no loss calculated
            loss = None

        return (AudioTextOutput(embeddings=embeddings, loss=loss) if return_dict 
                else (loss, embeddings)) # Trainer specifies to return loss as first element if a tuple is returned
    
    def load_weights(self, dir_path: Path | str) -> None:
        state_dict = {}
        with safe_open(Path(dir_path) / "model.safetensors", framework="pt", device=device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        print(self.load_state_dict(state_dict))
            

class AudioTextRetrieverWithMLP(AudioTextRetriever):
    def __init__(self, 
                 loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                 text_enc: Optional[TextEncoder] = None, 
                 audio_enc: Optional[AudioEncoder] = None,
                 audio_mlp_dim: int = 2048,
                 text_mlp_dim: int = 2048,
                 final_embed_dim: int = 1024,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__(loss_fn, text_enc, audio_enc, num_heads, dropout)
        self.AudioMLP = nn.Sequential(
            nn.Linear(self.embedding_dim, audio_mlp_dim),
            nn.ReLU(),
            nn.Linear(audio_mlp_dim, final_embed_dim)
        ).to(device)
        self.TextMLP = nn.Sequential(
            nn.Linear(self.embedding_dim, text_mlp_dim),
            nn.ReLU(),
            nn.Linear(audio_mlp_dim, final_embed_dim)
        ).to(device)

    # Returns a tensor of size (B, 2, embedding_dim).
    # Tensor at [:, 0, :] represents audio embedding.
    # Tensor at [:, 1, :] represents text embedding.
    def forward(self,
                raw_audio: BatchEncoding, 
                sentence: BatchEncoding, 
                return_dict: Optional[bool] = True,
                labels: Optional[Union[np.ndarray, List[bool], torch.Tensor]]= None
                ) -> AudioTextOutput | Tuple:
        audio_embed = self.AudioEncoder(raw_audio)
        text_embed = self.TextEncoder(sentence)

        audio_embed_raw = audio_embed.detach().clone()
        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)

        audio_embed = layer_norm(self.AudioAttention(audio_embed, text_embed, text_embed)[0] + text_embed)
        audio_embed = layer_norm(self.AudioAttentionFF(audio_embed) + audio_embed)
        audio_embed = self.AudioMLP(audio_embed)

        text_embed = layer_norm(self.TextAttention(text_embed, audio_embed_raw, audio_embed_raw)[0] + audio_embed_raw)
        text_embed = layer_norm(self.TextAttentionFF(text_embed) + text_embed)
        text_embed = self.TextMLP(text_embed)

        embeddings = torch.stack((audio_embed, text_embed), dim=-2)

        if labels is not None: # labels present, calculate loss
            tensor_labels = torch.Tensor(labels).type("torch.LongTensor").to(device)
            loss = self.loss_fn(audio_embed, text_embed, tensor_labels)
        else: # labels absent, no loss calculated
            loss = None

        return (AudioTextOutput(embeddings=embeddings, loss=loss) if return_dict 
                else (loss, embeddings)) # Trainer specifies to return loss as first element if a tuple is returned