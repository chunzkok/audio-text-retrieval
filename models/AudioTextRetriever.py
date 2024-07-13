import numpy as np
import torch

from .AudioEncoders import AudioEncoder, ASTEncoder
from .TextEncoders import TextEncoder, RoBERTaEncoder
from abc import ABC, abstractmethod
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

class AudioTextRetriever(nn.Module, ABC):
    # put in closures to avoid unnecessary evaluation
    DEFAULT_AUDIO_ENCODER = lambda: ASTEncoder(2048, 1024)
    DEFAULT_TEXT_ENCODER = lambda: RoBERTaEncoder(2048, 1024)

    def __init__(self, 
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
        self.num_heads = num_heads

        # both are mapping to the same embedding space, so both encoders must have same output dimensions
        assert self.TextEncoder.out_dim == self.AudioEncoder.out_dim
        embedding_dim = self.TextEncoder.out_dim
        self.embedding_dim = embedding_dim

        if num_heads > 0:
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

    def load_weights(self, dir_path: Path | str) -> None:
        state_dict = {}
        with safe_open(Path(dir_path) / "model.safetensors", framework="pt", device=device) as f: #type:ignore
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        print(self.load_state_dict(state_dict))

    @abstractmethod
    def forward(self,
                raw_audio: BatchEncoding, 
                sentence: BatchEncoding, 
                return_dict: Optional[bool] = True,
                labels: Optional[Union[np.ndarray, List[bool], torch.Tensor]]= None
                ) -> AudioTextOutput | Tuple:
        raise NotImplementedError()

    @abstractmethod
    def query_audio(self, text_query: str, raw_audios: np.ndarray | List[float] | List[np.ndarray] | torch.Tensor) -> torch.Tensor:
        """
        Returns a vector of size N, representing the logits for each of the N audio files when matched with the text query.
        """
        raise NotImplementedError()


class AudioTextRetrieverSelfAtt(AudioTextRetriever):
    def __init__(self, 
                 loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                 text_enc: Optional[TextEncoder] = None, 
                 audio_enc: Optional[AudioEncoder] = None,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__(text_enc, audio_enc, num_heads, dropout)
        self.loss_fn = loss_fn

    def encode_audio(self, raw_audio: BatchEncoding) -> torch.Tensor:
        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)
        audio_embed = self.AudioEncoder(raw_audio)
        if self.num_heads > 0:
            audio_embed = layer_norm(self.AudioAttention(audio_embed, audio_embed, audio_embed)[0] + audio_embed)
            audio_embed = layer_norm(self.AudioAttentionFF(audio_embed) + audio_embed)
        return audio_embed

    def encode_text(self, sentence: BatchEncoding) -> torch.Tensor:
        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)
        text_embed = self.TextEncoder(sentence)
        if self.num_heads > 0:
            text_embed = layer_norm(self.TextAttention(text_embed, text_embed, text_embed)[0] + text_embed)
            text_embed = layer_norm(self.TextAttentionFF(text_embed) + text_embed)
        return text_embed

    # Returns a tensor of size (B, 2, embedding_dim).
    # Tensor at [:, 0, :] represents audio embedding.
    # Tensor at [:, 1, :] represents text embedding.
    def forward(self,
                raw_audio: BatchEncoding, 
                sentence: BatchEncoding, 
                return_dict: Optional[bool] = True,
                labels: Optional[Union[np.ndarray, List[bool], torch.Tensor]]= None
                ) -> AudioTextOutput | Tuple:
        audio_embed = self.encode_audio(raw_audio)
        text_embed = self.encode_text(sentence)

        assert audio_embed.size(0) == text_embed.size(0)
        embeddings = torch.stack((audio_embed, text_embed), dim=-2)

        if labels is not None: # labels present, calculate loss
            tensor_labels = torch.Tensor(labels).to(device)
            loss = self.loss_fn(audio_embed, text_embed, tensor_labels)
        else: # labels absent, no loss calculated
            loss = None

        return (AudioTextOutput(embeddings=embeddings, loss=loss) if return_dict 
                else (loss, embeddings)) # Trainer specifies to return loss as first element if a tuple is returned
    
    def query_audio(self, text_query: str, raw_audios: Optional[np.ndarray | List[float] | List[np.ndarray] | torch.Tensor] = None,  
                    processed_audios: Optional[BatchEncoding] = None, batch_size: int = 128, sampling_rate: int = 16000) -> torch.Tensor:
        if raw_audios is None and processed_audios is None:
            raise Exception("Either one of raw_audios or processed_audios must be passed in!")
        text_embed = self.TextEncoder.preprocess([text_query], device=device)
        text_embed = self.encode_text(text_embed).cpu()

        if processed_audios is None:
            assert raw_audios is not None
            audio_embeds = []
            for start_index in range(0, len(raw_audios), batch_size):
                batch = raw_audios[start_index : start_index + batch_size]
                audio_inputs = self.AudioEncoder.preprocess(batch, sampling_rate=16_000, device=device)
                audio_embeds.append(self.encode_audio(audio_inputs).cpu())
            audio_embed = torch.cat(audio_embeds, dim=0).cpu()
        else:
            audio_embed = processed_audios
            

class AudioTextRetrieverWithMLP(AudioTextRetrieverSelfAtt):
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
        audio_embed = self.encode_audio(raw_audio)
        audio_embed = self.AudioMLP(audio_embed)

        text_embed = self.encode_text(sentence)
        text_embed = self.TextMLP(text_embed)

        embeddings = torch.stack((audio_embed, text_embed), dim=-2)

        if labels is not None: # labels present, calculate loss
            tensor_labels = torch.Tensor(labels).to(device)
            loss = self.loss_fn(audio_embed, text_embed, tensor_labels)
        else: # labels absent, no loss calculated
            loss = None

        return (AudioTextOutput(embeddings=embeddings, loss=loss) if return_dict 
                else (loss, embeddings)) # Trainer specifies to return loss as first element if a tuple is returned


class AudioTextRetrieverCrossAtt(AudioTextRetriever):
    def __init__(self, 
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 text_enc: Optional[TextEncoder] = None, 
                 audio_enc: Optional[AudioEncoder] = None,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__(text_enc, audio_enc, num_heads, dropout)
        self.loss_fn = loss_fn

    def do_cross_attention(self,
                           audio_embeds: torch.Tensor,
                           text_embeds: torch.Tensor,
                           output_embeddings: bool = False,
                           device: str = "cuda"
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        assert audio_embeds.shape == text_embeds.shape
        batch_size, embed_size = audio_embeds.shape

        embeddings = None if not output_embeddings else torch.empty(
            (batch_size, batch_size, 2, embed_size), 
            device=audio_embeds.device
        )
        logits = torch.empty((batch_size, batch_size), device=device)
        audio_embeds = audio_embeds.to(device)
        text_embeds = text_embeds.to(device)

        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)
        for i in range(batch_size):
            current_audio = audio_embeds[i]
            audio_embed = current_audio.repeat(batch_size, 1)

            tmp_audio_embed = layer_norm(self.AudioAttention.to(device)(text_embeds, audio_embed, audio_embed)[0] + text_embeds)
            tmp_text_embed = layer_norm(self.TextAttention.to(device)(audio_embed, text_embeds, text_embeds)[0] + audio_embed)

            audio_embed = layer_norm(self.AudioAttentionFF.to(device)(tmp_audio_embed) + tmp_audio_embed)
            text_embed = layer_norm(self.TextAttentionFF.to(device)(tmp_text_embed) + tmp_text_embed)

            if output_embeddings:
                assert embeddings is not None
                embeddings[i] = torch.stack((audio_embed, text_embed), dim=-2)
            logits[i] = (audio_embed * text_embed).sum(dim=1)

        return embeddings, logits
        

    # Returns a tensor of size (B, B, 2, embedding_dim) if output_embeddings is True 
    #     and skip_cross_attention is False.
    # Tensor at [:, :, 0, :] represents audio embedding.
    # Tensor at [:, :, 1, :] represents text embedding.
    # Note that this usually takes an ungodly amount of space. (O(N^2*D))
    #
    # Instead, consider setting skip_cross_attention to True.
    # In that case, the value of output_embeddings is ignored and only a
    #     tensor of size (B, 2, embedding_dim) is returned.
    # Tensor at [:, 0, :] represents audio embedding, without cross attention.
    # Tensor at [:, 1, :] represents text embedding, without cross attention.
    # The cross attention functionality will be encapsulated in the returned 
    #     'cross_attention' property of the AudioTextOutput.
    def forward(self,
                raw_audio: BatchEncoding, 
                sentence: BatchEncoding, 
                return_dict: Optional[bool] = True,
                labels: Optional[Union[np.ndarray, List[bool], torch.Tensor]]= None,
                output_embeddings: bool = False,
                skip_cross_attention: bool = True
                ) -> AudioTextOutput | Tuple:
        audio_embeds = self.AudioEncoder(raw_audio)
        text_embeds = self.TextEncoder(sentence)

        embeddings, logits = self.do_cross_attention(audio_embeds, text_embeds, 
                                                     not skip_cross_attention and output_embeddings)


        if labels is not None: # labels present, calculate loss
            tensor_labels = torch.Tensor(labels).to(device)
            loss = self.loss_fn(logits, tensor_labels)
        else: # labels absent, no loss calculated
            loss = None

        if skip_cross_attention:
            output = torch.stack((audio_embeds, text_embeds), dim=-2)
            return (AudioTextOutput(embeddings=output, loss=loss) if return_dict 
                    else (loss, output)) # Trainer specifies to return loss as first element if a tuple is returned
        else:
            output: torch.Tensor = embeddings if output_embeddings else logits #type:ignore
            return (AudioTextOutput(embeddings=output, loss=loss) if return_dict 
                    else (loss, output)) # Trainer specifies to return loss as first element if a tuple is returned

    def query_audio(self, text_query: str, raw_audios: np.ndarray | List[float] | List[np.ndarray] | torch.Tensor) -> torch.Tensor:
        pass