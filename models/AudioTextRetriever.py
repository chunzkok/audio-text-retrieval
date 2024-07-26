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
    logits: torch.Tensor
    audio_embed: Optional[torch.Tensor] = None
    text_embed: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    def __getitem__(self, key: Union[str, int, slice, None]):
        if key == None: 
            return None
        elif type(key) == int:
            if key == 0:
                return self.logits
            elif key == 2:
                return self.audio_embed
            elif key == 3:
                return self.text_embed
            elif key == 4:
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
        result = []
        for item in (self.logits, self.audio_embed, self.text_embed, self.loss):
            if item is not None:
                result.append(item)
        return tuple(result)

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
        self.saved_text_embeds = {}
        self.saved_audio_embeds = {}

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

    def encode_audio(self, raw_audio: np.ndarray | List[np.ndarray], sampling_rate: int = 16_000, 
                     batch_size: Optional[int] = None, save_as: Optional[str] = None, load_if_exists: bool = True) -> torch.Tensor:
        if save_as is None or not load_if_exists or save_as not in self.saved_audio_embeds:
            if batch_size is not None and type(raw_audio) == list:
                audio_embeds = []
                for start_index in range(0, len(raw_audio), batch_size):
                    batch = raw_audio[start_index : start_index + batch_size]
                    audio_inputs = self.AudioEncoder.preprocess(batch, sampling_rate, device=device)
                    audio_embeds.append(self.AudioEncoder(audio_inputs))
                audio_embed = torch.cat(audio_embeds, dim=0)
            else:
                audio_embed = self.AudioEncoder.preprocess(raw_audio, sampling_rate, device=device)
                audio_embed = self.AudioEncoder.forward(audio_embed)

            if save_as is None:
                return audio_embed
            else:
                self.saved_audio_embeds[save_as] = audio_embed
        return self.saved_audio_embeds[save_as]

    def encode_text(self, sentence: str | List[str], save_as: Optional[str] = None, batch_size: Optional[int] = None, 
                    load_if_exists: bool = True) -> torch.Tensor:
        if save_as is None or not load_if_exists or save_as not in self.saved_text_embeds:
            if batch_size is not None and type(sentence) == list:
                text_embeds = []
                for start_index in range(0, len(sentence), batch_size):
                    batch = sentence[start_index : start_index + batch_size]
                    text_inputs = self.TextEncoder.preprocess(batch, device=device)
                    text_embeds.append(self.TextEncoder(text_inputs))
                text_embed = torch.cat(text_embeds, dim=0)
            else:
                text_embed = self.TextEncoder.preprocess(sentence, device=device)
                text_embed = self.TextEncoder.forward(text_embed)

            if save_as is None:
                return text_embed
            else:
                self.saved_text_embeds[save_as] = text_embed
        return self.saved_text_embeds[save_as]

    @abstractmethod
    def forward(self,
                raw_audio: BatchEncoding, 
                sentence: BatchEncoding, 
                return_dict: Optional[bool] = True,
                labels: Optional[Union[np.ndarray, List[bool], torch.Tensor]]= None
                ) -> AudioTextOutput | Tuple:
        raise NotImplementedError()

    @abstractmethod
    def query_audio(self, text_query: str, raw_audios: np.ndarray | List[np.ndarray], sampling_rate: int = 16000,  
                    audio_set_name: Optional[str] = None, batch_size: Optional[int] = None) -> torch.Tensor:
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

    def do_self_attention(self, 
                          audio_embed: torch.Tensor, 
                          text_embed: torch.Tensor,
                          device: str = "cuda"
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)

        audio_embed = layer_norm(self.AudioAttention(audio_embed, audio_embed, audio_embed)[0] + audio_embed)
        audio_embed = layer_norm(self.AudioAttentionFF(audio_embed) + audio_embed)

        text_embed = layer_norm(self.TextAttention(text_embed, text_embed, text_embed)[0] + text_embed)
        text_embed = layer_norm(self.TextAttentionFF(text_embed) + text_embed)

        return audio_embed, text_embed


    def forward(self,
                raw_audio: BatchEncoding, 
                sentence: BatchEncoding, 
                return_dict: Optional[bool] = True,
                labels: Optional[Union[np.ndarray, List[bool], torch.Tensor]]= None
                ) -> AudioTextOutput | Tuple:

        audio_embed = self.AudioEncoder(raw_audio)
        text_embed = self.TextEncoder(sentence)

        if self.num_heads > 0:
            audio_embed, text_embed = self.do_self_attention(audio_embed, text_embed)

        assert audio_embed.size(0) == text_embed.size(0)
        if labels is not None: # labels present, calculate loss
            tensor_labels = torch.Tensor(labels).to(device)
            loss = self.loss_fn(audio_embed, text_embed, tensor_labels)
        else: # labels absent, no loss calculated
            loss = None

        logits = (audio_embed @ text_embed.T).squeeze()
        return (AudioTextOutput(logits, audio_embed, text_embed, loss) if return_dict 
                else (loss, logits, audio_embed, text_embed)) # Trainer specifies to return loss as first element if a tuple is returned
    
    def query_audio(self, text_query: str, raw_audios: np.ndarray | List[np.ndarray], sampling_rate: int = 16000,  
                    audio_set_name: Optional[str] = None, batch_size: Optional[int] = None) -> torch.Tensor:
        text_embed = self.encode_text(text_query)
        audio_embed = self.encode_audio(raw_audios, sampling_rate, batch_size, audio_set_name)

        if self.num_heads > 0:
            audio_embed, text_embed = self.do_self_attention(audio_embed, text_embed)

        return (text_embed @ audio_embed.T).squeeze()

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

    def forward(self,
                raw_audio: BatchEncoding, 
                sentence: BatchEncoding, 
                return_dict: Optional[bool] = True,
                labels: Optional[Union[np.ndarray, List[bool], torch.Tensor]]= None
                ) -> AudioTextOutput | Tuple:
        audio_embed = self.AudioEncoder(raw_audio)
        text_embed = self.TextEncoder(sentence)

        audio_embed, text_embed = self.do_self_attention(audio_embed, text_embed)

        audio_embed = self.AudioMLP(audio_embed)
        text_embed = self.TextMLP(text_embed)

        if labels is not None: # labels present, calculate loss
            tensor_labels = torch.Tensor(labels).to(device)
            loss = self.loss_fn(audio_embed, text_embed, tensor_labels)
        else: # labels absent, no loss calculated
            loss = None

        logits = (audio_embed @ text_embed.T).squeeze()
        return (AudioTextOutput(logits, audio_embed, text_embed, loss) if return_dict 
                else (loss, logits, audio_embed, text_embed)) # Trainer specifies to return loss as first element if a tuple is returned

    def query_audio(self, text_query: str, raw_audios: np.ndarray | List[np.ndarray], sampling_rate: int = 16000,  
                    audio_set_name: Optional[str] = None, batch_size: Optional[int] = None) -> torch.Tensor:
        text_embed = self.encode_text(text_query)
        audio_embed = self.encode_audio(raw_audios, sampling_rate, batch_size, audio_set_name)

        if self.num_heads > 0:
            audio_embed, text_embed = self.do_self_attention(audio_embed, text_embed)

        audio_embed = self.AudioMLP(audio_embed)
        text_embed = self.TextMLP(text_embed)

        return text_embed @ audio_embed.T


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
                           do_transpose: bool = False,
                           device: str = "cuda"
    ) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        num_audio = audio_embeds.size(0)
        num_text = text_embeds.size(0)

        audio_embeds = audio_embeds.to(device)
        text_embeds = text_embeds.to(device)

        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)

        if do_transpose:
            embeddings = None if not output_embeddings else (
                torch.empty((num_text, num_audio, self.embedding_dim), device=device),
                torch.empty((num_text, num_audio, self.embedding_dim), device=device) 
            )
            logits = torch.empty((num_text, num_audio), device=device)

            for i in range(num_text):
                current_text = text_embeds[i]
                text_embed = current_text.repeat(num_audio, 1)

                tmp_audio_embed = layer_norm(self.AudioAttention.to(device)(text_embed, audio_embeds, audio_embeds)[0] + text_embed)
                tmp_text_embed = layer_norm(self.TextAttention.to(device)(audio_embeds, text_embed, text_embed)[0] + audio_embeds)

                audio_embed = layer_norm(self.AudioAttentionFF.to(device)(tmp_audio_embed) + tmp_audio_embed)
                text_embed = layer_norm(self.TextAttentionFF.to(device)(tmp_text_embed) + tmp_text_embed)

                if output_embeddings:
                    assert embeddings is not None
                    embeddings[0][i] = audio_embed
                    embeddings[1][i] = text_embed
                logits[i] = (audio_embed * text_embed).sum(dim=1)
        else:
            embeddings = None if not output_embeddings else (
                torch.empty((num_audio, num_text, self.embedding_dim), device=device),
                torch.empty((num_audio, num_text, self.embedding_dim), device=device) 
            )
            logits = torch.empty((num_audio, num_text), device=device)

            for i in range(num_audio):
                current_audio = audio_embeds[i]
                audio_embed = current_audio.repeat(num_text, 1)

                tmp_audio_embed = layer_norm(self.AudioAttention.to(device)(text_embeds, audio_embed, audio_embed)[0] + text_embeds)
                tmp_text_embed = layer_norm(self.TextAttention.to(device)(audio_embed, text_embeds, text_embeds)[0] + audio_embed)

                audio_embed = layer_norm(self.AudioAttentionFF.to(device)(tmp_audio_embed) + tmp_audio_embed)
                text_embed = layer_norm(self.TextAttentionFF.to(device)(tmp_text_embed) + tmp_text_embed)

                if output_embeddings:
                    assert embeddings is not None
                    embeddings[0][i] = audio_embed
                    embeddings[1][i] = text_embed
                logits[i] = (audio_embed * text_embed).sum(dim=1)

        return embeddings, logits
        

    # Returns a tuple of 2 tensors of size (B, B, embedding_dim) if output_embeddings is True 
    #     and skip_cross_attention is False.
    # Note that this usually takes an ungodly amount of space. (O(N^2*D))
    #
    # Instead, consider setting skip_cross_attention to True.
    # In that case, the value of output_embeddings is ignored and only 2
    #     tensors of size (B, embedding_dim) are returned.
    def forward(self,
                raw_audio: BatchEncoding, 
                sentence: BatchEncoding, 
                return_dict: Optional[bool] = True,
                labels: Optional[Union[np.ndarray, List[bool], torch.Tensor]]= None,
                output_embeddings: bool = False,
                skip_cross_attention: bool = True
                ) -> AudioTextOutput | Tuple:
        audio_embeds = self.AudioEncoder.forward(raw_audio)
        text_embeds = self.TextEncoder.forward(sentence)

        embeddings, logits = self.do_cross_attention(audio_embeds, text_embeds, 
                                                     not skip_cross_attention and output_embeddings)


        if labels is not None: # labels present, calculate loss
            tensor_labels = torch.Tensor(labels).to(device)
            loss = self.loss_fn(logits, tensor_labels)
        else: # labels absent, no loss calculated
            loss = None

        if skip_cross_attention:
            return (AudioTextOutput(logits, audio_embeds, text_embeds, loss) if return_dict 
                    else (loss, logits, audio_embeds, text_embeds)) # Trainer specifies to return loss as first element if a tuple is returned
        else:
            assert embeddings is not None
            return (AudioTextOutput(logits, embeddings[0], embeddings[1], loss) if return_dict 
                    else (loss, logits, embeddings[0], embeddings[1])) # Trainer specifies to return loss as first element if a tuple is returned

    def query_audio(self, text_query: str, raw_audios: np.ndarray | List[np.ndarray], sampling_rate: int = 16000,  
                    audio_set_name: Optional[str] = None, batch_size: Optional[int] = None) -> torch.Tensor:
        text_embed = self.encode_text(text_query)
        audio_embed = self.encode_audio(raw_audios, sampling_rate, batch_size, audio_set_name)
        _, logits = self.do_cross_attention(audio_embed, text_embed, do_transpose=True)
        return logits.squeeze()

class AudioTextRetrieverCrossAtt2(AudioTextRetrieverCrossAtt):
    def __init__(self, 
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 text_enc: Optional[TextEncoder] = None, 
                 audio_enc: Optional[AudioEncoder] = None,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__(loss_fn, text_enc, audio_enc, num_heads, dropout)
        self.AudioAttention = nn.MultiheadAttention(self.embedding_dim, num_heads, dropout, batch_first=True)
        self.TextAttention = nn.MultiheadAttention(self.embedding_dim, num_heads, dropout, batch_first=True)

    def do_cross_attention(self,
                           audio_embeds: torch.Tensor,
                           text_embeds: torch.Tensor,
                           output_embeddings: bool = False,
                           do_transpose: bool = False,
                           device: str = "cuda"
    ) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        num_audio = audio_embeds.size(0)
        num_text = text_embeds.size(0)

        audio_embeds = audio_embeds.to(device)
        text_embeds = text_embeds.to(device)

        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)

        if do_transpose:
            embeddings = None if not output_embeddings else (
                torch.empty((num_text, num_audio, self.embedding_dim), device=device),
                torch.empty((num_text, num_audio, self.embedding_dim), device=device) 
            )
            logits = torch.empty((num_text, num_audio), device=device)

            for i in range(num_text):
                current_text = text_embeds[i]
                text_embed = current_text.repeat(num_audio, 1).unsqueeze(dim=1)
                audio_embed = audio_embeds.unsqueeze(dim=1)

                tmp_audio_embed = layer_norm(
                    self.AudioAttention.to(device)(text_embed, audio_embed, audio_embed)[0].squeeze()
                    + text_embed.squeeze())
                tmp_text_embed = layer_norm(
                    self.TextAttention.to(device)(audio_embed, text_embed, text_embed)[0].squeeze() 
                    + audio_embed.squeeze())

                audio_embed = layer_norm(self.AudioAttentionFF.to(device)(tmp_audio_embed) + tmp_audio_embed)
                text_embed = layer_norm(self.TextAttentionFF.to(device)(tmp_text_embed) + tmp_text_embed)

                if output_embeddings:
                    assert embeddings is not None
                    embeddings[0][i] = audio_embed
                    embeddings[1][i] = text_embed
                logits[i] = (audio_embed * text_embed).sum(dim=1)
        else:
            embeddings = None if not output_embeddings else (
                torch.empty((num_audio, num_text, self.embedding_dim), device=device),
                torch.empty((num_audio, num_text, self.embedding_dim), device=device) 
            )
            logits = torch.empty((num_audio, num_text), device=device)

            for i in range(num_audio):
                current_audio = audio_embeds[i]
                audio_embed = current_audio.repeat(num_text, 1).unsqueeze(dim=1)
                text_embed = text_embeds.unsqueeze(dim=1)

                tmp_audio_embed = layer_norm(
                    self.AudioAttention.to(device)(text_embed, audio_embed, audio_embed)[0].squeeze() 
                    + text_embed.squeeze())
                tmp_text_embed = layer_norm(
                    self.TextAttention.to(device)(audio_embed, text_embed, text_embed)[0].squeeze() 
                    + audio_embed.squeeze())

                audio_embed = layer_norm(self.AudioAttentionFF.to(device)(tmp_audio_embed) + tmp_audio_embed)
                text_embed = layer_norm(self.TextAttentionFF.to(device)(tmp_text_embed) + tmp_text_embed)

                if output_embeddings:
                    assert embeddings is not None
                    embeddings[0][i] = audio_embed
                    embeddings[1][i] = text_embed
                logits[i] = (audio_embed * text_embed).sum(dim=1)

        return embeddings, logits

class AudioTextRetrieverCrossAtt3(AudioTextRetrieverCrossAtt):
    def __init__(self, 
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 text_enc: Optional[TextEncoder] = None, 
                 audio_enc: Optional[AudioEncoder] = None,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__(loss_fn, text_enc, audio_enc, num_heads, dropout)
        self.AudioAttention = nn.MultiheadAttention(1, 1, dropout, batch_first=True)
        self.TextAttention = nn.MultiheadAttention(1, 1, dropout, batch_first=True)

    def do_cross_attention(self,
                           audio_embeds: torch.Tensor,
                           text_embeds: torch.Tensor,
                           output_embeddings: bool = False,
                           do_transpose: bool = False,
                           device: str = "cuda"
    ) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        num_audio = audio_embeds.size(0)
        num_text = text_embeds.size(0)

        audio_embeds = audio_embeds.to(device)
        text_embeds = text_embeds.to(device)

        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)

        if do_transpose:
            embeddings = None if not output_embeddings else (
                torch.empty((num_text, num_audio, self.embedding_dim), device=device),
                torch.empty((num_text, num_audio, self.embedding_dim), device=device) 
            )
            logits = torch.empty((num_text, num_audio), device=device)

            for i in range(num_text):
                current_text = text_embeds[i]
                text_embed = current_text.repeat(num_audio, 1).unsqueeze(dim=2)
                audio_embed = audio_embeds.unsqueeze(dim=2)

                tmp_audio_embed = layer_norm(
                    self.AudioAttention.to(device).forward(
                        text_embed, audio_embed, audio_embed, need_weights=False)[0].squeeze()
                    + text_embed.squeeze())
                tmp_text_embed = layer_norm(
                    self.TextAttention.to(device).forward(
                        audio_embed, text_embed, text_embed, need_weights=False)[0].squeeze() 
                    + audio_embed.squeeze())

                audio_embed = layer_norm(self.AudioAttentionFF.to(device)(tmp_audio_embed) + tmp_audio_embed)
                text_embed = layer_norm(self.TextAttentionFF.to(device)(tmp_text_embed) + tmp_text_embed)

                if output_embeddings:
                    assert embeddings is not None
                    embeddings[0][i] = audio_embed
                    embeddings[1][i] = text_embed
                logits[i] = (audio_embed * text_embed).sum(dim=1)
        else:
            embeddings = None if not output_embeddings else (
                torch.empty((num_audio, num_text, self.embedding_dim), device=device),
                torch.empty((num_audio, num_text, self.embedding_dim), device=device) 
            )
            logits = torch.empty((num_audio, num_text), device=device)

            for i in range(num_audio):
                current_audio = audio_embeds[i]
                audio_embed = current_audio.repeat(num_text, 1).unsqueeze(dim=2)
                text_embed = text_embeds.unsqueeze(dim=2)

                tmp_audio_embed = layer_norm(
                    self.AudioAttention.to(device).forward(
                        text_embed, audio_embed, audio_embed, need_weights=False)[0].squeeze() 
                    + text_embed.squeeze())
                tmp_text_embed = layer_norm(
                    self.TextAttention.to(device).forward(
                        audio_embed, text_embed, text_embed, need_weights=False)[0].squeeze() 
                    + audio_embed.squeeze())

                audio_embed = layer_norm(self.AudioAttentionFF.to(device)(tmp_audio_embed) + tmp_audio_embed)
                text_embed = layer_norm(self.TextAttentionFF.to(device)(tmp_text_embed) + tmp_text_embed)

                if output_embeddings:
                    assert embeddings is not None
                    embeddings[0][i] = audio_embed
                    embeddings[1][i] = text_embed
                logits[i] = (audio_embed * text_embed).sum(dim=1)

        return embeddings, logits

class TemporalAudioTextRetrieverCrossAtt(AudioTextRetrieverCrossAtt):
    def __init__(self, 
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 text_enc: Optional[TextEncoder] = None, 
                 audio_enc: Optional[AudioEncoder] = None,
                 num_heads: int = 8,
                 dropout: float = 0.2,
                 att_aggregation: str = "collapse"):
        super().__init__(loss_fn, text_enc, audio_enc, num_heads, dropout)
        self.AudioAttention = nn.MultiheadAttention(self.embedding_dim, num_heads, dropout, batch_first=True)
        self.TextAttention = nn.MultiheadAttention(self.embedding_dim, num_heads, dropout, batch_first=True)
        self.att_aggregation = att_aggregation

    def get_logits(self,
                   embed_1: torch.Tensor,
                   embed_2: torch.Tensor,
                   diag_window_size: Optional[int] = None # defaults to approx. 5% width on each side
    ) -> torch.Tensor:
        if self.att_aggregation == "sumdiag":
            logit_matrices = embed_1 @ embed_2.transpose(-1, -2)
            global_sum = logit_matrices.sum(dim=(-1, -2))
            diag_sum = logit_matrices.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)

            window = min(embed_1.size(1), embed_2.size(1)) // 20 if diag_window_size is None else diag_window_size
            for offset in range(window):
                diag_sum += logit_matrices.diagonal(offset=offset+1, dim1=-1, dim2=-2).sum(dim=-1)
                diag_sum += logit_matrices.diagonal(offset=-(offset+1), dim1=-1, dim2=-2).sum(dim=-1)

            logits = global_sum + diag_sum
            return logits / (logit_matrices.size(-1) * logit_matrices.size(-2)) # scale by total number of items
        elif self.att_aggregation == "maxpoolmean":
            logit_matrices = embed_1 @ embed_2.transpose(-1, -2)
            max_patches = nn.functional.max_pool2d(logit_matrices, 16, 10)
            return max_patches.mean((-1, -2))
        elif self.att_aggregation == "mean":
            logit_matrices = embed_1 @ embed_2.transpose(-1, -2)
            mean_patches = nn.functional.avg_pool2d(logit_matrices, 16, 10)
            return mean_patches.mean((-1, -2))
        else:
            collapsed_embed1 = embed_1.mean(dim=-2)
            collapsed_embed2 = embed_2.mean(dim=-2)
            return (collapsed_embed1 * collapsed_embed2).sum(dim=-1)

    def do_cross_attention(self,
                           audio_embeds: torch.Tensor,
                           text_embeds: torch.Tensor,
                           output_embeddings: bool = False,
                           do_transpose: bool = False,
                           device: str = "cuda"
    ) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        num_audio = audio_embeds.size(0)
        num_text = text_embeds.size(0)

        audio_embeds = audio_embeds.to(device)
        text_embeds = text_embeds.to(device)

        layer_norm = nn.LayerNorm(self.embedding_dim).to(device)

        if do_transpose:
            embeddings = None if not output_embeddings else (
                torch.empty((num_text, num_audio, self.embedding_dim), device=device),
                torch.empty((num_text, num_audio, self.embedding_dim), device=device) 
            )
            logits = torch.empty((num_text, num_audio), device=device)

            for i in range(num_text):
                current_text = text_embeds[i]
                text_embed = current_text.repeat(num_audio, 1, 1)
                audio_embed = audio_embeds

                tmp_audio_embed = layer_norm(
                    self.AudioAttention.to(device).forward(
                        audio_embed, text_embed, text_embed, need_weights=False)[0]
                    + audio_embed)
                tmp_text_embed = layer_norm(
                    self.TextAttention.to(device).forward(
                        text_embed, audio_embed, audio_embed, need_weights=False)[0]
                    + text_embed)

                audio_embed = layer_norm(self.AudioAttentionFF.to(device)(tmp_audio_embed) + tmp_audio_embed)
                text_embed = layer_norm(self.TextAttentionFF.to(device)(tmp_text_embed) + tmp_text_embed)

                if output_embeddings:
                    assert embeddings is not None
                    embeddings[0][i] = audio_embed
                    embeddings[1][i] = text_embed
                logits[i] = self.get_logits(text_embed, audio_embed)
        else:
            embeddings = None if not output_embeddings else (
                torch.empty((num_audio, num_text, self.embedding_dim), device=device),
                torch.empty((num_audio, num_text, self.embedding_dim), device=device) 
            )
            logits = torch.empty((num_audio, num_text), device=device)

            for i in range(num_audio):
                current_audio = audio_embeds[i]
                audio_embed = current_audio.repeat(num_text, 1, 1)
                text_embed = text_embeds

                tmp_audio_embed = layer_norm(
                    self.AudioAttention.to(device).forward(
                        audio_embed, text_embed, text_embed, need_weights=False)[0]
                    + audio_embed)
                tmp_text_embed = layer_norm(
                    self.TextAttention.to(device).forward(
                        text_embed, audio_embed, audio_embed, need_weights=False)[0]
                    + text_embed)

                audio_embed = layer_norm(self.AudioAttentionFF.to(device)(tmp_audio_embed) + tmp_audio_embed)
                text_embed = layer_norm(self.TextAttentionFF.to(device)(tmp_text_embed) + tmp_text_embed)

                if output_embeddings:
                    assert embeddings is not None
                    embeddings[0][i] = audio_embed
                    embeddings[1][i] = text_embed
                logits[i] = self.get_logits(audio_embed, text_embed)

        return embeddings, logits