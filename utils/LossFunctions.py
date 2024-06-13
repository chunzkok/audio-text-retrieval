import torch

from torch.nn import functional
from typing import Optional


def contrastiveCE(
        audio_embed: torch.FloatTensor, 
        text_embed: torch.FloatTensor, 
        labels: torch.Tensor, 
        temperature: Optional[float] = 0.2
    ):

    # wait... aren't they just transposes of each other
    logits_audio2text = audio_embed @ text_embed.T / temperature
    logits_text2audio = text_embed @ audio_embed.T / temperature

    return 1/2 * (
        functional.cross_entropy(logits_audio2text, labels)
        + functional.cross_entropy(logits_text2audio, labels)
    )


