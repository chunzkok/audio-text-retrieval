import torch

from torch.nn import functional
from typing import Optional


def contrastiveCE(
        audio_embed: torch.Tensor, 
        text_embed: torch.Tensor, 
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

def create_contrastive_loss(temperature: float = 0.2):
    print(f"Created contrastive loss function with temperature {temperature}")
    return lambda audio, text, labels: contrastiveCE(audio, text, labels, temperature)
