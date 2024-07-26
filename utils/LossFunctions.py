import torch
import torch.nn.functional as F

from typing import Optional


def contrastiveCE(
        audio_embed: torch.Tensor, 
        text_embed: torch.Tensor, 
        labels: torch.Tensor, 
        temperature: Optional[float] = 0.2
    ):
    logits_audio2text = audio_embed @ text_embed.T / temperature
    probs_audio2text = F.softmax(logits_audio2text, dim=1)

    logits_text2audio = logits_audio2text.T
    probs_text2audio = F.softmax(logits_text2audio, dim=1)

    return 1/2 * (
        F.binary_cross_entropy(probs_audio2text.diagonal(), labels)
        + F.binary_cross_entropy(probs_text2audio.diagonal(), labels)
    )

def contrastiveCrossAtt(
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        temperature: Optional[float] = 0.2
    ):
    logits /= temperature
    probs_audio2text = F.log_softmax(logits, dim=1)
    probs_text2audio = F.log_softmax(logits.T, dim=1)

    return 1/2 * (
        F.binary_cross_entropy(torch.exp(probs_audio2text.diagonal()), labels)
        + F.binary_cross_entropy(torch.exp(probs_text2audio.diagonal()), labels)
    )
    """
    return F.binary_cross_entropy_with_logits(logits.diagonal(), labels)
    """

def create_contrastive_loss(temperature: float = 0.2, cross_attn: bool = False):
    print(f"Created contrastive loss function for {'cross-' if cross_attn else 'self-'}attention model with temperature {temperature}.")
    if cross_attn:
        return lambda logits, labels: contrastiveCrossAtt(logits, labels, temperature)
    else:
        return lambda audio, text, labels: contrastiveCE(audio, text, labels, temperature)
