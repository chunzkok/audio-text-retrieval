import numpy as np
import torch

from torch.nn import functional

CONTRASTIVE_TEMP = 0.2 # from the pdf

def contrastiveCE(audio_embed: torch.FloatTensor, text_embed: torch.FloatTensor, labels: torch.Tensor):
    num = (audio_embed @ text_embed.T) / CONTRASTIVE_TEMP

    denom_audio = (audio_embed @ text_embed.T) / CONTRASTIVE_TEMP
    denom_audio = np.exp(denom_audio)
    denom_audio = np.sum(denom_audio, axis=1)
    denom_audio = np.log(denom_audio)

    denom_text = (text_embed @ audio_embed.T) / CONTRASTIVE_TEMP
    denom_text = np.exp(denom_text)
    denom_text = np.sum(denom_text, axis=1)
    denom_text = np.log(denom_text)

    loss = (functional.cross_entropy(num - denom_audio, labels)
        + functional.cross_entropy(num - denom_text, labels)) / 2

    return loss


