import numpy as np
import torch
import torchaudio
from typing import Callable, List, Optional, Union

def path_to_audio(path: str | List[str], sampling_rate: int = 16_000) -> np.ndarray | List[np.ndarray]:
    if type(path) == list:
        return [_path_to_audio_single(p, sampling_rate) for p in path]
    else:
        assert type(path) == str
        return _path_to_audio_single(path, sampling_rate)

def _path_to_audio_single(path: str, target_rate: int) -> np.ndarray:
    raw_audio, sampling_rate = torchaudio.load(path)
    if (sampling_rate != target_rate):
        raw_audio = torchaudio.transforms.Resample(sampling_rate, target_rate)(raw_audio)
    return raw_audio.squeeze().numpy()

class AudioTextDataCollator:
    def __init__(
            self, 
            k: int,
            path_to_audio_converter: Callable[[Union[str, List[str]]], Union[np.ndarray, List[np.ndarray], torch.Tensor]] 
                = path_to_audio
        ):
        self.path_to_audio_converter = path_to_audio_converter # should output at the correct sampling rate too!
        self.k = k # Half of the number of negative samples per audio file (i.e. # neg samples = 2k)

    def __call__(self, inputs):
        raw_audio = self.path_to_audio_converter([input["path"] for input in inputs])
        caption = [input["caption"] for input in inputs]
        batch = {
            "raw_audio": [],
            "sentence": [],
            "labels": []
        }
        N = len(inputs)

        for i in range(N):
            audio = raw_audio[i]

            if type(caption[i] == list):
                sentence = np.random.choice(caption[i])
            else:
                sentence = caption[i]

            batch["raw_audio"].append(audio)
            batch["sentence"].append(sentence)
            batch["labels"].append(1)

            # generate negative samples for the current audio
            neg_indices = self._random_index_excluding(N, i, self.k)
            for j in neg_indices:
                negative_caption = np.random.choice(caption[j]) if type(caption[j]) == list else caption[j]
                batch["raw_audio"].append(audio)
                batch["sentence"].append(negative_caption)
                batch["labels"].append(0)

            # generate negative samples for the current caption (or one of the current captions, if there are multiple)
            neg_indices = self._random_index_excluding(N, i, self.k)
            for j in neg_indices:
                negative_audio = raw_audio[j]
                current_caption = np.random.choice(caption[i]) if type(caption[i]) == list else caption[i]
                batch["raw_audio"].append(negative_audio)
                batch["sentence"].append(current_caption)
                batch["labels"].append(0)

        return batch

    def _random_index_excluding(
            self, 
            upper_bound: int, 
            exclude: int, 
            size: int = 1
        ) -> np.ndarray:
        indices = np.random.randint(upper_bound - 1, size=size)
        return indices + (indices >= exclude)

