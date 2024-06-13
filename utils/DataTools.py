import numpy as np
import torch
from typing import Callable, List, Optional, Union

class AudioTextDataCollator:
    def __init__(
            self, 
            path_to_audio_converter: Callable[[Union[str, List[str]]], Union[np.ndarray, List[np.ndarray], torch.Tensor]], 
            k: int):
        self.path_to_audio_converter = path_to_audio_converter # should output at the correct sampling rate too!
        self.k = k # Number of negative samples per audio file

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

            if type(caption[i]) == list:
                for sentence in caption[i]:
                    batch["raw_audio"].append(audio)
                    batch["sentence"].append(sentence)
                    batch["labels"].append(1)
            else:
                batch["raw_audio"].append(audio)
                batch["sentence"].append(caption[i])
                batch["labels"].append(1)

            neg_indices = self._random_index_excluding(N, i, self.k)
            assert type(neg_indices) == np.ndarray # hacky solution, we are only using clotho which should have 5 captions
            for j in neg_indices:
                batch["raw_audio"].append(audio)
                if type(caption[j] == list):
                    batch["sentence"].append(np.random.choice(caption[j]))
                else:
                    batch["sentence"].append(caption[j])
                batch["labels"].append(0)

        return batch

    def _random_index_excluding(
            self, 
            upper_bound: int, 
            exclude: int, 
            size: Optional[int] = None
        ) -> Union[int, np.ndarray]:
        if size is not None:
            indices = np.random.randint(upper_bound - 1, size=size)
            return indices + (indices >= exclude)
        else:
            i = np.random.randint(upper_bound - 1)
            return i if i < exclude else i+1

