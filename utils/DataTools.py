import numpy as np
import torch
import torchaudio
from typing import Any, Callable, Dict, Iterable, List, Union
from transformers import BatchEncoding

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

def concat_encodings(encodings: Iterable[BatchEncoding]) -> BatchEncoding:
    data = {}
    for key in next(iter(encodings)).keys():
        data[key] = torch.concat([encoding[key] for encoding in encodings if type(encoding) == torch.Tensor], dim=0)
        if len(data[key]) == 0:
            raise Exception(f"utils.DataTools.concat_encodings found an empty BatchEncoding key {key}! "
                            + "This could be because the values are not of type torch.Tensor.")
    return BatchEncoding(data)

def random_index_excluding(
        upper_bound: int, 
        exclude: int, 
        size: int = 1
    ) -> np.ndarray:
    indices = np.random.randint(upper_bound - 1, size=size)
    return indices + (indices >= exclude)

def generate_samples(pos_samples: Dict[str, Any], num_pos: int = 1, num_neg = 0) -> Dict[str, Any]:
    N = len(pos_samples["path"])
    samples = {
        "path": [],
        "caption": [],
        "labels": []
    }
    for i, (path, caption) in enumerate(zip(pos_samples["path"], pos_samples["caption"])):
        # positive samples
        samples["path"].extend([path] * num_pos)
        if type(caption) == list:
            samples["caption"].extend(np.random.choice(
                caption, 
                size=num_pos, 
                replace=(num_pos > len(caption))
            ))
        else:
            samples["caption"].extend([caption] * num_pos)
        samples["labels"].extend([1] * num_pos)

        # negative samples
        if num_neg == 0: 
            continue
        neg_indices = random_index_excluding(upper_bound=N, exclude=i, size=num_neg)
        for j in neg_indices:
            # choose between fixing audio/caption for the negative sample
            coin_flip = bool(np.random.randint(2))
            if coin_flip:
                # fix audio
                samples["path"].append(path)
                if type(caption) == list:
                    samples["caption"].append(np.random.choice(pos_samples["caption"][j]))
                else:
                    samples["caption"].append(pos_samples["caption"][j])
                samples["labels"].append(0)
            else:
                # fix caption
                samples["path"].append(pos_samples["path"][j])
                if type(caption) == list:
                    samples["caption"].append(np.random.choice(caption))
                else:
                    samples["caption"].append(caption)
                samples["labels"].append(0)

    return samples

def create_sample_generator(num_pos, num_neg) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def sample_generator(batch: Dict[str, Any]) -> Dict[str, Any]:
        return generate_samples(batch, num_pos, num_neg)
    return sample_generator


class AudioTextDataCollator:
    def __init__(
            self, 
            audio_processor: Callable[[Union[np.ndarray, List[np.ndarray], torch.Tensor], int], BatchEncoding],
            text_tokenizer: Callable[[Union[str, List[str]]], BatchEncoding],
            path_to_audio_converter: Callable[[Union[str, List[str]], int], Union[np.ndarray, List[np.ndarray], torch.Tensor]] 
                = path_to_audio
        ):
        self.path_to_audio_converter = path_to_audio_converter # should output at the correct sampling rate too!
        self.audio_processor = audio_processor
        self.text_tokenizer = text_tokenizer
        # avoid repeated computation
        self.extracted_audios: Dict[str, BatchEncoding] = {}
        self.tokenized_texts : Dict[str, BatchEncoding] = {}

    def __call__(self, inputs, sampling_rate: int = 16_000):
        audio_paths = [input["path"] for input in inputs]
        caption = [input["caption"] for input in inputs]
        batch: Dict[str, Any] = {
            "raw_audio": [],
            "sentence": [],
            "labels": [input["label"] for input in inputs]
        }

        batch["raw_audio"] = self._process_if_needed(
            audio_paths,
            lambda paths: self.audio_processor(path_to_audio(paths), sampling_rate),
            self.extracted_audios
        )
        batch["sentence"] = self._process_if_needed(
            caption,
            self.text_tokenizer,
            self.tokenized_texts
        )
        return batch


    def _process_if_needed(
        self,
        samples: Iterable[str],
        processor: Callable[[List[str]], BatchEncoding], 
        memo: Dict[str, BatchEncoding]
    ) -> BatchEncoding:
        unknown_samples = {sample for sample in samples if sample not in memo}
        if len(unknown_samples) > 0:
            processed = processor(list(unknown_samples))
            for i, sample in enumerate(unknown_samples):
                memo[sample] = BatchEncoding({
                    key : val[i].unsqueeze(dim=0) 
                    for key, val in processed.items() 
                    if type(val) == torch.Tensor
                })
        # for checking that the memoization works
        N = len(list(samples))
        if len(unknown_samples) < N:
            print(f"Saved {N - len(unknown_samples)} calculations!")
        #####################################
        return concat_encodings([memo[name] for name in samples])