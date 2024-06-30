import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from functools import cache
from transformers import BatchEncoding
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

def path_to_audio(path: str | List[str], sampling_rate: int = 16_000) -> np.ndarray | List[np.ndarray]:
    if type(path) == list:
        return [_path_to_audio_single(p, sampling_rate) for p in path]
    else:
        assert type(path) == str
        return _path_to_audio_single(path, sampling_rate)

def _path_to_audio_single(path: str, target_rate: int, min_samples: int = 400) -> np.ndarray:
    raw_audio, sampling_rate = torchaudio.load(path)
    if (sampling_rate != target_rate):
        raw_audio = torchaudio.transforms.Resample(sampling_rate, target_rate)(raw_audio)
    return raw_audio.mean(dim=0).numpy() 

def concat_tensors_with_padding(tensors: Iterable[torch.Tensor], dim: int = 1) -> torch.Tensor:
    needs_padding = False
    max_len = next(iter(tensors)).shape[dim]

    for tensor in tensors:
        if tensor.shape[dim] != max_len:
            needs_padding = True
        max_len = max(tensor.shape[dim], max_len)

    if needs_padding:
        padded_tensors = []
        for tensor in tensors:
            pad_size = max_len - tensor.shape[dim]
            padded_tensors.append(F.pad(tensor, (0, pad_size) + (0, 0) * (tensor.dim() - dim - 1)))
        return torch.concat(padded_tensors)
    else:
        return torch.concat(list(tensors))
    

def concat_encodings(encodings: Iterable[BatchEncoding]) -> BatchEncoding:
    data = {}
    for key in next(iter(encodings)).keys():
        vals = [encoding[key] for encoding in encodings]
        if len(vals) == 0:
            raise Exception(f"utils.DataTools.concat_encodings found an empty BatchEncoding key {key}! "
                            + "This could be because the values are not of type torch.Tensor.")
        data[key] = concat_tensors_with_padding(vals)
    return BatchEncoding(data)

def random_index_excluding(
        upper_bound: int, 
        exclude: int, 
        size: int = 1
    ) -> np.ndarray:
    indices = np.random.randint(upper_bound - 1, size=size)
    return indices + (indices >= exclude)

def generate_samples(pos_samples: Dict[str, Any], num_pos: int = 1, num_neg = 0, read_audios: bool = False) -> Dict[str, Any]:
    N = len(pos_samples["path"])
    samples = {
        "path": [],
        "audio": [],
        "caption": [],
        "label": []
    }
    for i, (path, caption) in enumerate(zip(pos_samples["path"], pos_samples["caption"])):
        audio = path_to_audio(path) if read_audios else None

        # positive samples
        if read_audios:
            samples["audio"].extend([audio] * num_pos)
        else:
            samples["path"].extend([path] * num_pos)
        if type(caption) == list:
            samples["caption"].extend(np.random.choice(
                caption, 
                size=num_pos, 
                replace=(num_pos > len(caption))
            ))
        else:
            samples["caption"].extend([caption] * num_pos)
        samples["label"].extend([1] * num_pos)

        # negative samples
        if num_neg == 0: 
            continue
        neg_indices = random_index_excluding(upper_bound=N, exclude=i, size=num_neg)
        for j in neg_indices:
            # choose between fixing audio/caption for the negative sample
            coin_flip = bool(np.random.randint(2))
            if coin_flip:
                # fix audio
                if read_audios:
                    samples["audio"].append(audio)
                else:
                    samples["path"].append(path)

                # find negative caption
                if type(caption) == list:
                    samples["caption"].append(np.random.choice(pos_samples["caption"][j]))
                else:
                    samples["caption"].append(pos_samples["caption"][j])
                samples["label"].append(0)
            else:
                # find negative audio
                if read_audios:
                    samples["audio"].append(path_to_audio(pos_samples["path"][j]))
                else:
                    samples["path"].append(pos_samples["path"][j])

                # fix caption
                if type(caption) == list:
                    samples["caption"].append(np.random.choice(caption))
                else:
                    samples["caption"].append(caption)
                samples["label"].append(0)
    if read_audios:
        del samples["path"]
    else:
        del samples["audio"]
    
    return samples

def create_sample_generator(num_pos=1, num_neg=0, read_audios=False) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def sample_generator(batch: Dict[str, Any]) -> Dict[str, Any]:
        return generate_samples(batch, num_pos, num_neg, read_audios)
    return sample_generator

def is_valid_audio(path: str, min_len: Optional[float] = None) -> bool:
    try:
        audio, sampling_rate = torchaudio.load(path)
    except:
        return False

    if min_len:
        return audio.size(1) / sampling_rate >= min_len

    return True


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
        """
        self.extracted_audios: Dict[str, BatchEncoding] = {}
        self.tokenized_texts : Dict[str, BatchEncoding] = {}
        """

    def __call__(self, inputs, sampling_rate: int = 16_000):
        audio_paths = [input["path"] for input in inputs]
        caption = [input["caption"] for input in inputs]
        batch: Dict[str, Any] = {
            "raw_audio": [],
            "sentence": [],
            "labels": torch.tensor([input["label"] for input in inputs]).type("torch.FloatTensor")
        }

        batch["raw_audio"] = self.audio_processor(path_to_audio(audio_paths), sampling_rate)
        batch["sentence"] = self.text_tokenizer(caption)

        """
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
        """
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
        #N = len(list(samples))
        #if len(unknown_samples) < N:
        #    print(f"Saved {N - len(unknown_samples)} calculations!")
        #####################################
        return concat_encodings([memo[name] for name in samples])

class ProcessedAudioTextDataCollator:
    def __init__(
            self, 
            audio_processor: Callable[[Union[np.ndarray, List[np.ndarray], torch.Tensor], int], BatchEncoding],
            text_tokenizer: Callable[[Union[str, List[str]]], BatchEncoding],
        ):
        self.audio_processor = audio_processor
        self.text_tokenizer = text_tokenizer

    def __call__(self, inputs, sampling_rate: int = 16_000):
        audios = [input["audio"] for input in inputs]
        caption = [input["caption"] for input in inputs]
        batch: Dict[str, Any] = {
            "raw_audio": [],
            "sentence": [],
            "labels": torch.tensor([input["label"] for input in inputs])
        }

        batch["raw_audio"] = self.audio_processor(audios, sampling_rate)
        batch["sentence"] = self.text_tokenizer(caption)

        return batch