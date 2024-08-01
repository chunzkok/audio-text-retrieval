import datasets
import gc
import numpy as np
import torch

from argparse import ArgumentParser, ArgumentTypeError
from functools import cache
from models.AudioTextRetriever import AudioTextRetriever
from utils.DataTools import path_to_audio
from utils.LossFunctions import contrastiveCE
from pathlib import Path
from typing import List, Tuple, Optional

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextAudioRetrieval:
    def __init__(self, retriever: AudioTextRetriever, weights_dir: Path, data_dir: Path, cache_dir: Path, batch_size: int = 256):
        print("Loading model weights...")
        retriever.load_weights(weights_dir)

        self.retriever = retriever
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._audio_files = None
        self._raw_audios = None
        self.processed_audios = None

    @property
    def audio_files(self) -> List[str]:
        if self._audio_files is None:
            self._audio_files = [str(path) for path in self.data_dir.glob("*.wav")]
            print(f"Found {len(self._audio_files)} audio files!")
        return self._audio_files

    @property
    def raw_audios(self) -> datasets.Dataset | datasets.DatasetDict:
        if self._raw_audios is None:
            paths = datasets.Dataset.from_dict({"path": self.audio_files})
            save_dir = self.cache_dir / paths._fingerprint

            print(f"Reading {len(paths)} audio files...")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if save_dir.is_dir():
                self._raw_audios = datasets.load_from_disk(str(save_dir))
            else:
                self._raw_audios = paths.map(
                    lambda row: {"audio": path_to_audio(row["path"])}, 
                )
                self._raw_audios.save_to_disk(str(save_dir))
            print("Done reading!")
        return self._raw_audios


    def swap_retriever(self, new_retriever: AudioTextRetriever, weights_dir: Optional[Path]):
        del self.retriever # free all references to old retriever
        gc.collect()
        torch.cuda.empty_cache()
        self.predict.cache_clear()

        if weights_dir is not None:
            new_retriever.load_weights(weights_dir)
        self.retriever = new_retriever

    def swap_data_dir(self, new_data_dir: Path):
        self.data_dir = new_data_dir
        self._audio_files = None
        self._raw_audios = None
        self.processed_audios = None

    @cache
    def predict(self, query: str, topk: int) -> List[Tuple[str, float]]:
        print("Querying audio files...")
        logits = self.retriever.query_audio(query, self.raw_audios["audio"], sampling_rate=16000, 
                                            audio_set_name=self.raw_audios._fingerprint, batch_size=self.batch_size).cpu()
        probs = logits.softmax(dim=0)
        top_k_probs, top_k_indices = probs.topk(topk)

        return [(self.audio_files[i], r.item()) for i, r in zip(top_k_indices, top_k_probs)]


if __name__=="__main__":
    parser = ArgumentParser(description="Uses a fine-tuned model to retrieve "
                            + "the most relevant text/audio from the given data.")
    parser.add_argument("--model", required=True, type=Path,
                        help="The directory containing the fine-tuned model.")
    parser.add_argument("--intype", required=True, choices=["text"], # only text to audio for now
                        help="Whether to perform text-to-audio or audio-to-text retrieval.")
    parser.add_argument("--data", required=True, type=Path,
                        help="Path containing the data to retrieve from. "
                        + "Should point to a directory containing .wav files for text-to-audio retrieval, "
                        + "or a .csv file with 1 caption per line for audio-to-text retrieval.")
    parser.add_argument("--query", required=True,
                        help="The query used for retrieval. Should be a string for text-to-audio "
                        + "retrieval, or the path to a .wav file for audio-to-text retrieval.")
    parser.add_argument("--topk", required=False, default=5, type=int, 
                        help="The number of results to return, ordered in descending relevance.")
    parser.add_argument("--batchsize", required=False, default=256, type=int,
                        help="The batch size to be used when passing inputs to the model for prediction.")
    parser.add_argument("--cachedir", required=False, default="/tmp/kokcz/.cache/huggingface/datasets/predict", type=Path,
                        help="The directory used to cache dataset objects.")

    args = parser.parse_args()
    ## do some additional data validation
    if not args.data.exists():
        raise ArgumentTypeError("The input to --data must be a valid path!")

    torch.set_grad_enabled(False)
    if args.intype == "text":
        if not (args.data.is_dir() and list(args.data.glob("*.wav"))): 
            raise ArgumentTypeError("The input to --data must be a directory containing at least 1 .wav file!")
        retriever = AudioTextRetriever(contrastiveCE).to(device)
        print(TextAudioRetrieval(retriever, args.model, args.data, args.cachedir, args.batchsize).predict(args.query, args.topk))
    else:
        if not (args.data.is_file() and args.data.suffix == ".csv"):
            raise ArgumentTypeError("The input to --data must be a .csv file with one caption per line!")
        args.query = Path(args.query)
        if not (args.query.is_file() and args.query.suffix == ".wav"):
            raise ArgumentTypeError("The input to --query must be a .wav file!")
            