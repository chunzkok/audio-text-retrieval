import numpy as np
import torch

from argparse import ArgumentParser, ArgumentTypeError
from models.AudioTextRetriever import AudioTextRetriever
from utils.DataTools import path_to_audio
from utils.LossFunctions import contrastiveCE
from pathlib import Path
from typing import List, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextAudioRetrieval:
    @classmethod
    def predict(cls, model: Path, data: Path, query: str, topk: int, batch_size: int = 256) -> List[str]:
        retriever = AudioTextRetriever(contrastiveCE).to(device)
        print("Loading model weights...")
        retriever.load_weights(model)
        audio_files = [str(path) for path in data.glob("*.wav")]
        print(f"Reading {len(audio_files)} audio files...")
        raw_audios = path_to_audio(audio_files)
        with torch.no_grad():
            print("Processing inputs...")
            split_embeds = [cls._extract_embeds(retriever, raw_audios[i:i+batch_size], query) 
                            for i in range(0, len(raw_audios), batch_size)]
            embeds = torch.cat(split_embeds, dim=0)
            print("Calculating relevance...")
            audio_embed = embeds[:, 0, :]
            text_embed = embeds[:, 1, :]
            logits = torch.multiply(audio_embed, text_embed).sum(dim=1)
            _, top_k_indices = logits.topk(topk)

        return [audio_files[i] for i in top_k_indices]

    @staticmethod
    def _extract_embeds(retriever: AudioTextRetriever, raw_audios: np.ndarray | List[np.ndarray], query: str) -> torch.Tensor:
        with torch.no_grad():
            audio_inputs = retriever.AudioEncoder.preprocess(raw_audios, sampling_rate=16_000, device=device)
            text_inputs = retriever.TextEncoder.preprocess([query] * len(raw_audios), device=device)

            # With return_dict=True, we know for sure we are getting a dictionary (more precisely: an AudioTextOutput)
            return retriever.forward(audio_inputs, text_inputs, return_dict=True).embeddings # type: ignore





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

    args = parser.parse_args()
    ## do some additional data validation
    if not args.data.exists():
        raise ArgumentTypeError("The input to --data must be a valid path!")

    if args.intype == "text":
        if not (args.data.is_dir() and list(args.data.glob("*.wav"))): 
            raise ArgumentTypeError("The input to --data must be a directory containing at least 1 .wav file!")
        print(TextAudioRetrieval.predict(args.model, args.data, args.query, args.topk, args.batchsize))
    else:
        if not (args.data.is_file() and args.data.suffix == ".csv"):
            raise ArgumentTypeError("The input to --data must be a .csv file with one caption per line!")
        args.query = Path(args.query)
        if not (args.query.is_file() and args.query.suffix == ".wav"):
            raise ArgumentTypeError("The input to --query must be a .wav file!")
            