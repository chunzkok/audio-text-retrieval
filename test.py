import torch
import numpy as np

from data_parsers.Parser import ClothoParser, SplitType
from utils.DataTools import AudioTextDataCollator
from utils.LossFunctions import contrastiveCE
from models.AudioTextRetriever import AudioTextRetriever

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ClothoParser("../datasets/clotho")
train_set = parser.to_hf(SplitType.DEV)
val_set = parser.to_hf(SplitType.VAL)

retriever = AudioTextRetriever(contrastiveCE).to(device)
collator = AudioTextDataCollator(2, retriever.AudioEncoder.preprocess, retriever.TextEncoder.preprocess)
print(retriever(**collator(train_set.take(3))))