import torch
import numpy as np

from data_parsers.Parser import ClothoParser, SplitType
from utils.DataTools import AudioTextDataCollator
from utils.LossFunctions import contrastiveCE
from models.AudioTextRetriever import AudioTextRetriever
from transformers import EvalPrediction, Trainer, TrainingArguments
from typing import Dict

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = ClothoParser("../datasets/clotho")
    train_set = parser.to_hf(SplitType.DEV)
    val_set = parser.to_hf(SplitType.VAL)

    collator = AudioTextDataCollator(3)
    retriever = AudioTextRetriever(contrastiveCE).to(device)

    def compute_metrics(output: EvalPrediction) -> Dict[str, float]:
        preds = output.predictions[0] if type(output[0]) == tuple else output.predictions
        assert type(preds) == np.ndarray # should get a (B, 2, 1024) size tensor where B is the batch size
        audio_embed = torch.tensor(preds[:, 0, :], device=device)
        text_embed = torch.tensor(preds[:, 1, :], device=device)
        labels = torch.tensor(output.label_ids, device=device)
        return {
            "Contrastive Loss": contrastiveCE(audio_embed, text_embed, labels).item()
        }


    train_args = TrainingArguments(
        output_dir="../train_out",
        overwrite_output_dir=True,
        group_by_length=False,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=50,
        fp16=False,
        save_steps=1024,
        eval_steps=512,
        logging_steps=64,
        optim="adamw_torch",
        learning_rate=0.2,
        save_total_limit=2,
        dataloader_num_workers=6,
        load_best_model_at_end=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=retriever,
        args=train_args,
        data_collator=collator,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=val_set
    )

    trainer.train()