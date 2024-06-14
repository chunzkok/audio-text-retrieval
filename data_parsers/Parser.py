import pandas as pd

from abc import ABC, abstractmethod
from datasets import Dataset
from enum import Enum
from pathlib import Path
from typing import Optional

class SplitType(Enum):
    DEV = 1
    VAL = 2
    TEST = 3

class Parser(ABC):
    """
    An abstract class that defines interfaces to work with the defined data collators 
    under the /utils/DataTools.py file.
    This class should implement methods to export the parsed data in either pd.DataFrame or
    datasets.Dataset form.
    The exported data should contain 2-3 columns: 
        1. path to the audio file (str), and
        2. caption(s) (str | List[str])
        3. split (Optional[SplitType])
            - only present when no arguments are passed into to_hf or to_pd

    Methods
    -------
    to_hf(split: Optional[SplitType]):
        Abstract method.
        Should return a dataset.Datset instance containing the columns as described.

    to_pd(split: Optional[SplitType]):
        Abstract method.
        Should return a pd.DataFrame instance containing the columns as described.
    """

    @abstractmethod
    def to_hf(self, split: Optional[SplitType]) -> Dataset:
        raise NotImplementedError("Parser::to_hf is not implemented yet!")

    @abstractmethod
    def to_pd(self, split: Optional[SplitType]) -> pd.DataFrame:
        raise NotImplementedError("Parser::to_pd is not implemented yet!")



class ClothoParser(Parser):
    """
    A parser that parses the Clotho v2.1 dataset into different formats (pd.DataFrame or 
    datasets.arrow_dataset.Dataset).

    Attributes
    ----------
    clotho_path: str | pathlib.Path
        Path to the Clotho v2.1 dataset.
        This path should point to a directory organised as such:
        clotho_path
        ├── development/
        ├── validation/
        ├── evaluation/
        ├── clotho_captions_development.csv
        ├── clotho_captions_validation.csv
        └── clotho_captions_evaluation.csv
        where the subdirectories contain the .wav files.

    Methods
    -------
    to_hf():
        Returns a datasets.Dataset containing three columns: path, caption and split.

    to_pd():
        Returns a pd.DataFrame containing three columns: path, caption and split.
    """

    def __init__(self, clotho_path: str | Path):
        self.root_path = Path(clotho_path)
        self.file_paths = []
        self.captions = []
        self.split = []
        split_mapping = {
            "development": "DEV",
            "validation": "VAL",
            "evaluation": "TEST"
        }

        for split_name in ("development", "validation", "evaluation"):
            caps_path = self.root_path.joinpath(f"clotho_captions_{split_name}.csv")
            df = pd.read_csv(caps_path)
            
            self.file_paths.extend([str(caps_path.joinpath(fname)) for fname in df["file_name"]])

            captions_df = df.iloc[:, 1:]
            self.captions.extend(captions_df.apply(lambda row: list(row), axis=1))

            self.split.extend([split_mapping[split_name]] * len(df))
            assert len(self.file_paths) == len(self.captions) == len(self.split)

    def to_hf(self, split: Optional[SplitType] = None) -> Dataset:
        return Dataset.from_pandas(self.to_pd(split))

    def to_pd(self, split: Optional[SplitType] = None) -> pd.DataFrame:
        df = pd.DataFrame({
                "path": self.file_paths,
                "caption": self.captions,
                "split": self.split
            })

        if split != None:
            df = df[df["split"] == split.name] # filter DEV rows
            df = df.iloc[:, :2] # remove `split`` col

        return df.reset_index(drop=True)