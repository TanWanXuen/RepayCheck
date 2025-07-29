from typing import Any
import pandas as pd
from enum import Enum


class DatasetStage(Enum):
    RAW_TEXT = 1  # Only one stage needed now (raw data)


class DatasetBundle:
    def __init__(
        self,
        origination: pd.DataFrame,
        monthly_performance: pd.DataFrame,
    ):
        self.origination = origination
        self.monthly_performance = monthly_performance


class ModelInterface:

    def __init__(self):    
        assert isinstance(self.identifier_raw, str)
        assert self.input_features is not None

    @property
    def identifier_raw(self) -> str:
        """Unique identifier for raw file version"""
        pass

    @property
    def input_features(self) -> list[str]:
        """
        The list and order of input features required by the model.
        """
        pass

    @property
    def model(self) -> Any:
        """
        The actual ML model used for inference.
        """
        pass

    @property
    def categorical_encoder(self) -> Any:
        """
        Encoder for categorical features.
        """
        pass

    def infer(self, datasets: DatasetBundle) -> pd.DataFrame:
        """
        1. Preprocess the data from raw format.
        2. Run inference using the model.
        3. Return predictions.
        """
        processed_data = self.preprocess_data(datasets)
        predictions = self.model.predict(processed_data[self.input_features])
        return pd.DataFrame({"prediction": predictions})
