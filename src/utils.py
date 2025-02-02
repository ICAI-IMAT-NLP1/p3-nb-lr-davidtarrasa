from typing import List, Dict
import torch
import re
import string


def remove_punctuations(input_col):
    """To remove all the punctuations present in the text.Input the text column"""
    table = str.maketrans("", "", string.punctuation)
    return input_col.translate(table)


# Tokenizes a input_string. Takes a input_string (a sentence), splits out punctuation and contractions, and returns a list of
# strings, with each input_string being a token.
def tokenize(input_string):
    input_string = remove_punctuations(input_string)
    input_string = re.sub(r"[^A-Za-z0-9(),.!?\'`\-\"]", " ", input_string)
    input_string = re.sub(r"\'s", " 's", input_string)
    input_string = re.sub(r"\'ve", " 've", input_string)
    input_string = re.sub(r"n\'t", " n't", input_string)
    input_string = re.sub(r"\'re", " 're", input_string)
    input_string = re.sub(r"\'d", " 'd", input_string)
    input_string = re.sub(r"\'ll", " 'll", input_string)
    input_string = re.sub(r"\.", " . ", input_string)
    input_string = re.sub(r",", " , ", input_string)
    input_string = re.sub(r"!", " ! ", input_string)
    input_string = re.sub(r"\?", " ? ", input_string)
    input_string = re.sub(r"\(", " ( ", input_string)
    input_string = re.sub(r"\)", " ) ", input_string)
    input_string = re.sub(r"\-", " - ", input_string)
    input_string = re.sub(r"\"", ' " ', input_string)
    # We may have introduced double spaces, so collapse these down
    input_string = re.sub(r"\s{2,}", " ", input_string)
    return list(filter(lambda x: len(x) > 0, input_string.split(" ")))


class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[str]): List of words.
        label (int): Sentiment label (0 for negative, 1 for positive).
    """

    def __init__(self, words: List[str], label: int):
        self._words = words
        self._label = label

    def __repr__(self) -> str:
        if self.label is not None:
            return f"{self.words}; label={self.label}"
        else:
            return f"{self.words}, no label"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, SentimentExample):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.words == other.words and self.label == other.label

    @property
    def words(self):
        return self._words

    @words.setter
    def words(self, value):
        raise NotImplemented

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        raise NotImplemented


def evaluate_classification(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate classification metrics including accuracy, precision, recall, and F1-score.

    Args:
        predictions (torch.Tensor): Predictions from the model.
        labels (torch.Tensor): Actual ground truth labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    metrics: Dict[str, float] = {}

    TP = ((predictions == 1) & (labels == 1)).sum().item()
    TN = ((predictions == 0) & (labels == 0)).sum().item()
    FP = ((predictions == 1) & (labels == 0)).sum().item()
    FN = ((predictions == 0) & (labels == 1)).sum().item()

    # Accuracy
    accuracy = (TP + TN) / len(labels)
    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # F1 Score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    return metrics
