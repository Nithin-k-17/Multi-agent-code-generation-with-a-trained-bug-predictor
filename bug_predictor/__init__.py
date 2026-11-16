"""Bug-prediction agent: a fine-tuned CodeBERT classifier that scores Python code for bug risk."""
from .predictor import BugPredictor

__all__ = ["BugPredictor"]
