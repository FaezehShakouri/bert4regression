from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

BASE_MODEL = "bert-base-uncased"


def create_model(model_name: str = BASE_MODEL) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    return model