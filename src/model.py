from transformers import AutoModelForSequenceClassification

BASE_MODEL = "bert-base-uncased"


def create_model(model_name=BASE_MODEL):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    return model