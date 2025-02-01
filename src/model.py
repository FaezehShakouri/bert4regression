from transformers import AutoModelForSequenceClassification, LongformerForSequenceClassification

BASE_MODEL = "bert-base-uncased"
LONGFORMER_MODEL = "allenai/longformer-base-4096"

def create_bert_model(model_name: str = BASE_MODEL) -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

def create_longformer_model(model_name: str = LONGFORMER_MODEL) -> LongformerForSequenceClassification:
    return LongformerForSequenceClassification.from_pretrained(model_name, num_labels=1)