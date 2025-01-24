import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from model import BASE_MODEL

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def load_data(train_file, validation_file, test_file):
    train_dataset = tokenize_dataset(train_file, tokenizer)
    validation_dataset = tokenize_dataset(validation_file, tokenizer)
    test_dataset = tokenize_dataset(test_file, tokenizer)

    return train_dataset, validation_dataset, test_dataset

def preprocess_data(examples):
  text = examples["text"]
  label = examples['label']

  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
  encoding["label"] = label
  
  return encoding

def tokenize_dataset(file_path, tokenizer):
    dataset = load_dataset('json', data_files=file_path)['train']
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
    encoded_dataset.set_format("torch")
    return encoded_dataset
