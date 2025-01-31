import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

from model import BASE_MODEL

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
logger = logging.getLogger(__name__)


def load_data(train_file, validation_file, test_file):
    logger.info("Loading datasets...")
    train_dataset = tokenize_dataset(train_file, tokenizer)
    validation_dataset = tokenize_dataset(validation_file, tokenizer)
    test_dataset = tokenize_dataset(test_file, tokenizer)

    return train_dataset, validation_dataset, test_dataset

def preprocess_data(examples):
    logger.info("Preprocessing data...")
    project_as = examples["project_a"]
    project_bs = examples["project_b"]
    label = examples['weight_a']

    text = [f"Project A: {project_a}, Project B: {project_b}" for project_a, project_b in zip(project_as, project_bs)]

    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    encoding["label"] = label
  
    return encoding

def tokenize_dataset(file_path, tokenizer):
    logger.info("Tokenizing dataset...")
    dataset = load_dataset('csv', data_files=file_path)['train']
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
    encoded_dataset.set_format("torch")
    return encoded_dataset
