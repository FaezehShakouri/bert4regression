import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from model import BASE_MODEL

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
logger = logging.getLogger(__name__)


def load_data(dataset_file):
    logger.info("Loading datasets...")

    split_data(dataset_file)

    train_dataset = tokenize_dataset('data/splitted_data/train.csv', tokenizer)
    validation_dataset = tokenize_dataset('data/splitted_data/validation.csv', tokenizer)
    test_dataset = tokenize_dataset('data/splitted_data/test.csv', tokenizer)

    return train_dataset, validation_dataset, test_dataset


def load_predict_data(file_path):
    data = tokenize_dataset(file_path, tokenizer, mapping_function=preprocess_predict_data)
    return data

def split_data(file_path, train_size=0.7, validation_size=0.15, test_size=0.15):
    df = pd.read_csv(file_path)

    logger.info("Splitting datasets to train, validation and test...")
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=42, shuffle=True)
    validation_size_adjusted = validation_size / (validation_size + test_size)
    validation_df, test_df = train_test_split(temp_df, train_size=validation_size_adjusted, random_state=42, shuffle=True)

    # Ensure the output directory exists
    output_dir = 'data/splitted_data'
    os.makedirs(output_dir, exist_ok=True)

    # Save the splitted datasets to new CSV files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    validation_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    logger.info("Data split completed:")
    logger.info(f"Training data: {len(train_df)} rows")
    logger.info(f"Validation data: {len(validation_df)} rows")
    logger.info(f"Test data: {len(test_df)} rows")

def preprocess_data(examples):
    logger.info("Preprocessing data...")
    project_as = examples["project_a"]
    project_bs = examples["project_b"]
    label = examples['weight_a']

    text = [f"Project A: {project_a}, Project B: {project_b}" for project_a, project_b in zip(project_as, project_bs)]

    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    encoding["label"] = label
  
    return encoding

def preprocess_predict_data(examples):
    logger.info("Preprocessing predict data...")
    project_as = examples["project_a"]
    project_bs = examples["project_b"]

    text = [f"Project A: {project_a}, Project B: {project_b}" for project_a, project_b in zip(project_as, project_bs)]

    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
  
    return encoding

def tokenize_dataset(file_path, tokenizer, mapping_function=preprocess_data):
    logger.info("Tokenizing dataset...")
    dataset = load_dataset('csv', data_files=file_path)['train']
    encoded_dataset = dataset.map(mapping_function, batched=True, remove_columns=dataset.column_names)
    encoded_dataset.set_format("torch")
    return encoded_dataset
