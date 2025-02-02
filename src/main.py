import argparse
from datetime import datetime
import logging
import os
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification

from model import create_bert_model, create_longformer_model, BASE_MODEL
from utils import compute_metrics_for_regression

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_type, batch_size=16, tokenizer_max_length=512):
        self.model_type = model_type
        self.batch_size = batch_size
        self.metric_name = "f1"
        self.model = self._get_model()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer_max_length = tokenizer_max_length
        self.trainer = None
    
    def _get_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_type, num_labels=1)

    def preprocess_data(self, examples, include_label=True):
        logger.info("Preprocessing data...")
        project_as = examples["project_a"]
        project_bs = examples["project_b"]

        text = [f"Project A: {project_a}, Project B: {project_b}" for project_a, project_b in zip(project_as, project_bs)]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.tokenizer_max_length)
        
        if include_label:
            encoding["label"] = examples['weight_a']
      
        return encoding

    def tokenize_dataset(self, file_path, mapping_function=None):
        if mapping_function is None:
            mapping_function = self.preprocess_data
            
        logger.info("Tokenizing dataset...")
        dataset = load_dataset('csv', data_files=file_path)['train']
        encoded_dataset = dataset.map(mapping_function, batched=True, remove_columns=dataset.column_names)
        encoded_dataset.set_format("torch")
        return encoded_dataset

    def split_data(self, file_path, train_size=0.7, validation_size=0.15, test_size=0.15):
        df = pd.read_csv(file_path)

        logger.info("Splitting datasets to train, validation and test...")
        train_df, temp_df = train_test_split(df, train_size=train_size, random_state=42, shuffle=True)
        validation_size_adjusted = validation_size / (validation_size + test_size)
        validation_df, test_df = train_test_split(temp_df, train_size=validation_size_adjusted, random_state=42, shuffle=True)

        output_dir = 'data/splitted_data'
        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        validation_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

        logger.info("Data split completed:")
        logger.info(f"Training data: {len(train_df)} rows")
        logger.info(f"Validation data: {len(validation_df)} rows")
        logger.info(f"Test data: {len(test_df)} rows")

    def load_data(self, dataset_file):
        logger.info("Loading datasets...")

        self.split_data(dataset_file)

        train_dataset = self.tokenize_dataset('data/splitted_data/train.csv')
        validation_dataset = self.tokenize_dataset('data/splitted_data/validation.csv')
        test_dataset = self.tokenize_dataset('data/splitted_data/test.csv')

        return train_dataset, validation_dataset, test_dataset

    def load_test_data(self, file_path):
        return self.tokenize_dataset(file_path, mapping_function=lambda x: self.preprocess_data(x, include_label=False))

    def save_predictions(self, predictions, output_file):
        # Load the original test data to get IDs
        predict_df = pd.read_csv("data/test.original.csv")
        
        # Create DataFrame with predictions
        results_df = pd.DataFrame({
            'id': predict_df['id'],
            'pred': predictions.predictions.flatten()
        })
        
        # Add timestamp and model type to filename
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"artifacts/predictions_{self.model_type}_{timestamp}.csv"
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)

    def train(self, args):
        # Add timestamp and model type to output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"artifacts/{args.output_dir}_{self.model_type}_{timestamp}"

        # Load datasets
        train_dataset, validation_dataset, test_dataset = self.load_data(args.dataset_file)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            eval_strategy='epoch',
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
        )

        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics_for_regression
        )
        
        # Train model
        if args.checkpoint:
            self.trainer.train(resume_from_checkpoint=args.checkpoint)
        else:
            self.trainer.train()

        # Evaluate the model
        self.trainer.evaluate(test_dataset)

        # Save model if specified
        if args.save_model:
            # Add timestamp and model type to model save path
            save_path = args.save_model.rstrip("/")
            args.save_model = f"artifacts/{save_path}_{self.model_type}_{timestamp}"
            self.trainer.save_model(args.save_model)

    def predict(self, test_file, output_file):
        if self.trainer is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Make predictions
        predictions = self.trainer.predict(self.load_test_data(test_file))
        self.save_predictions(predictions, output_file)

def main(args):
    trainer = ModelTrainer(args.model_type)
    trainer.train(args)
    trainer.predict(args.test_file, args.predictions_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, default='data/data.csv', help='Path to the dataset')
    parser.add_argument('--model-type', type=str, default='longformer', choices=['bert', 'longformer'], help='Type of model to use')
    parser.add_argument('--tokenizer-max-length', type=int, default=512, help='Max length of the tokenizer')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory for output files')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--save-model', type=str, help='Path to save the final model')
    parser.add_argument('--test-file', type=str, default='data/test.log.csv', help='File to make predictions on')
    parser.add_argument('--predictions-output', type=str, default='predictions.log.csv', help='Where to save predictions')
    args = parser.parse_args()
    main(args)