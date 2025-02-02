import argparse
from datetime import datetime
import logging
import os
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
from utils import compute_metrics_for_regression

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelTrainer:
    def __init__(self, args):
        self.args = args
        self.model_type = args.model_type
        self.batch_size = 16
        self.metric_name = "f1"
        self.model = self._get_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.tokenizer_max_length = args.tokenizer_max_length
        self.trainer = None

        logger.info("Initializing trainer with:")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Tokenizer max length: {self.tokenizer_max_length}")
    
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

    def tokenize_dataset(self, data_file, mapping_function=None):
        if mapping_function is None:
            mapping_function = self.preprocess_data
            
        logger.info("Tokenizing dataset...")
        dataset = load_dataset('csv', data_files=data_file)['train']
        encoded_dataset = dataset.map(mapping_function, batched=True, remove_columns=dataset.column_names)
        encoded_dataset.set_format("torch")
        return encoded_dataset

    def split_data(self, train_size=0.7, validation_size=0.15, test_size=0.15):
        df = pd.read_csv(self.args.dataset_file)

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

        self.split_data()

        train_dataset = self.tokenize_dataset('data/splitted_data/train.csv')
        validation_dataset = self.tokenize_dataset('data/splitted_data/validation.csv')
        test_dataset = self.tokenize_dataset('data/splitted_data/test.csv')

        return train_dataset, validation_dataset, test_dataset

    def load_test_data(self, test_file=None):
        if test_file is None:
            test_file = self.args.test_file
        return self.tokenize_dataset(test_file, mapping_function=lambda x: self.preprocess_data(x, include_label=False))

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

    def train(self):
        # Add timestamp and model type to output directory if not specified
        if self.args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.args.output_dir = f"artifacts/output_{self.model_type}_{timestamp}"

        # Load datasets
        train_dataset, validation_dataset, test_dataset = self.load_data(self.args.dataset_file)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            eval_strategy='epoch',
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.args.num_epochs,
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
        if self.args.checkpoint:
            self.trainer.train(resume_from_checkpoint=self.args.checkpoint)
        else:
            self.trainer.train()

        # Evaluate the model
        self.trainer.evaluate(test_dataset)

        # Save model if specified
        if self.args.save_model:
            # Add timestamp and model type to model save path
            save_path = "model"  # Default path since save_model is now a bool flag
            model_save_path = f"artifacts/{save_path}_{self.model_type}_{timestamp}"
            self.trainer.save_model(model_save_path)

    def predict(self, test_file=None, output_file=None):
        if self.trainer is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Make predictions
        if test_file is None:
            test_file = self.args.test_file
        if output_file is None:
            output_file = self.args.predictions_output
        
        predictions = self.trainer.predict(self.load_test_data(test_file))
        self.save_predictions(predictions, output_file)

def main(args):
    trainer = ModelTrainer(args)
    trainer.train()
    trainer.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, default='data/data.mirror.csv', help='Path to the dataset')
    parser.add_argument('--model-type', type=str, default='roberta-base', help='Type of model to use')
    parser.add_argument('--tokenizer-max-length', type=int, default=512, help='Max length of the tokenizer')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory for output files')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--save-model', action='store_true', help='Whether to save the final model')
    parser.add_argument('--test-file', type=str, default='data/test.csv', help='File to make predictions on')
    parser.add_argument('--predictions-output', type=str, default=None, help='Where to save predictions')
    args = parser.parse_args()
    main(args)