import argparse
from transformers import Trainer, TrainingArguments

from data_loader import load_data
from model import create_model
from utils import compute_metrics_for_regression
from data_loader import tokenizer

batch_size = 16
metric_name = "f1"


def main(dataset_file):
    # Load datasets
    train_dataset, validation_dataset, test_dataset = load_data(dataset_file)

    # Create model
    model = create_model()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_for_regression
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate(test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, default='data/data.csv', help='Path to the dataset')
    args = parser.parse_args()
    main(args.dataset_file)