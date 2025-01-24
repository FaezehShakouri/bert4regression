import argparse
from transformers import Trainer, TrainingArguments

from data_loader import load_data
from model import create_model
from utils import compute_metrics_for_regression
from data_loader import tokenizer

batch_size = 16
metric_name = "f1"


def main(train_file, validation_file, test_file):
    # Load datasets
    train_dataset, validation_dataset, test_dataset = load_data(train_file, validation_file, test_file)

    # Create model
    model = create_model()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
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
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_for_regression
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate(test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--validation_file', type=str, required=True, help='Path to the validation dataset')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test dataset')
    args = parser.parse_args()
    main(args.train_file, args.validation_file, args.test_file)