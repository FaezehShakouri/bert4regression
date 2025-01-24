# BERT-based Regression Model

This project implements a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers) to perform regression tasks. 

**Note:** The dataset used in this project is only test and sample data, intended for demonstration purposes.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Metrics](#metrics)


## Installation

To set up the project, clone the repository and install the required packages:
```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Usage

To train the model, run the following command:

```bash
python src/main.py --train_file <path-to-train-file> --validation_file <path-to-validation-file> --test_file <path-to-test-file>
```

Make sure to adjust any parameters in the script as needed.

## Metrics

The model evaluates its performance using various metrics, including:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R2)**

These metrics provide insights into the model's accuracy and effectiveness in generating predictions.
