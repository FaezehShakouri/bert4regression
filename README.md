# Text Regression Using Transformers With One Line Summary of Projects

This project employs a transformer-based text regression approach to analyze Git project logs and predict funding amounts. By utilizing models like BERT, we summarize project activities and assess project similarities, achieving a mean squared error (MSE) of 0.0206.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Metrics](#metrics)


## Installation

To set up the project, clone the repository and install the required packages:
```bash
git clone <repository-url>
cd <repository-directory>/model
pip install -r requirements.txt
```

## Usage

To train the model, run the following command:

```bash
python src/main.py
```

Make sure to adjust any parameters in the script as needed.

## Metrics

The model evaluates its performance using various metrics, including:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R2)**

These metrics provide insights into the model's accuracy and effectiveness in generating predictions.
