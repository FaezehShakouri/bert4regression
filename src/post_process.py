import pandas as pd
import logging

logger = logging.getLogger(__name__)

def post_process_predictions(input_file, output_file):
    logger.info(f"Loading predictions from {input_file}")
    df = pd.read_csv(input_file)
    
    # Clip predictions between 0 and 1
    logger.info("Clipping predictions to [0,1] range")
    df['pred'] = df['pred'].clip(lower=0, upper=1)
    
    # Save processed predictions
    df.to_csv(output_file, index=False)
    logger.info(f"Saved processed predictions to {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    input_file = "artifacts/predictions_bert-base-uncased_20250203_221716.csv"  
    output_file = "artifacts/predictions_bert-base-uncased_20250203_221716_postprocessed.csv"
    post_process_predictions(input_file, output_file)
