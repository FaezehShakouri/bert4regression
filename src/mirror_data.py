import pandas as pd
import logging

logger = logging.getLogger(__name__)

def mirror_data(input_file='data/data.csv', output_file='data/data.mirror.csv'):
    logger.info("Loading original data...")
    df = pd.read_csv(input_file)
    
    logger.info("Creating mirrored data...")
    # Create mirrored data by swapping project_a/b and weight_a/b
    mirrored_df = df.copy()
    mirrored_df['project_a'] = df['project_b']
    mirrored_df['project_b'] = df['project_a'] 
    mirrored_df['weight_a'] = df['weight_b']
    mirrored_df['weight_b'] = df['weight_a']

    # Combine original and mirrored data
    combined_df = pd.concat([df, mirrored_df], ignore_index=True)
    
    logger.info(f"Original data size: {len(df)}")
    logger.info(f"Combined data size: {len(combined_df)}")
    
    # Save combined data
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Saved mirrored data to {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mirror_data()
