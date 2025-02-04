import pandas as pd
import logging

logger = logging.getLogger(__name__)

def mirror_data(input_file='data/data.enriched.csv', output_file='data/data.enriched.mirror.csv'):
    logger.info("Loading original data...")
    df = pd.read_csv(input_file)
    
    logger.info("Creating mirrored data...")
    # Create mirrored data by swapping project_a/b, weight_a/b and metrics
    mirrored_df = df.copy()
    mirrored_df['project_a'] = df['project_b']
    mirrored_df['project_b'] = df['project_a']
    mirrored_df['weight_a'] = df['weight_b']
    mirrored_df['weight_b'] = df['weight_a']
    mirrored_df['star_count_a'] = df['star_count_b']
    mirrored_df['star_count_b'] = df['star_count_a']
    mirrored_df['fork_count_a'] = df['fork_count_b']
    mirrored_df['fork_count_b'] = df['fork_count_a']
    
    # Combine original and mirrored data
    combined_df = pd.concat([df, mirrored_df], ignore_index=True)
    
    logger.info(f"Original data size: {len(df)}")
    logger.info(f"Combined data size: {len(combined_df)}")
    
    # Save combined data with proper quoting
    combined_df.to_csv(output_file, index=False, quoting=1)  # QUOTE_ALL mode
    logger.info(f"Saved mirrored data to {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mirror_data()
