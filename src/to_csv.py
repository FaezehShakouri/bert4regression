import pandas as pd

# Load the datasets
test_df = pd.read_csv('data/test.csv')
huggingface_df = pd.read_csv('data/data.huggingface.csv')

# Create mapping from id to star/fork counts using huggingface data
metrics_map = {}
for _, row in huggingface_df.iterrows():
    metrics_map[row['id']] = {
        'star_count_a': row['star_count_a'],
        'fork_count_a': row['fork_count_a'],
        'star_count_b': row['star_count_b'],
        'fork_count_b': row['fork_count_b']
    }

# Add metrics columns
star_counts_a = []
fork_counts_a = []
star_counts_b = []
fork_counts_b = []

# Iterate through test data to get metrics
for _, row in test_df.iterrows():
    metrics = metrics_map.get(row['id'], {
        'star_count_a': 0,
        'fork_count_a': 0,
        'star_count_b': 0,
        'fork_count_b': 0
    })
    
    star_counts_a.append(metrics['star_count_a'])
    fork_counts_a.append(metrics['fork_count_a'])
    star_counts_b.append(metrics['star_count_b'])
    fork_counts_b.append(metrics['fork_count_b'])

# Add new columns to test_df
test_df['star_count_a'] = star_counts_a
test_df['fork_count_a'] = fork_counts_a
test_df['star_count_b'] = star_counts_b
test_df['fork_count_b'] = fork_counts_b

# Save enriched test data
test_df.to_csv('data/test.enriched.csv', index=False)
