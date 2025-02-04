import pandas as pd
import itertools

# Load predictions and original data
predictions_df = pd.read_csv("artifacts/predictions_bert-base-uncased_20250203_221716_postprocessed.csv")
test_data = pd.read_csv("data/test.original.csv")

# Create a dictionary mapping id to prediction
id_to_pred = dict(zip(predictions_df['id'], predictions_df['pred']))

# Get pairs from original data
pairs = []
for _, row in test_data.iterrows():
    if pd.notna(row['project_a']) and pd.notna(row['project_b']):
        pairs.append((row['id'], row['project_a'], row['project_b']))

# Build graph of connected pairs
connected_pairs = set()
for id1, proj_a, proj_b in pairs:
    connected_pairs.add((id1, proj_a))
    connected_pairs.add((id1, proj_b))

# Function to check transitivity for a triplet
def check_transitivity(id1, id2, id3):
    pred1 = id_to_pred[id1]
    pred2 = id_to_pred[id2]
    pred3 = id_to_pred[id3]
    
    # Check if pred1 < pred2 and pred2 < pred3
    if pred1 < pred2 and pred2 < pred3:
        # If true, pred1 should be < pred3
        if not pred1 < pred3:
            return False
    return True

# Find all inconsistencies among connected triplets
inconsistencies = []
for id1, id2, id3 in itertools.combinations(id_to_pred.keys(), 3):
    # Check if these IDs form connected pairs in original data
    if ((id1, id2) in connected_pairs and 
        (id2, id3) in connected_pairs and
        (id1, id3) in connected_pairs):
        if not check_transitivity(id1, id2, id3):
            inconsistencies.append((id1, id2, id3))

# Print results
print(f"Found {len(inconsistencies)} transitivity violations")
print("\nDetailed violations:")
for id1, id2, id3 in inconsistencies:
    print(f"\nViolation between IDs {id1}, {id2}, {id3}:")
    print(f"Pred({id1}) = {id_to_pred[id1]:.4f}")
    print(f"Pred({id2}) = {id_to_pred[id2]:.4f}")
    print(f"Pred({id3}) = {id_to_pred[id3]:.4f}")

