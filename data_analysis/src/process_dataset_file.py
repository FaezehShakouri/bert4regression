import csv
import json

# Set to store unique repo URLs
unique_repos = set()

# Read the CSV file
with open('data/dataset.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        # Add both repository URLs from each row
        unique_repos.add(row[1])  # First repo URL
        unique_repos.add(row[2])  # Second repo URL

# Convert set to sorted list for consistent output
repos_list = sorted(list(unique_repos))

# Save to repos.json
with open('data/repos.json', 'w') as outfile:
    json.dump(repos_list, outfile, indent=2)