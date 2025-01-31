import pandas as pd
import sys

def check_csv_format(file_path):
    try:
        # Try reading the CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['project_a', 'project_b', 'weight_a', 'weight_b']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        print(f"\nChecking CSV file: {file_path}")
        print("-" * 50)
        
        if missing_columns:
            print("✗ Missing required columns:", ", ".join(missing_columns))
            print(f"Required format: {','.join(required_columns)}")
            print(f"Found columns: {', '.join(df.columns)}")
            return
            
        # Check for extra columns
        extra_columns = [col for col in df.columns if col not in required_columns]
        if extra_columns:
            print("✗ Extra columns found:", ", ".join(extra_columns))
            print(f"Required format: {','.join(required_columns)}")
            return
            
        # Get total number of rows
        total_rows = len(df)
        print(f"Total rows: {total_rows}")
        
        # Check each column for empty values
        issues_found = False
        for column in required_columns:
            empty_rows = df[df[column].isna()].index.tolist()
            if empty_rows:
                issues_found = True
                print(f"Column '{column}' has {len(empty_rows)} empty values at rows:")
                print(f"Line numbers: {[x + 2 for x in empty_rows]}")  # Adding 2 to account for 0-based index and header row
                print()
        
        if not issues_found:
            print("✓ CSV format is valid with all required columns and no empty values.")
        else:
            print("✗ Empty values found in the CSV file. Please fix the empty values.")
            
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except pd.errors.ParserError:
        print("Error: Unable to parse the CSV file. Please check if it's a valid CSV format.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python csv_checker.py <path_to_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    check_csv_format(csv_file)
