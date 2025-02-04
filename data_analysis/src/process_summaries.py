import csv
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from typing import Dict, List
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_project_summaries(project_name: str) -> str:
    """Load all summary files for a given project and combine them."""
    summary_dir = f"data/summaries/{project_name}"
    if not os.path.exists(summary_dir):
        logger.warning(f"No summaries found for {project_name}")
        return ""
    
    combined_summary = ""
    summary_files = sorted([f for f in os.listdir(summary_dir) if f.startswith("log_summary_")])
    
    for file_name in summary_files:
        file_path = os.path.join(summary_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                combined_summary += file.read() + "\n"
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    return combined_summary

def generate_project_description(model: genai.GenerativeModel, project_name: str, summary: str) -> str:
    """Generate a one-line description for a project using Gemini."""
    prompt = f"""Based on the following git commit history summary, generate a single-line description (max 20 words) 
    that captures the most important aspects of the project '{project_name}'. Focus on its main purpose, 
    technology stack, and key features. The description should be concise and informative.

    Summary:
    {summary}

    Format your response as a single line without any prefixes or quotes.
    """
    
    try:
        response = model.generate_content(prompt)
        time.sleep(1)  # Rate limiting
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating description for {project_name}: {e}")
        return f"Error: Unable to generate description for {project_name}"
def process_dataset(input_path: str, output_dir: str) -> None:
    """Process the dataset and create separate files for descriptions and relationships.
    
    Creates two files:
    1. project_descriptions.json - Contains all unique project descriptions
    2. project_relationships.csv - Contains the relationships with project IDs
    
    Descriptions are written incrementally to disk as they are generated.
    """
    # Load environment variables and configure Gemini
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Read the dataset
    df = pd.read_csv(input_path)
    
    # Get unique projects
    unique_projects = set(df['project_a'].unique()) | set(df['project_b'].unique())
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    descriptions_path = os.path.join(output_dir, "project_descriptions.json")
    
    # Initialize or load existing descriptions
    if os.path.exists(descriptions_path):
        with open(descriptions_path, 'r') as f:
            descriptions = json.load(f)
    else:
        descriptions = {}
    
    # Process each project incrementally
    for project in unique_projects:
        if project in descriptions:
            logger.info(f"Description already exists for {project}")
            continue
            
        description_path = f"data/summaries/{project}/description.txt"
        
        if os.path.exists(description_path):
            with open(description_path, 'r') as f:
                description = f.read().strip()
            logger.info(f"Loaded existing description for {project}")
        else:
            logger.info(f"Processing {project}")
            summary = load_project_summaries(project)
            
            if summary:
                description = generate_project_description(model, project, summary)
                
                # Store individual description
                os.makedirs(os.path.dirname(description_path), exist_ok=True)
                with open(description_path, 'w') as f:
                    f.write(description)
            else:
                logger.warning(f"No summary found for {project}")
                description = f"No description available for {project}"
        
        # Update descriptions dict and write to disk
        descriptions[project] = description
        with open(descriptions_path, 'w') as f:
            json.dump(descriptions, f, indent=2)
        
        logger.info(f"Saved description for {project}")
    
    # Save relationships CSV with original data
    relationships_path = os.path.join(output_dir, "project_relationships.csv")
    df.to_csv(relationships_path, index=False)
    
    logger.info(f"Saved project relationships to {relationships_path}")

if __name__ == "__main__":
    process_dataset("data/dataset.csv", "data/processed")