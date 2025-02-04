import logging
from dotenv import load_dotenv
import os
import time
import google.generativeai as genai
import json
from typing import List

from git import GitLogAnalyzer

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[
        logging.StreamHandler(),  
    ]
)
logger = logging.getLogger(__name__)

def process_repository(git_remote_url: str, model: genai.GenerativeModel) -> bool:
    """Process a single repository and generate summaries.
    
    Args:
        git_remote_url: URL of the git repository
        model: Configured Gemini model instance
        
    Returns:
        bool: True if processing succeeded, False if it failed
    """
    try:
        repo_name = git_remote_url.split("/")[-1]  # Extract repository name from URL
        logger.info(f"Processing repository: {repo_name}")
        
        # Check if a summary already exists for this repository
        summary_dir = f"data/summaries/{repo_name}/log_summary_0.txt"
        if os.path.exists(summary_dir):
            logger.info(f"Summary already exists for {repo_name}. Skipping.")
            # Check if there are more than one log_summary files
            summary_files = [f for f in os.listdir(f"data/summaries/{repo_name}") if f.startswith("log_summary_") and f.endswith(".txt")]
            if len(summary_files) > 1:
                if os.path.exists(f"data/summaries/{repo_name}/log_summary_0_main.txt"):
                    return True
                # Prepare prompt for Gemini to merge summaries
                prompt = "Merge the following summaries with the same format and structure into a single summary:\n"
                for file in summary_files:
                    with open(f"data/summaries/{repo_name}/{file}", "r") as f:
                        prompt += f.read() + "\n"
                
                # Generate response using Gemini
                response = model.generate_content(prompt)
                
                # Rename the original log_summary_0.txt to log_summary_0_main.txt
                os.rename(f"data/summaries/{repo_name}/log_summary_0.txt", f"data/summaries/{repo_name}/log_summary_0_main.txt")
                logger.info(f"Renamed log_summary_0.txt to log_summary_0_main.txt")
                
                # Store the merged summary in log_summary_0.txt
                with open(f"data/summaries/{repo_name}/log_summary_0.txt", "w") as file:
                    file.write(response.text)
                logger.info(f"Stored the merged summary in log_summary_0.txt")
                
            return True
        
        git_log_analyzer = GitLogAnalyzer(git_remote_url)
        git_log = git_log_analyzer.get_git_log()
        commits = git_log_analyzer.split_log_by_max_token_size(git_log, 400000)

        # Create a directory for the repository if it doesn't exist
        os.makedirs(f"data/summaries/{repo_name}", exist_ok=True)

        for period, commit_log in enumerate(commits):
            # Check if file already exists
            file_name = f"data/summaries/{repo_name}/log_summary_{period}.txt"
            if os.path.exists(file_name):
                logger.info(f"Skipping period {period} - file already exists")
                continue
            
            prompt = git_log_analyzer.analyze_log_chunk_promt(commit_log, "")
            logger.info(f"Waiting for Gemini...")

            time.sleep(2)

            # Generate response using Gemini
            response = model.generate_content(prompt)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            
            # Create a new text file for each response
            with open(file_name, "w") as file:
                file.write(response.text)

        return True
        
    except Exception as e:
        logger.error(f"Error processing repository {git_remote_url}: {str(e)}")
        return False

def process_repositories(repositories: List[str], max_retries: int = 3) -> None:
    """Process multiple repositories with retry logic.
    
    Args:
        repositories: List of repository URLs to process
        max_retries: Maximum number of retry attempts per repository
    """
    # Load environment variables
    load_dotenv()

    # Configure Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-pro')

    total_repos = len(repositories)
    logger.info(f"Starting processing {total_repos} repositories.")

    for index, git_remote_url in enumerate(repositories, start=1):
        retries = 0
        success = False
        
        while not success and retries < max_retries:
            if retries > 0:
                logger.info(f"Retry attempt {retries} for repository {git_remote_url}")
                time.sleep(5 * retries)  # Exponential backoff
                
            success = process_repository(git_remote_url, model)
            retries += 1
            
        if success:
            logger.info(f"Finished processing repository {index}/{total_repos}: {git_remote_url}")
        else:
            logger.error(f"Failed to process repository after {max_retries} attempts: {git_remote_url}")

    logger.info("All repositories have been processed.")

if __name__ == "__main__":
    # Load the list of GitHub repositories from the JSON file
    with open('data/repos.json', 'r') as json_file:
        repos_data = json.load(json_file)
        repositories = repos_data['repositories']

    process_repositories(repositories)